#!/usr/bin/env Rscript
# Test interaction terms

library(polars)
library(duckdb)
library(DBI)
source("R/common.R")
source("R/polars.R")
source("R/duckdb.R")

cat("Testing interaction terms...\n\n")

# Test 1: Basic interaction (first category dropped as reference)
cat("Test 1: Basic interaction... ")
df <- pl$DataFrame(
  y = rep(c(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), 200),
  treatment = rep(c(0L, 1L, 0L, 1L, 0L, 1L), 200),
  region = rep(c("A", "A", "B", "B", "C", "C"), 200),
  fe1 = (0:1199) %% 100,
  fe2 = (0:1199) %% 50
)

result <- fast_feols_polars(df, formula = "y ~ treatment:i(region) | fe1 + fe2", vcov = "HC1")

# Verify interaction terms created (first category A is reference, dropped)
stopifnot("treatment_B" %in% names(result$coefficients))
stopifnot("treatment_C" %in% names(result$coefficients))
stopifnot(length(result$coefficients) == 2)  # B and C only, A is reference

# Verify all have standard errors
stopifnot(all(result$std_errors >= 0))
cat("PASSED\n")

# Test 2: Interaction with robust standard errors
cat("Test 2: Interaction with robust SE... ")
result <- fast_feols_polars(df, formula = "y ~ treatment:i(region) | fe1 + fe2", vcov = "HC1")

stopifnot(result$vcov_type == "HC1")
stopifnot("treatment_B" %in% names(result$std_errors))
stopifnot("treatment_C" %in% names(result$std_errors))
stopifnot(all(result$std_errors >= 0))
cat("PASSED\n")

# Test 3: Interaction with clustered standard errors
cat("Test 3: Interaction with clustered SE... ")
set.seed(42)
n <- 3000

df <- pl$DataFrame(
  treatment = rbinom(n, 1, 0.5),
  region = sample(c("A", "B", "C"), n, replace = TRUE),
  fe1 = rep(0:499, each = 6),  # 500 clusters, 6 obs each
  fe2 = rep(0:249, 12)
)

# Generate outcome with treatment effects by region
df <- df$with_columns(
  y = pl$lit(10.0) +
    (pl$col("treatment") * (pl$col("region") == "A"))$cast(pl$Float64) * 0.5 +
    (pl$col("treatment") * (pl$col("region") == "B"))$cast(pl$Float64) * 1.0 +
    (pl$col("treatment") * (pl$col("region") == "C"))$cast(pl$Float64) * 1.5 +
    pl$col("fe1")$cast(pl$Float64) * 0.01 +
    pl$lit(rnorm(n, 0, 1))
)

result <- fast_feols_polars(
  df, 
  formula = "y ~ treatment:i(region) | fe1 + fe2",
  vcov = "cluster", 
  cluster_cols = "fe1"
)

stopifnot(result$vcov_type == "cluster")
stopifnot(result$n_clusters == 500)
stopifnot(all(result$std_errors > 0))
cat("PASSED\n")

cat("\n✅ All 3 tests passed!\n")
