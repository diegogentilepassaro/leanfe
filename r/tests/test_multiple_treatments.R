#!/usr/bin/env Rscript
# Test multiple treatment arms functionality

library(polars)
library(duckdb)
library(DBI)
source("R/common.R")
source("R/polars.R")
source("R/duckdb.R")

# Generate test data
set.seed(42)
n <- 10000

df <- pl$DataFrame(
  customer_id = rep(1:1000, each = 10),
  product_id = rep(1:100, times = 100),
  region = sample(c("R1", "R2", "R3"), n, replace = TRUE),
  treatment_A = rbinom(n, 1, 0.3),
  treatment_B = rbinom(n, 1, 0.4),
  treatment_C = sample(0:2, n, replace = TRUE)
)

# Generate outcome with known effects
df <- df$with_columns(
  revenue = pl$lit(10.0) +
    pl$col("treatment_A") * 0.5 +
    pl$col("treatment_B") * 0.8 +
    pl$col("treatment_C") * 1.2 +
    pl$col("customer_id")$cast(pl$Float64) * 0.01 +
    pl$col("product_id")$cast(pl$Float64) * 0.02 +
    pl$lit(rnorm(n, 0, 1))
)

cat("Testing multiple treatment arms...\n\n")

# Test 1: Multiple binary treatments (Polars)
cat("Test 1: Multiple binary treatments (Polars)... ")
result <- fast_feols_polars(
  df,
  formula = "revenue ~ treatment_A + treatment_B | customer_id + product_id"
)
stopifnot("treatment_A" %in% names(result$coefficients))
stopifnot("treatment_B" %in% names(result$coefficients))
stopifnot(abs(result$coefficients["treatment_A"] - 0.5) < 0.3)
stopifnot(abs(result$coefficients["treatment_B"] - 0.8) < 0.3)
cat("PASSED\n")

# Test 2: Multiple binary treatments (DuckDB)
cat("Test 2: Multiple binary treatments (DuckDB)... ")
result <- fast_feols_duckdb(
  df,
  formula = "revenue ~ treatment_A + treatment_B | customer_id + product_id"
)
stopifnot("treatment_A" %in% names(result$coefficients))
stopifnot("treatment_B" %in% names(result$coefficients))
stopifnot(abs(result$coefficients["treatment_A"] - 0.5) < 0.3)
stopifnot(abs(result$coefficients["treatment_B"] - 0.8) < 0.3)
cat("PASSED\n")

# Test 3: Mixed treatment types (Polars)
cat("Test 3: Mixed treatment types (Polars)... ")
result <- fast_feols_polars(
  df,
  formula = "revenue ~ treatment_A + treatment_B + treatment_C | customer_id + product_id"
)
stopifnot("treatment_A" %in% names(result$coefficients))
stopifnot("treatment_B" %in% names(result$coefficients))
stopifnot("treatment_C" %in% names(result$coefficients))
stopifnot(abs(result$coefficients["treatment_A"] - 0.5) < 0.3)
stopifnot(abs(result$coefficients["treatment_B"] - 0.8) < 0.3)
stopifnot(abs(result$coefficients["treatment_C"] - 1.2) < 0.3)
cat("PASSED\n")

# Test 4: Multiple interactions (Polars)
cat("Test 4: Multiple interactions (Polars)... ")
result <- fast_feols_polars(
  df,
  formula = "revenue ~ treatment_A:i(region) + treatment_B:i(region) | customer_id + product_id"
)
treatment_a_coeffs <- grep("^treatment_A_", names(result$coefficients), value = TRUE)
treatment_b_coeffs <- grep("^treatment_B_", names(result$coefficients), value = TRUE)
stopifnot(length(treatment_a_coeffs) == 2)  # 3 regions - 1 reference = 2
stopifnot(length(treatment_b_coeffs) == 2)  # 3 regions - 1 reference = 2
cat("PASSED\n")

# Test 5: Multiple interactions (DuckDB)
cat("Test 5: Multiple interactions (DuckDB)... ")
result <- fast_feols_duckdb(
  df,
  formula = "revenue ~ treatment_A:i(region) + treatment_B:i(region) | customer_id + product_id"
)
treatment_a_coeffs <- grep("^treatment_A_", names(result$coefficients), value = TRUE)
treatment_b_coeffs <- grep("^treatment_B_", names(result$coefficients), value = TRUE)
stopifnot(length(treatment_a_coeffs) == 2)  # 3 regions - 1 reference = 2
stopifnot(length(treatment_b_coeffs) == 2)  # 3 regions - 1 reference = 2
cat("PASSED\n")

# Test 6: Multiple treatments with robust SE
cat("Test 6: Multiple treatments with robust SE... ")
result <- fast_feols_polars(
  df,
  formula = "revenue ~ treatment_A + treatment_B | customer_id + product_id",
  vcov = "HC1"
)
stopifnot(result$vcov_type == "HC1")
stopifnot("treatment_A" %in% names(result$coefficients))
stopifnot("treatment_B" %in% names(result$coefficients))
stopifnot(result$std_errors["treatment_A"] > 0)
stopifnot(result$std_errors["treatment_B"] > 0)
cat("PASSED\n")

# Test 7: Multiple treatments with clustered SE
cat("Test 7: Multiple treatments with clustered SE... ")
result <- fast_feols_polars(
  df,
  formula = "revenue ~ treatment_A + treatment_B | customer_id + product_id",
  vcov = "cluster",
  cluster_cols = "region"
)
stopifnot(result$vcov_type == "cluster")
stopifnot(result$n_clusters == 3)
stopifnot("treatment_A" %in% names(result$coefficients))
stopifnot("treatment_B" %in% names(result$coefficients))
cat("PASSED\n")

# Test 8: Multiple interactions with different factors
cat("Test 8: Multiple interactions with different factors... ")
df_time <- df$with_columns(
  time_period = (pl$col("customer_id") %% 5)$cast(pl$String)
)
result <- fast_feols_polars(
  df_time,
  formula = "revenue ~ treatment_A:i(region) + treatment_B:i(time_period) | customer_id + product_id"
)
treatment_a_coeffs <- grep("^treatment_A_", names(result$coefficients), value = TRUE)
treatment_b_coeffs <- grep("^treatment_B_", names(result$coefficients), value = TRUE)
stopifnot(length(treatment_a_coeffs) == 2)  # 3 regions - 1 reference = 2
stopifnot(length(treatment_b_coeffs) == 4)  # 5 time periods - 1 reference = 4
cat("PASSED\n")

cat("\n✅ All 8 tests passed!\n")
