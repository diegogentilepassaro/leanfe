#!/usr/bin/env Rscript
# Test binary treatment × continuous variable interactions

library(polars)
library(duckdb)
library(DBI)
source("R/common.R")
source("R/polars.R")
source("R/duckdb.R")

set.seed(42)
n <- 1000

cat("Testing binary × continuous interactions...\n\n")

# Generate test data with binary treatment × continuous variable
df <- pl$DataFrame(
  customer_id = rep(0:99, each = 10),
  product_id = rep(0:49, 20),
  treatment = rbinom(n, 1, 0.5),
  price = runif(n, 10, 100)  # Continuous moderator
)

# Generate outcome with interaction effect
# True model: y = 10 + 2*treatment + 0.5*price + 1.5*treatment*price + FE + noise
df <- df$with_columns(
  revenue = pl$lit(10.0) +
    pl$lit(2.0) * pl$col("treatment") +
    pl$lit(0.5) * pl$col("price") +
    pl$lit(1.5) * pl$col("treatment") * pl$col("price") +
    pl$col("customer_id")$cast(pl$Float64) * 0.01 +
    pl$lit(rnorm(n, 0, 1))
)

# Test 1: Binary × continuous interaction (Polars)
cat("Test 1: Binary × continuous interaction (Polars)... ")

# Create interaction manually
df_interact <- df$with_columns(
  treatment_x_price = pl$col("treatment") * pl$col("price")
)

result <- suppressWarnings(fast_feols_polars(
  df_interact,
  formula = "revenue ~ treatment + price + treatment_x_price | customer_id + product_id"
))

# Check all coefficients exist
stopifnot("treatment" %in% names(result$coefficients))
stopifnot("price" %in% names(result$coefficients))
stopifnot("treatment_x_price" %in% names(result$coefficients))

# Check coefficients are close to true values (within 0.5)
stopifnot(abs(result$coefficients["treatment"] - 2.0) < 0.5)
stopifnot(abs(result$coefficients["price"] - 0.5) < 0.5)
stopifnot(abs(result$coefficients["treatment_x_price"] - 1.5) < 0.5)
cat("PASSED\n")

# Test 2: Binary × continuous interaction (DuckDB)
cat("Test 2: Binary × continuous interaction (DuckDB)... ")

result <- suppressWarnings(fast_feols_duckdb(
  df_interact,
  formula = "revenue ~ treatment + price + treatment_x_price | customer_id + product_id"
))

stopifnot("treatment" %in% names(result$coefficients))
stopifnot("price" %in% names(result$coefficients))
stopifnot("treatment_x_price" %in% names(result$coefficients))

stopifnot(abs(result$coefficients["treatment"] - 2.0) < 0.5)
stopifnot(abs(result$coefficients["price"] - 0.5) < 0.5)
stopifnot(abs(result$coefficients["treatment_x_price"] - 1.5) < 0.5)
cat("PASSED\n")

# Test 3: Continuous main effect only
cat("Test 3: Continuous main effect only... ")

result <- suppressWarnings(fast_feols_polars(
  df,
  formula = "revenue ~ price | customer_id + product_id"
))

stopifnot("price" %in% names(result$coefficients))
stopifnot(result$coefficients["price"] > 0)
cat("PASSED\n")

# Test 4: Multiple continuous variables
cat("Test 4: Multiple continuous variables... ")

df_multi <- pl$DataFrame(
  customer_id = rep(0:99, each = 10),
  product_id = rep(0:49, 20),
  price = runif(n, 10, 100),
  quantity = runif(n, 1, 10)
)

df_multi <- df_multi$with_columns(
  revenue = pl$lit(10.0) +
    pl$lit(0.5) * pl$col("price") +
    pl$lit(2.0) * pl$col("quantity") +
    pl$col("customer_id")$cast(pl$Float64) * 0.01 +
    pl$lit(rnorm(n, 0, 1))
)

result <- suppressWarnings(fast_feols_polars(
  df_multi,
  formula = "revenue ~ price + quantity | customer_id + product_id"
))

stopifnot("price" %in% names(result$coefficients))
stopifnot("quantity" %in% names(result$coefficients))
stopifnot(abs(result$coefficients["price"] - 0.5) < 0.3)
stopifnot(abs(result$coefficients["quantity"] - 2.0) < 0.3)
cat("PASSED\n")

cat("\n✅ All 4 tests passed!\n")
