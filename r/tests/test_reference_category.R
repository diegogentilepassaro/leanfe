#!/usr/bin/env Rscript
# Test reference category selection in factor variables and interactions

library(polars)
library(duckdb)
library(DBI)
source("R/common.R")
source("R/polars.R")
source("R/duckdb.R")

# Generate test data
set.seed(42)
n <- 1000

df <- pl$DataFrame(
  customer_id = rep(1:100, each = 10),
  product_id = rep(1:50, times = 20),
  region = sample(c("R1", "R2", "R3"), n, replace = TRUE),
  treatment = rbinom(n, 1, 0.5)
)

df <- df$with_columns(
  revenue = pl$lit(10.0) +
    pl$col("treatment") * 0.5 +
    pl$col("customer_id")$cast(pl$Float64) * 0.01 +
    pl$lit(rnorm(n, 0, 1))
)

cat("Testing reference category selection...\n\n")

# Test 1: Default reference (first category)
cat("Test 1: Default reference is first category (Polars)... ")
result <- fast_feols_polars(df, formula = "revenue ~ treatment + i(region) | customer_id + product_id")
stopifnot("region_R2" %in% names(result$coefficients))
stopifnot("region_R3" %in% names(result$coefficients))
stopifnot(!("region_R1" %in% names(result$coefficients)))  # R1 is reference
cat("PASSED\n")

# Test 2: Custom reference (Polars)
cat("Test 2: Custom reference category (Polars)... ")
result <- fast_feols_polars(df, formula = "revenue ~ treatment + i(region, ref=R2) | customer_id + product_id")
stopifnot("region_R1" %in% names(result$coefficients))
stopifnot("region_R3" %in% names(result$coefficients))
stopifnot(!("region_R2" %in% names(result$coefficients)))  # R2 is reference
cat("PASSED\n")

# Test 3: Custom reference (DuckDB)
cat("Test 3: Custom reference category (DuckDB)... ")
result <- fast_feols_duckdb(df, formula = "revenue ~ treatment + i(region, ref=R3) | customer_id + product_id")
stopifnot("region_R1" %in% names(result$coefficients))
stopifnot("region_R2" %in% names(result$coefficients))
stopifnot(!("region_R3" %in% names(result$coefficients)))  # R3 is reference
cat("PASSED\n")

# Test 4: Interaction with custom reference (Polars)
cat("Test 4: Interaction with custom reference (Polars)... ")
result <- fast_feols_polars(df, formula = "revenue ~ treatment:i(region, ref=R2) | customer_id + product_id")
stopifnot("treatment_R1" %in% names(result$coefficients))
stopifnot("treatment_R3" %in% names(result$coefficients))
stopifnot(!("treatment_R2" %in% names(result$coefficients)))  # R2 is reference
cat("PASSED\n")

# Test 5: Interaction with custom reference (DuckDB)
cat("Test 5: Interaction with custom reference (DuckDB)... ")
result <- fast_feols_duckdb(df, formula = "revenue ~ treatment:i(region, ref=R1) | customer_id + product_id")
stopifnot("treatment_R2" %in% names(result$coefficients))
stopifnot("treatment_R3" %in% names(result$coefficients))
stopifnot(!("treatment_R1" %in% names(result$coefficients)))  # R1 is reference
cat("PASSED\n")

# Test 6: Invalid reference raises error
cat("Test 6: Invalid reference raises error... ")
error_caught <- FALSE
tryCatch({
  fast_feols_polars(df, formula = "revenue ~ treatment + i(region, ref=INVALID) | customer_id + product_id")
}, error = function(e) {
  if (grepl("not found", e$message)) {
    error_caught <<- TRUE
  }
})
stopifnot(error_caught)
cat("PASSED\n")

# Test 7: Quoted reference values work
cat("Test 7: Quoted reference values... ")
result <- fast_feols_polars(df, formula = 'revenue ~ treatment + i(region, ref="R2") | customer_id + product_id')
stopifnot("region_R1" %in% names(result$coefficients))
stopifnot("region_R3" %in% names(result$coefficients))
stopifnot(!("region_R2" %in% names(result$coefficients)))  # R2 is reference
cat("PASSED\n")

cat("\n✅ All 7 tests passed!\n")
