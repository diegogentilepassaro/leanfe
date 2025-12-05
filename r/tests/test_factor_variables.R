#!/usr/bin/env Rscript
# Test factor variable expansion

library(polars)
library(duckdb)
library(DBI)
source("R/common.R")
source("R/polars.R")
source("R/duckdb.R")

set.seed(42)
n <- 1000

cat("Testing factor variable expansion...\n\n")

# Test 1: Factor variables with Polars
cat("Test 1: Factor variables (Polars)... ")
df <- pl$DataFrame(
  y = rnorm(n),
  x = rnorm(n),
  region = sample(c("A", "B", "C"), n, replace = TRUE),
  fe1 = sample(0:9, n, replace = TRUE),
  fe2 = sample(0:4, n, replace = TRUE)
)

result <- fast_feols_polars(df, formula = "y ~ x + i(region) | fe1 + fe2")

# Should have coefficient for x plus 2 region dummies (A is reference)
stopifnot("x" %in% names(result$coefficients))
stopifnot("region_B" %in% names(result$coefficients))
stopifnot("region_C" %in% names(result$coefficients))
stopifnot(length(result$coefficients) == 3)  # x + 2 region dummies
cat("PASSED\n")
cat(sprintf("  Coefficients: %s\n", paste(names(result$coefficients), collapse = ", ")))
cat("  (region_A is reference category)\n")

# Test 2: Factor variables with DuckDB
cat("Test 2: Factor variables (DuckDB)... ")
result <- fast_feols_duckdb(df, formula = "y ~ x + i(region) | fe1 + fe2")

stopifnot("x" %in% names(result$coefficients))
stopifnot("region_B" %in% names(result$coefficients))
stopifnot("region_C" %in% names(result$coefficients))
stopifnot(length(result$coefficients) == 3)
cat("PASSED\n")
cat(sprintf("  Coefficients: %s\n", paste(names(result$coefficients), collapse = ", ")))
cat("  (region_A is reference category)\n")

cat("\n✅ All 2 tests passed!\n")
