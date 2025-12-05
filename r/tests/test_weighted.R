#!/usr/bin/env Rscript
# Test weighted regression functionality

library(polars)
library(duckdb)
library(DBI)
source("R/common.R")
source("R/polars.R")
source("R/duckdb.R")

set.seed(42)
n <- 1000

cat("Testing weighted regression...\n\n")

# Test 1: Basic weighted regression
cat("Test 1: Basic weighted regression... ")
x <- rnorm(n)
fe1 <- sample(0:9, n, replace = TRUE)
weights <- runif(n, 0.5, 2.0)
y <- 2.0 * x + rnorm(n)

df <- pl$DataFrame(
  y = y,
  x = x,
  fe1 = fe1,
  weights = weights
)

result <- fast_feols_polars(df, y_col = "y", x_cols = "x", fe_cols = "fe1", weights = "weights")

stopifnot("coefficients" %in% names(result))
stopifnot("x" %in% names(result$coefficients))
stopifnot(abs(result$coefficients["x"] - 2.0) < 0.2)  # Close to true value
cat("PASSED\n")

# Test 2: Weighted regression with formula API
cat("Test 2: Weighted regression with formula... ")
df <- pl$DataFrame(
  y = rnorm(n),
  x = rnorm(n),
  fe1 = sample(0:9, n, replace = TRUE),
  weights = runif(n, 0.5, 2.0)
)

result <- fast_feols_polars(df, formula = "y ~ x | fe1", weights = "weights")

stopifnot("coefficients" %in% names(result))
stopifnot("x" %in% names(result$coefficients))
cat("PASSED\n")

# Test 3: Weighted regression with HC1 standard errors
cat("Test 3: Weighted regression with HC1 SE... ")
result <- fast_feols_polars(
  df, 
  y_col = "y", 
  x_cols = "x", 
  fe_cols = "fe1",
  weights = "weights", 
  vcov = "HC1"
)

stopifnot(result$vcov_type == "HC1")
stopifnot("std_errors" %in% names(result))
cat("PASSED\n")

cat("\n✅ All 3 tests passed!\n")
