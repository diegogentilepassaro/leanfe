#!/usr/bin/env Rscript
# Basic tests for fast-hdfe-reg package

library(polars)
library(duckdb)
library(DBI)
source("R/common.R")
source("R/polars.R")
source("R/duckdb.R")
source("R/fast_feols.R")

set.seed(42)
n <- 1000

cat("Testing basic functionality...\n\n")

# Test 1: Basic Polars functionality
cat("Test 1: Basic Polars functionality... ")
df <- pl$DataFrame(
  y = rnorm(n),
  x = rnorm(n),
  fe1 = sample(0:9, n, replace = TRUE),
  fe2 = sample(0:4, n, replace = TRUE)
)

result <- fast_feols_polars(df, y_col = "y", x_cols = "x", fe_cols = c("fe1", "fe2"))

stopifnot("coefficients" %in% names(result))
stopifnot("std_errors" %in% names(result))
stopifnot("x" %in% names(result$coefficients))
stopifnot(result$n_obs > 0)
cat("PASSED\n")

# Test 2: Formula API
cat("Test 2: Formula API... ")
result <- fast_feols_polars(df, formula = "y ~ x | fe1 + fe2")

stopifnot("coefficients" %in% names(result))
stopifnot("x" %in% names(result$coefficients))
cat("PASSED\n")

# Test 3: Standard errors (IID, HC1, Clustered)
cat("Test 3: Standard errors... ")
df_cluster <- pl$DataFrame(
  y = rnorm(n),
  x = rnorm(n),
  fe1 = sample(0:9, n, replace = TRUE),
  cluster = sample(0:19, n, replace = TRUE)
)

# IID
result_iid <- fast_feols_polars(df_cluster, y_col = "y", x_cols = "x", fe_cols = "fe1", vcov = "iid")
stopifnot(result_iid$vcov_type == "iid")

# HC1
result_hc1 <- fast_feols_polars(df_cluster, y_col = "y", x_cols = "x", fe_cols = "fe1", vcov = "HC1")
stopifnot(result_hc1$vcov_type == "HC1")

# Clustered
result_cluster <- fast_feols_polars(
  df_cluster, 
  y_col = "y", 
  x_cols = "x", 
  fe_cols = "fe1",
  vcov = "cluster", 
  cluster_cols = "cluster"
)
stopifnot(result_cluster$vcov_type == "cluster")
stopifnot(result_cluster$n_clusters > 0)
cat("PASSED\n")

# Test 4: Unified API with Polars backend
cat("Test 4: Unified API (backend='polars')... ")
result_unified <- fast_feols(df, formula = "y ~ x | fe1 + fe2", backend = "polars")
stopifnot("x" %in% names(result_unified$coefficients))
cat("PASSED\n")

# Test 5: Unified API with DuckDB backend
cat("Test 5: Unified API (backend='duckdb')... ")
result_duckdb <- fast_feols(df, formula = "y ~ x | fe1 + fe2", backend = "duckdb")
stopifnot("x" %in% names(result_duckdb$coefficients))
cat("PASSED\n")

# Test 6: Unified API default backend (polars)
cat("Test 6: Unified API (default backend)... ")
result_default <- fast_feols(df, formula = "y ~ x | fe1 + fe2")
stopifnot("x" %in% names(result_default$coefficients))
cat("PASSED\n")

# Test 7: Unified API invalid backend raises error
cat("Test 7: Unified API (invalid backend)... ")
error_caught <- FALSE
tryCatch({
  fast_feols(df, formula = "y ~ x | fe1", backend = "invalid")
}, error = function(e) {
  if (grepl("backend must be", e$message)) {
    error_caught <<- TRUE
  }
})
stopifnot(error_caught)
cat("PASSED\n")

cat("\n✅ All 7 tests passed!\n")
