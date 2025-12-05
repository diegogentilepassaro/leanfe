#!/usr/bin/env Rscript
# Test continuous treatment variable warning

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
  treatment_continuous = runif(n, 0, 10),  # Continuous
  treatment_binary = rbinom(n, 1, 0.5)     # Binary
)

df <- df$with_columns(
  revenue = pl$lit(10.0) +
    pl$col("treatment_continuous") * 0.5 +
    pl$col("treatment_binary") * 1.0 +
    pl$col("customer_id")$cast(pl$Float64) * 0.01 +
    pl$lit(rnorm(n, 0, 1))
)

cat("Testing continuous treatment warnings...\n\n")

# Test 1: Continuous treatment triggers warning (Polars)
cat("Test 1: Continuous treatment warning (Polars)... ")
result <- tryCatch({
  suppressWarnings({
    fast_feols_polars(df, formula = "revenue ~ treatment_continuous | customer_id + product_id")
  })
}, warning = function(w) {
  if (grepl("Continuous treatment variable", w$message)) {
    cat("WARNING DETECTED ✓\n")
    return(suppressWarnings(
      fast_feols_polars(df, formula = "revenue ~ treatment_continuous | customer_id + product_id")
    ))
  }
})
stopifnot("treatment_continuous" %in% names(result$coefficients))
cat("PASSED\n")

# Test 2: Continuous treatment triggers warning (DuckDB)
cat("Test 2: Continuous treatment warning (DuckDB)... ")
result <- tryCatch({
  suppressWarnings({
    fast_feols_duckdb(df, formula = "revenue ~ treatment_continuous | customer_id + product_id")
  })
}, warning = function(w) {
  if (grepl("Continuous treatment variable", w$message)) {
    cat("WARNING DETECTED ✓\n")
    return(suppressWarnings(
      fast_feols_duckdb(df, formula = "revenue ~ treatment_continuous | customer_id + product_id")
    ))
  }
})
stopifnot("treatment_continuous" %in% names(result$coefficients))
cat("PASSED\n")

# Test 3: Binary treatment does NOT trigger warning
cat("Test 3: Binary treatment no warning... ")
warned <- FALSE
result <- tryCatch({
  fast_feols_polars(df, formula = "revenue ~ treatment_binary | customer_id + product_id")
}, warning = function(w) {
  if (grepl("Continuous treatment variable", w$message)) {
    warned <<- TRUE
  }
  suppressWarnings(
    fast_feols_polars(df, formula = "revenue ~ treatment_binary | customer_id + product_id")
  )
})
stopifnot(!warned)
stopifnot("treatment_binary" %in% names(result$coefficients))
cat("PASSED\n")

# Test 4: Mixed treatments trigger warning
cat("Test 4: Mixed treatments warning... ")
result <- tryCatch({
  suppressWarnings({
    fast_feols_polars(df, formula = "revenue ~ treatment_binary + treatment_continuous | customer_id + product_id")
  })
}, warning = function(w) {
  if (grepl("Continuous treatment variable", w$message)) {
    cat("WARNING DETECTED ✓\n")
    return(suppressWarnings(
      fast_feols_polars(df, formula = "revenue ~ treatment_binary + treatment_continuous | customer_id + product_id")
    ))
  }
})
stopifnot("treatment_binary" %in% names(result$coefficients))
stopifnot("treatment_continuous" %in% names(result$coefficients))
cat("PASSED\n")

# Test 5: Categorical treatment (few unique values) does NOT trigger warning
cat("Test 5: Categorical treatment no warning... ")
df_cat <- pl$DataFrame(
  customer_id = rep(1:100, each = 10),
  product_id = rep(1:50, times = 20),
  treatment_categorical = sample(c(0.0, 1.0, 2.0), n, replace = TRUE)
)
df_cat <- df_cat$with_columns(
  revenue = pl$lit(10.0) +
    pl$col("treatment_categorical") * 1.0 +
    pl$col("customer_id")$cast(pl$Float64) * 0.01 +
    pl$lit(rnorm(n, 0, 1))
)

warned <- FALSE
result <- tryCatch({
  fast_feols_polars(df_cat, formula = "revenue ~ treatment_categorical | customer_id + product_id")
}, warning = function(w) {
  if (grepl("Continuous treatment variable", w$message)) {
    warned <<- TRUE
  }
  suppressWarnings(
    fast_feols_polars(df_cat, formula = "revenue ~ treatment_categorical | customer_id + product_id")
  )
})
stopifnot(!warned)
stopifnot("treatment_categorical" %in% names(result$coefficients))
cat("PASSED\n")

cat("\n✅ All 5 tests passed!\n")
