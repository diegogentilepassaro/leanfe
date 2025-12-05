# leanfe: Lean Fixed Effects Regression

Lean, fast fixed effects regression using Polars and DuckDB. Optimized for speed and memory efficiency.

## Performance

Benchmarked on 12.7M observations with 4 high-dimensional fixed effects:

| Backend | Time (IID) | Time (Clustered) | Memory |
|---------|------------|------------------|--------|
| **Polars** | **15.1s** | **12.2s** | **986 MB** |
| **DuckDB** | 18.3s | 20.7s | **714 MB** |
| fixest | 11.0s | 10.9s | 2,944 MB |

**Key wins:**
- **3x less memory than fixest** with similar speed
- **DuckDB uses 4x less memory** than fixest
- Both backends handle clustered SEs efficiently

## Installation

```r
# Install Polars from R-multiverse
install.packages("polars", repos = "https://community.r-multiverse.org")

# For DuckDB backend (optional but recommended)
install.packages("duckdb")
```

## Quick Start

```r
source("R/common.R")
source("R/polars.R")
source("R/duckdb.R")
source("R/fast_feols.R")
library(polars)

df <- pl$read_parquet("data.parquet")

# Default: uses Polars backend (fastest)
result <- fast_feols(df, formula = "revenue ~ treatment | customer_id + product_id + day")

# For very large datasets: use DuckDB backend (lowest memory)
result <- fast_feols(
  df, 
  formula = "revenue ~ treatment | customer_id + product_id + day",
  backend = "duckdb"
)

cat(sprintf("Coefficient: %.4f\n", result$coefficients["treatment"]))
cat(sprintf("Std Error: %.4f\n", result$std_errors["treatment"]))
cat(sprintf("N obs: %d\n", result$n_obs))
```

## Choosing a Backend

The `backend` parameter controls which implementation is used:

| Backend | Speed | Memory | Best For |
|---------|-------|--------|----------|
| `"polars"` (default) | ⚡ Fastest | Moderate | Most use cases, data fits in RAM |
| `"duckdb"` | Fast | ⚡ Lowest | Very large datasets, memory-constrained |

**Use Polars (default) when:**
- Speed is the priority
- Data fits comfortably in memory
- You need the fastest possible execution

**Use DuckDB when:**
- Dataset is larger than available RAM
- Memory is constrained (e.g., shared servers)
- Reading directly from parquet files without loading into memory

```r
# Polars: ~15s, ~1 GB (default)
result <- fast_feols(df, formula = "y ~ x | fe1 + fe2")

# DuckDB: ~18s, ~700 MB
result <- fast_feols(df, formula = "y ~ x | fe1 + fe2", backend = "duckdb")
```

## Features

### Clustered Standard Errors

```r
# One-way clustering
result <- fast_feols(
  df, 
  formula = "y ~ x | fe1 + fe2",
  vcov = "cluster",
  cluster_cols = c("customer_id")
)

# Multi-way clustering
result <- fast_feols(
  df, 
  formula = "y ~ x | fe1 + fe2",
  vcov = "cluster",
  cluster_cols = c("customer_id", "region")
)
```

### Factor Variables

```r
# Expand categorical variable into dummies (first category is reference by default)
result <- fast_feols(df, formula = "y ~ treatment + i(region) | customer + product")
# Creates: treatment, region_B, region_C (region_A is reference)

# Specify custom reference category
result <- fast_feols(df, formula = "y ~ treatment + i(region, ref=B) | customer + product")
# Creates: treatment, region_A, region_C (region_B is reference)
```

### Interaction Terms

```r
# Heterogeneous treatment effects by region
result <- fast_feols(df, formula = "y ~ treatment:i(region) | customer + product")
# Creates: treatment_B, treatment_C (treatment effect for each region vs. reference A)

# With custom reference category
result <- fast_feols(df, formula = "y ~ treatment:i(region, ref=B) | customer + product")
# Creates: treatment_A, treatment_C (treatment effect relative to region B)
```

### Instrumental Variables (IV/2SLS)

```r
# Two-stage least squares
result <- fast_feols(df, formula = "y ~ x | fe1 + fe2 | z1 + z2")
# x is endogenous, z1 and z2 are instruments
```

### Weighted Regression

```r
result <- fast_feols(df, formula = "y ~ x | fe1 + fe2", weights = "weight_col")
```

### Difference-in-Differences

```r
# Classic TWFE DiD
result <- fast_feols(
  df, 
  formula = "y ~ treated_post | state + year",
  vcov = "cluster", 
  cluster_cols = c("state")
)

# Event study
result <- fast_feols(
  df,
  formula = "y ~ lead2 + lead1 + lag0 + lag1 + lag2 | state + year", 
  vcov = "cluster", 
  cluster_cols = c("state")
)
```

## API Reference

### fast_feols()

```r
fast_feols(
  data,
  formula,
  vcov = "iid",              # "iid", "HC1", or "cluster"
  cluster_cols = NULL,
  weights = NULL,
  demean_tol = 1e-3,
  n_iter = 100,
  sample_frac = NULL,
  backend = "polars"         # "polars" or "duckdb"
)
```

**Parameters:**
- `data`: Polars DataFrame, R data.frame, or path to parquet file
- `formula`: R-style formula (see syntax below)
- `vcov`: Variance estimator - "iid", "HC1", or "cluster"
- `cluster_cols`: Clustering variables for clustered SEs
- `weights`: Column name for regression weights
- `backend`: "polars" (fast, default) or "duckdb" (low memory)

**Returns:**
```r
list(
  coefficients = c(treatment = 0.1234, ...),
  std_errors = c(treatment = 0.0567, ...),
  n_obs = 12722538,
  iterations = 6,
  vcov_type = "cluster",
  n_clusters = 5040
)
```

### Backend-Specific Functions

For advanced use cases, you can also use the backend-specific functions directly:

```r
source("R/polars.R")
source("R/duckdb.R")

# These have the same API as fast_feols() minus the backend parameter
result <- fast_feols_polars(df, formula = "y ~ x | fe")
result <- fast_feols_duckdb(df, formula = "y ~ x | fe")
```

## Formula Syntax

```
y ~ x1 + x2 + i(cat_var) + x:i(cat_var) | fe1 + fe2 | z1 + z2
│   │                                     │            │
│   └─ Treatment variables                │            └─ Instruments (IV)
│      - Regular: x1, x2                  │
│      - Factor: i(cat_var)               └─ Fixed effects
│      - Interaction: x:i(cat_var)
└─ Outcome variable
```

## Comparison with fixest

| Feature | leanfe | fixest |
|---------|------------|--------|
| Speed | Similar | Similar |
| Memory | 3x lower | Higher |
| Clustered SEs | Fast | Fast |
| Formula syntax | Compatible | Native |
| Backend choice | Polars/DuckDB | C++ |

## Algorithm

Uses the Frisch-Waugh-Lovell (FWL) theorem:
1. Iteratively demean variables within each fixed effect group
2. Converge when group means < tolerance (typically 4-6 iterations)
3. OLS on fully demeaned data
4. Adjust standard errors for absorbed degrees of freedom

**Memory-efficient clustered SEs:** Computes cluster scores via GROUP BY aggregation, avoiding N×K matrix materialization.

## License

MIT License
