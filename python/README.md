# leanfe: Lean Fixed Effects Regression

Lean, fast fixed effects regression using Polars and DuckDB. Optimized for speed and memory efficiency.

## Performance

Benchmarked on 12.7M observations with 4 high-dimensional fixed effects:

| Backend | Time (IID) | Time (Clustered) | Memory |
|---------|------------|------------------|--------|
| **Polars** | **5.8s** | **5.5s** | **291 MB** |
| **DuckDB** | 18.4s | 18.2s | **27 MB** |
| PyFixest | 91.2s | timeout | 6,691 MB |

**Key wins:**
- **16x faster than PyFixest** with IID standard errors
- **23x less memory** than PyFixest
- **DuckDB uses <30 MB** - ideal for datasets larger than RAM

## Installation

```bash
pip install polars pyarrow numpy

# For DuckDB backend (optional but recommended)
pip install duckdb

# Install package
cd python
pip install -e .
```

## Quick Start

```python
from leanfe import fast_feols
import polars as pl

df = pl.read_parquet("data.parquet")

# Default: uses Polars backend (fastest)
result = fast_feols(df, formula="revenue ~ treatment | customer_id + product_id + day")

# For very large datasets: use DuckDB backend (lowest memory)
result = fast_feols(
    df, 
    formula="revenue ~ treatment | customer_id + product_id + day",
    backend="duckdb"
)

print(f"Coefficient: {result['coefficients']['treatment']:.4f}")
print(f"Std Error: {result['std_errors']['treatment']:.4f}")
print(f"N obs: {result['n_obs']:,}")
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

```python
# Polars: ~6s, ~300 MB (default)
result = fast_feols(df, formula="y ~ x | fe1 + fe2")

# DuckDB: ~18s, ~30 MB
result = fast_feols(df, formula="y ~ x | fe1 + fe2", backend="duckdb")
```

## Features

### Clustered Standard Errors

```python
# One-way clustering
result = fast_feols(
    df, 
    formula="y ~ x | fe1 + fe2",
    vcov="cluster",
    cluster_cols=["customer_id"]
)

# Multi-way clustering
result = fast_feols(
    df, 
    formula="y ~ x | fe1 + fe2",
    vcov="cluster",
    cluster_cols=["customer_id", "region"]
)
```

### Factor Variables

```python
# Expand categorical variable into dummies (first category is reference by default)
result = fast_feols(df, formula="y ~ treatment + i(region) | customer + product")
# Creates: treatment, region_B, region_C (region_A is reference)

# Specify custom reference category
result = fast_feols(df, formula="y ~ treatment + i(region, ref=B) | customer + product")
# Creates: treatment, region_A, region_C (region_B is reference)
```

### Interaction Terms

```python
# Heterogeneous treatment effects by region
result = fast_feols(df, formula="y ~ treatment:i(region) | customer + product")
# Creates: treatment_B, treatment_C (treatment effect for each region vs. reference A)

# With custom reference category
result = fast_feols(df, formula="y ~ treatment:i(region, ref=B) | customer + product")
# Creates: treatment_A, treatment_C (treatment effect relative to region B)
```

### Instrumental Variables (IV/2SLS)

```python
# Two-stage least squares
result = fast_feols(df, formula="y ~ x | fe1 + fe2 | z1 + z2")
# x is endogenous, z1 and z2 are instruments
```

### Weighted Regression

```python
result = fast_feols(df, formula="y ~ x | fe1 + fe2", weights="weight_col")
```

### Difference-in-Differences

```python
# Classic TWFE DiD
result = fast_feols(
    df, 
    formula="y ~ treated_post | state + year",
    vcov="cluster", 
    cluster_cols=["state"]
)

# Event study
result = fast_feols(
    df,
    formula="y ~ lead2 + lead1 + lag0 + lag1 + lag2 | state + year", 
    vcov="cluster", 
    cluster_cols=["state"]
)
```

## API Reference

### fast_feols()

```python
fast_feols(
    data: Union[str, pl.DataFrame],
    formula: str,
    vcov: str = "iid",              # "iid", "HC1", or "cluster"
    cluster_cols: List[str] = None,
    weights: str = None,
    demean_tol: float = 1e-3,
    n_iter: int = 100,
    sample_frac: float = None,
    backend: str = "polars"         # "polars" or "duckdb"
) -> dict
```

**Parameters:**
- `data`: Polars DataFrame or path to parquet file
- `formula`: R-style formula (see syntax below)
- `vcov`: Variance estimator - "iid", "HC1", or "cluster"
- `cluster_cols`: Clustering variables for clustered SEs
- `weights`: Column name for regression weights
- `backend`: "polars" (fast, default) or "duckdb" (low memory)

**Returns:**
```python
{
    'coefficients': {'treatment': 0.1234, ...},
    'std_errors': {'treatment': 0.0567, ...},
    'n_obs': 12722538,
    'iterations': 6,
    'vcov_type': 'cluster',
    'n_clusters': 5040,
    'is_iv': False,
    'n_instruments': None
}
```

### Backend-Specific Functions

For advanced use cases, you can also import the backend-specific functions directly:

```python
from leanfe import fast_feols_polars, fast_feols_duckdb

# These have the same API as fast_feols() minus the backend parameter
result = fast_feols_polars(df, formula="y ~ x | fe")
result = fast_feols_duckdb(df, formula="y ~ x | fe")
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

## Algorithm

Uses the Frisch-Waugh-Lovell (FWL) theorem:
1. Iteratively demean variables within each fixed effect group
2. Converge when group means < tolerance (typically 4-6 iterations)
3. OLS on fully demeaned data
4. Adjust standard errors for absorbed degrees of freedom

**Memory-efficient clustered SEs:** Computes cluster scores via GROUP BY aggregation, avoiding N×K matrix materialization.

## License

MIT License
