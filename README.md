# 🍃💨 leanfe

**Lean, Fast Fixed Effects Regression for Python and R**

[![Documentation](https://img.shields.io/badge/docs-online-blue)](https://diegogentilepassaro.github.io/leanfe/)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![R](https://img.shields.io/badge/R-4.0+-blue.svg)](https://www.r-project.org/)

High-dimensional fixed effects regression that's **fast** when you need speed, and **memory-efficient** when you're working with data larger than RAM.

## Key Features

- ⚡ **Polars backend** - Blazing fast in-memory computation
- 💾 **DuckDB backend** - Process datasets larger than RAM
- 🔄 **Unified API** - Same syntax in Python and R
- 📊 **Full econometrics toolkit** - Clustered SEs, factor variables, IV/2SLS
- 🚀 **YOCO compression** - [Wong et al. (2021)](https://arxiv.org/abs/2102.11297) optimal data compression with sparse matrices for blazing fast clustered standard errors

## Quick Start

### Python

```python
from leanfe import leanfe

result = leanfe(
    formula="outcome ~ treatment + controls | unit_id + time_id",
    data=df,
    vcov="cluster",
    cluster_cols=["unit_id"],
    backend="polars"  # or "duckdb" for large datasets
)
```

### R

```r
library(leanfe)

result <- leanfe(
    formula = "outcome ~ treatment + controls | unit_id + time_id",
    data = df,
    vcov = "cluster",
    cluster_cols = c("unit_id"),
    backend = "polars"
)
```

## Installation

### Python

```bash
pip install git+https://github.com/diegogentilepassaro/leanfe.git#subdirectory=python
```

### R

```r
remotes::install_github("diegogentilepassaro/leanfe", subdir = "r")
```

## Documentation

📖 **[Full Documentation](https://diegogentilepassaro.github.io/leanfe/)**

- [Get Started](https://diegogentilepassaro.github.io/leanfe/get-started.html)
- [Tutorials](https://diegogentilepassaro.github.io/leanfe/tutorials/basic-usage.html)
- [API Reference](https://diegogentilepassaro.github.io/leanfe/reference/python.html)
- [Benchmarks](https://diegogentilepassaro.github.io/leanfe/benchmarks/overview.html)

## Performance

**Two backends optimized for different scenarios:**

| Backend | Best For | Trade-off |
|---------|----------|-----------|
| **Polars** | Maximum speed | Higher memory usage |
| **DuckDB** | Large datasets | Slightly slower, minimal memory |

Both backends use YOCO compression + sparse matrices for all SE types (IID, HC1, clustered).

**Tested up to 50M observations** — see [live benchmarks](https://diegogentilepassaro.github.io/leanfe/benchmarks/overview.html) for current performance numbers.

## License

MIT
