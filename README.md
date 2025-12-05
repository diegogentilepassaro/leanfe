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

**Polars is faster, DuckDB uses far less memory.**

| Dataset Size | Backend | Run Time | Peak Memory |
|-------------|---------|----------|-------------|
| 1M obs | Polars | ~2 sec | ~150 MB |
| 1M obs | DuckDB | ~5 sec | ~50 MB |
| 10M obs | Polars | ~15 sec | ~1.2 GB |
| 10M obs | DuckDB | ~40 sec | ~150 MB |

## License

MIT
