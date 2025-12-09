# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-12-09

### Performance Improvements

- **Vectorized clustered SEs**: All 4 implementations (Python/R × Polars/DuckDB) now use sparse matrix multiplication for clustered standard errors instead of loops. ~31x speedup for clustered SEs (464s → 15s for 5,040 clusters).

- **Smart FE ordering**: Fixed effects are now automatically sorted by cardinality (low-card first) during FWL demeaning. Low-cardinality FEs have fewer groups, making GROUP BY operations faster. ~14% speedup in demeaning phase.

- **Improved strategy selection**: Enhanced `should_use_compress()` with cost model that automatically chooses the fastest path:
  - Single FE > 10K levels → FWL demeaning (avoids huge sparse matrix)
  - Total FE levels > 20K → FWL demeaning
  - Otherwise → YOCO compression
  - Users don't need to configure anything — leanfe picks the optimal strategy based on data characteristics.

## [0.2.0] - 2025-12-05

### Changed

- **Renamed API**: All functions renamed from `fast_feols*` to `leanfe*`
  - `fast_feols()` → `leanfe()`
  - `fast_feols_polars()` → `leanfe_polars()`
  - `fast_feols_duckdb()` → `leanfe_duckdb()`
- **Formatted output**: Results now display as a formatted table (like fixest/statsmodels)
  - Shows formula, observations, fixed effects, R² (within)
  - Displays coefficients with t-statistics, p-values, and significance stars
  - New `LeanFEResult` class with methods: `coef()`, `se()`, `tstat()`, `pvalue()`, `confint()`
  - Backwards compatible: still supports dict-like access (`result['coefficients']`)

### Removed

- Removed continuous regressor warning (was not useful - cannot distinguish treatment from control variables)

## [0.1.0] - 2025-12-03

### Added

- Initial release of leanfe
- **Unified API**: `leanfe()` function with `backend` parameter ("polars" or "duckdb")
- **Polars backend**: Optimized for speed (~16x faster than PyFixest)
- **DuckDB backend**: Optimized for memory efficiency (~100x less memory)
- **Formula syntax**: R-style formulas like `"y ~ x1 + x2 | fe1 + fe2"`
- **Factor variables**: `i(region)` syntax for automatic dummy expansion
- **Custom reference categories**: `i(region, ref=R2)` syntax (like fixest)
- **Interaction terms**: `treatment:i(region)` for heterogeneous effects
- **IV/2SLS**: Instrumental variables via `"y ~ x | fe | z"` syntax
- **Standard errors**: IID, HC1 (robust), and clustered (one-way and multi-way)
- **Weighted regression**: WLS via `weights` parameter
- **Python package**: `leanfe` with full test coverage
- **R package**: `leanfe` with full test coverage
- **Cross-platform consistency**: Identical API and results in Python and R

### Performance

Benchmarked on 12.7M observations with 4 high-dimensional fixed effects:

| Implementation | Time (IID) | Memory |
|----------------|------------|--------|
| Python Polars | 5.8s | 291 MB |
| Python DuckDB | 18.4s | 27 MB |
| PyFixest | 91.2s | 6,691 MB |
| R Polars | 15.1s | 986 MB |
| R DuckDB | 18.3s | 714 MB |
| fixest | 11.0s | 2,944 MB |
