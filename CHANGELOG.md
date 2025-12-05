# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
- **Python package**: `leanfe` with full test coverage (39 tests)
- **R package**: `fasthdferg` with full test coverage (39 tests)
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
