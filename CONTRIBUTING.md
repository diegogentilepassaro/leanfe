# Contributing to leanfe

Thank you for your interest in contributing! This document provides guidelines for contributing to the project.

## Development Setup

### Python

```bash
cd package/python
pip install -e ".[dev]"
```

### R

```r
# Install dependencies
install.packages(c("polars", "duckdb", "DBI"))

# Source files for development
source("R/common.R")
source("R/polars.R")
source("R/duckdb.R")
source("R/fast_feols.R")
```

## Running Tests

### Python

```bash
cd package/python
pytest tests/ -v
```

### R

```bash
cd package/r
for f in tests/*.R; do Rscript "$f"; done
```

## Code Style

### Python

- Follow PEP 8
- Use type hints for function signatures
- Maximum line length: 100 characters
- Format with `black`: `black --line-length 100 .`

### R

- Follow tidyverse style guide where applicable
- Use roxygen2 comments for documentation
- Internal functions should be prefixed with `.` (e.g., `.parse_formula`)

## Adding Features

When adding new features:

1. **Implement in both Python and R** - The packages should maintain feature parity
2. **Add tests** - Both packages should have equivalent tests
3. **Update documentation** - Update READMEs, docstrings, and man pages
4. **Update CHANGELOG** - Document the change

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Run all tests to ensure nothing is broken
5. Commit with clear messages
6. Push to your fork
7. Open a Pull Request

## Reporting Issues

When reporting issues, please include:

- Python/R version
- Package version
- Minimal reproducible example
- Expected vs actual behavior
- Full error message/traceback

## Architecture

```
package/
├── python/
│   └── leanfe/
│       ├── fast_feols.py    # Unified API
│       ├── common.py        # Shared utilities
│       ├── polars_impl.py   # Polars backend
│       └── duckdb_impl.py   # DuckDB backend
│
└── r/
    └── R/
        ├── fast_feols.R     # Unified API
        ├── common.R         # Shared utilities
        ├── polars.R         # Polars backend
        └── duckdb.R         # DuckDB backend
```

The `common` modules contain shared logic (formula parsing, IV estimation, SE computation) to avoid duplication between backends.

## Questions?

Feel free to open an issue for questions or discussion.
