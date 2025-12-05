"""leanfe: Lean, fast fixed effects regression using Polars and DuckDB."""

from .fast_feols import fast_feols, leanfe
from .polars_impl import fast_feols_polars
from .duckdb_impl import fast_feols_duckdb
from .common import parse_formula

__version__ = "0.1.0"
__all__ = ["leanfe", "fast_feols", "fast_feols_polars", "fast_feols_duckdb", "parse_formula"]
