"""
Fast high-dimensional fixed effects regression.

Main entry point for the package with support for Polars and DuckDB backends.
"""

from typing import List, Optional, Union, Literal
import polars as pl

from .polars_impl import leanfe_polars
from .duckdb_impl import leanfe_duckdb


def leanfe(
    data: Union[str, pl.DataFrame],
    y_col: Optional[str] = None,
    x_cols: Optional[List[str]] = None,
    fe_cols: Optional[List[str]] = None,
    formula: Optional[str] = None,
    weights: Optional[str] = None,
    demean_tol: float = 1e-3,
    n_iter: int = 100,
    vcov: str = "iid",
    cluster_cols: Optional[List[str]] = None,
    ssc: bool = False,
    sample_frac: Optional[float] = None,
    backend: Literal["polars", "duckdb"] = "polars"
) -> dict:
    """
    Fast fixed effects regression using Polars or DuckDB backend.
    
    This is the main entry point for the package. By default uses the Polars
    backend for speed; use DuckDB for very large datasets that don't fit in memory.
    
    Parameters
    ----------
    data : str or polars.DataFrame
        Input data: either a Polars DataFrame or path to a Parquet file.
    y_col : str, optional
        Dependent variable column name (optional if formula provided).
    x_cols : list of str, optional
        Independent variable column names (optional if formula provided).
    fe_cols : list of str, optional
        Fixed effect column names (optional if formula provided).
    formula : str, optional
        R-style formula: "y ~ x1 + x2 + x:i(factor) | fe1 + fe2 | z1 + z2" (IV).
        Supports:
        - Regular variables: x1, x2
        - Factor variables: i(region) → automatic dummy expansion
        - Interactions: treatment:i(region) → heterogeneous effects
        - Instruments (IV): third part after second |
    weights : str, optional
        Column name for regression weights (WLS).
    demean_tol : float, default 1e-3
        Convergence tolerance for iterative demeaning.
    n_iter : int, default 100
        Maximum iterations for demeaning.
    vcov : str, default "iid"
        Variance-covariance estimator: "iid", "HC1", or "cluster".
    cluster_cols : list of str, optional
        Clustering variables (required if vcov="cluster").
    ssc : bool, default False
        Small sample correction for clustered standard errors.
    sample_frac : float, optional
        Fraction of data to sample (e.g., 0.1 for 10%).
    backend : {"polars", "duckdb"}, default "polars"
        Computation backend to use:
        - "polars": Faster, better for data that fits in memory (default)
        - "duckdb": Lower memory usage, better for very large datasets
    
    Returns
    -------
    dict
        Dictionary containing:
        - coefficients: dict mapping variable names to coefficient estimates
        - std_errors: dict mapping variable names to standard errors
        - n_obs: number of observations used
        - iterations: number of demeaning iterations
        - vcov_type: type of variance estimator used
        - is_iv: whether IV/2SLS was used
        - n_instruments: number of instruments (if IV)
        - n_clusters: number of clusters (if clustered SEs)
    
    Notes
    -----
    **Choosing a Backend:**
    
    Use Polars (default) when:
    - Speed is the priority
    - Data fits comfortably in memory
    - You need the fastest possible execution
    
    Use DuckDB when:
    - Dataset is larger than available RAM
    - Memory is constrained
    - Reading directly from parquet files without loading into memory
    
    **Performance Comparison (12.7M obs, 4 FEs):**
    - Polars: ~6s, ~300 MB memory
    - DuckDB: ~18s, ~30 MB memory
    
    Examples
    --------
    Basic usage with formula:
    
    >>> from leanfe import leanfe
    >>> result = leanfe(df, formula="y ~ treatment | customer + product")
    >>> print(result['coefficients']['treatment'])
    
    With clustered standard errors:
    
    >>> result = leanfe(
    ...     df,
    ...     formula="y ~ treatment | customer + product",
    ...     vcov="cluster",
    ...     cluster_cols=["customer"]
    ... )
    
    Using DuckDB for large datasets:
    
    >>> result = leanfe(
    ...     "large_data.parquet",
    ...     formula="y ~ treatment | fe1 + fe2",
    ...     backend="duckdb"
    ... )
    
    Difference-in-Differences:
    
    >>> result = leanfe(
    ...     df,
    ...     formula="y ~ treated_post | state + year",
    ...     vcov="cluster",
    ...     cluster_cols=["state"]
    ... )
    """
    if backend == "polars":
        return leanfe_polars(
            data=data,
            y_col=y_col,
            x_cols=x_cols,
            fe_cols=fe_cols,
            formula=formula,
            weights=weights,
            demean_tol=demean_tol,
            n_iter=n_iter,
            vcov=vcov,
            cluster_cols=cluster_cols,
            ssc=ssc,
            sample_frac=sample_frac
        )
    elif backend == "duckdb":
        return leanfe_duckdb(
            data=data,
            y_col=y_col,
            x_cols=x_cols,
            fe_cols=fe_cols,
            formula=formula,
            weights=weights,
            demean_tol=demean_tol,
            n_iter=n_iter,
            vcov=vcov,
            cluster_cols=cluster_cols,
            ssc=ssc,
            sample_frac=sample_frac
        )
    else:
        raise ValueError(f"backend must be 'polars' or 'duckdb', got '{backend}'")


