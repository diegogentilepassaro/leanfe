"""
Polars-based fixed effects regression implementation.

Optimized for speed using Polars DataFrame operations.
Uses YOCO compression automatically for IID/HC1 standard errors.
"""

import polars as pl
import numpy as np
from typing import List, Optional, Union

from .common import (
    parse_formula,
    iv_2sls,
    compute_standard_errors,
    build_result
)
from .compress import should_use_compress, leanfe_compress_polars


def _expand_factors(df: pl.DataFrame, factor_vars: List[tuple]) -> tuple:
    """
    Expand factor variables into dummy variables, dropping reference category.
    
    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame
    factor_vars : list of tuples
        List of (var_name, ref_category) tuples. ref_category is None for first category,
        or a specific value to use as reference.
    
    Returns
    -------
    tuple
        (df, dummy_cols) - DataFrame with dummies added and list of dummy column names
    """
    dummy_cols = []
    for var, ref in factor_vars:
        categories = df.select(pl.col(var).unique().sort()).to_series().to_list()
        if ref is None:
            # Default: drop first category
            ref_cat = categories[0]
        else:
            # User specified reference - try to match type
            ref_cat = ref
            # Convert ref to match category type if needed
            if categories and not isinstance(categories[0], str):
                try:
                    ref_cat = type(categories[0])(ref)
                except (ValueError, TypeError):
                    pass
            if ref_cat not in categories:
                raise ValueError(f"Reference category '{ref}' not found in {var}. Available: {categories}")
        
        for cat in categories:
            if cat == ref_cat:
                continue  # Skip reference category
            dummy_name = f"{var}_{cat}"
            df = df.with_columns(
                (pl.col(var) == cat).cast(pl.Int8).alias(dummy_name)
            )
            dummy_cols.append(dummy_name)
    return df, dummy_cols


def _expand_interactions(df: pl.DataFrame, interactions: List[tuple]) -> tuple:
    """
    Expand interaction terms: var:i(factor) -> var_cat1, var_cat2, ...
    
    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame
    interactions : list of tuples
        List of (var, factor, ref) tuples. ref is None for first category,
        or a specific value to use as reference.
    
    Returns
    -------
    tuple
        (df, interaction_cols) - DataFrame with interactions added and list of column names
    """
    interaction_cols = []
    for var, factor, ref in interactions:
        categories = df.select(pl.col(factor).unique().sort()).to_series().to_list()
        if ref is None:
            # Default: drop first category
            ref_cat = categories[0]
        else:
            # User specified reference - try to match type
            ref_cat = ref
            if categories and not isinstance(categories[0], str):
                try:
                    ref_cat = type(categories[0])(ref)
                except (ValueError, TypeError):
                    pass
            if ref_cat not in categories:
                raise ValueError(f"Reference category '{ref}' not found in {factor}. Available: {categories}")
        
        for cat in categories:
            if cat == ref_cat:
                continue  # Skip reference category
            col_name = f"{var}_{cat}"
            df = df.with_columns(
                (pl.col(var) * (pl.col(factor) == cat)).cast(pl.Float64).alias(col_name)
            )
            interaction_cols.append(col_name)
    return df, interaction_cols


def _optimize_dtypes(df: pl.DataFrame, fe_cols: List[str]) -> pl.DataFrame:
    """Optimize column types: downcast integers, categorize fixed effects."""
    for col in df.columns:
        dtype = df[col].dtype
        if dtype in (pl.Int64, pl.Int32, pl.Int16, pl.Int8):
            col_min = df[col].min()
            col_max = df[col].max()
            if col_min >= -128 and col_max <= 127:
                df = df.with_columns(pl.col(col).cast(pl.Int8))
            elif col_min >= -32768 and col_max <= 32767:
                df = df.with_columns(pl.col(col).cast(pl.Int16))
            elif col_min >= -2147483648 and col_max <= 2147483647:
                df = df.with_columns(pl.col(col).cast(pl.Int32))
        elif col in fe_cols and dtype == pl.Utf8:
            df = df.with_columns(pl.col(col).cast(pl.Categorical))
    return df


def leanfe_polars(
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
    sample_frac: Optional[float] = None
) -> dict:
    """
    Fast fixed effects regression using Polars and FWL theorem.
    
    Parameters
    ----------
    data : str or polars.DataFrame
        Input data: either DataFrame or path to Parquet file
    y_col : str, optional
        Dependent variable column name (optional if formula provided)
    x_cols : list of str, optional
        Independent variable column names (optional if formula provided)
    fe_cols : list of str, optional
        Fixed effect column names (optional if formula provided)
    formula : str, optional
        R-style formula: "y ~ x1 + x2 + x:i(factor) | fe1 + fe2 | z1 + z2" (IV)
    weights : str, optional
        Column name for regression weights
    demean_tol : float, default 1e-3
        Convergence tolerance for demeaning
    n_iter : int, default 100
        Maximum iterations for demeaning
    vcov : str, default "iid"
        Variance-covariance estimator: "iid", "HC1", or "cluster"
    cluster_cols : list of str, optional
        Clustering variables (required if vcov="cluster")
    ssc : bool, default False
        Small sample correction for clustered SEs
    sample_frac : float, optional
        Fraction of data to sample (e.g., 0.1 for 10%)
    
    Returns
    -------
    dict
        coefficients, std_errors, n_obs, iterations, vcov_type, is_iv, n_instruments, n_clusters
    """
    # Parse formula if provided
    if formula is not None:
        y_col, x_cols, fe_cols, factor_vars, interactions, instruments = parse_formula(formula)
    elif y_col is None or x_cols is None or fe_cols is None:
        raise ValueError("Must provide either 'formula' or (y_col, x_cols, fe_cols)")
    else:
        factor_vars = []
        interactions = []
        instruments = []
    
    # Build needed columns
    needed_cols = [y_col] + list(x_cols) + list(fe_cols) + instruments
    # Add factor variable columns
    for var, ref in factor_vars:
        if var not in needed_cols:
            needed_cols.append(var)
    # Add interaction variable columns
    for var, factor, ref in interactions:
        if var not in needed_cols:
            needed_cols.append(var)
        if factor not in needed_cols:
            needed_cols.append(factor)
    if cluster_cols is not None:
        needed_cols += [c for c in cluster_cols if c not in needed_cols]
    if weights is not None and weights not in needed_cols:
        needed_cols.append(weights)
    
    # Load data
    if isinstance(data, str):
        df = pl.scan_parquet(data).select(needed_cols).collect()
    else:
        df = data.select(needed_cols)
    
    # Make x_cols mutable
    x_cols = list(x_cols)
    
    # Expand interactions
    if interactions:
        df, interaction_cols = _expand_interactions(df, interactions)
        x_cols = x_cols + interaction_cols
    
    # Optimize types (always done for memory efficiency)
    factor_var_names = [var for var, ref in factor_vars]
    df = _optimize_dtypes(df, fe_cols + factor_var_names)
    
    # Sample data if requested
    if sample_frac is not None:
        df = df.sample(fraction=sample_frac, seed=42)
    
    # Expand factor variables into dummies
    if factor_vars:
        df, dummy_cols = _expand_factors(df, factor_vars)
        x_cols = x_cols + dummy_cols
    
    # Check if we should use compression strategy (faster for IID/HC1 without IV)
    is_iv = len(instruments) > 0
    use_compress = should_use_compress(vcov, is_iv)
    
    if use_compress:
        # Use YOCO compression strategy - much faster for discrete regressors
        result = leanfe_compress_polars(
            df=df,
            y_col=y_col,
            x_cols=x_cols,
            fe_cols=fe_cols,
            weights=weights,
            vcov=vcov
        )
        # Add missing fields for compatibility
        result["iterations"] = 0
        result["is_iv"] = False
        result["n_instruments"] = None
        result["n_clusters"] = None
        result["formula"] = formula
        result["fe_cols"] = fe_cols
        return result
    
    # Fall back to FWL demeaning for cluster SEs or IV
    # Extract weights if provided
    if weights is not None:
        w = df.select(weights).to_numpy().flatten()
    else:
        w = None
    
    # Drop singletons
    prev_len = len(df) + 1
    while len(df) < prev_len:
        prev_len = len(df)
        for fe in fe_cols:
            counts = df.group_by(fe).agg(pl.len().alias("cnt"))
            df = df.join(counts, on=fe, how="left").filter(pl.col("cnt") > 1).drop("cnt")
    
    n_obs = len(df)
    cols_to_demean = [y_col] + x_cols + instruments
    
    # FWL demeaning
    for it in range(1, n_iter + 1):
        for fe in fe_cols:
            if weights is not None:
                means = df.group_by(fe).agg([
                    (pl.col(c) * pl.col(weights)).sum().truediv(pl.col(weights).sum()).alias(f"{c}_mean") 
                    for c in cols_to_demean
                ])
            else:
                means = df.group_by(fe).agg([pl.col(c).mean().alias(f"{c}_mean") for c in cols_to_demean])
            
            df = df.join(means, on=fe, how="left")
            for c in cols_to_demean:
                df = df.with_columns((pl.col(c) - pl.col(f"{c}_mean")).alias(c)).drop(f"{c}_mean")
        
        if it >= 3:
            max_mean = max(
                df.group_by(fe).agg(pl.col(y_col).mean().alias("m"))
                .select(pl.col("m").abs().max()).item()
                for fe in fe_cols
            )
            if max_mean < demean_tol:
                break
    
    # Extract X and Y for OLS/IV solve
    X = df.select(x_cols).to_numpy()
    Y = df.select(y_col).to_numpy().flatten()
    
    # IV/2SLS or OLS
    is_iv = len(instruments) > 0
    if is_iv:
        Z = df.select(instruments).to_numpy()
        beta, X_hat = iv_2sls(Y, X, Z, w)
    else:
        if w is not None:
            sqrt_w = np.sqrt(w)
            X_w = X * sqrt_w[:, np.newaxis]
            Y_w = Y * sqrt_w
            XtX = X_w.T @ X_w
            Xty = X_w.T @ Y_w
        else:
            XtX = X.T @ X
            Xty = X.T @ Y
        beta = np.linalg.solve(XtX, Xty)
        X_hat = X
    
    # Compute (X'X)^-1 for standard errors
    if w is not None:
        sqrt_w = np.sqrt(w)
        X_hat_w = X_hat * sqrt_w[:, np.newaxis]
        XtX_inv = np.linalg.inv(X_hat_w.T @ X_hat_w)
    else:
        XtX_inv = np.linalg.inv(X_hat.T @ X_hat)
    
    # Residuals
    resid = Y - X_hat @ beta
    
    # Calculate absorbed degrees of freedom
    fe_counts = {fe: df.group_by(fe).agg(pl.len()).height for fe in fe_cols}
    absorbed_df = sum(fe_counts.values()) - len(fe_cols)
    df_resid = n_obs - len(x_cols) - absorbed_df
    
    # Build cluster IDs if needed
    cluster_ids = None
    if vcov == "cluster":
        if cluster_cols is None:
            raise ValueError("cluster_cols must be provided when vcov='cluster'")
        if len(cluster_cols) == 1:
            cluster_ids = df.select(cluster_cols[0]).to_numpy().flatten()
        else:
            cluster_ids = df.select(
                pl.concat_str([pl.col(c).cast(pl.Utf8) for c in cluster_cols], separator="_")
            ).to_numpy().flatten()
    
    # Compute standard errors
    se, n_clusters = compute_standard_errors(
        XtX_inv=XtX_inv,
        resid=resid,
        n_obs=n_obs,
        df_resid=df_resid,
        vcov=vcov,
        X=X_hat,
        weights=w,
        cluster_ids=cluster_ids,
        ssc=ssc
    )
    
    # Compute R-squared (within)
    rss = np.sum(resid**2)
    tss = np.sum((Y - np.mean(Y))**2)
    r_squared_within = 1 - rss / tss if tss > 0 else None
    
    return build_result(
        x_cols=x_cols,
        beta=beta,
        se=se,
        n_obs=n_obs,
        iterations=it,
        vcov=vcov,
        is_iv=is_iv,
        n_instruments=len(instruments) if is_iv else None,
        n_clusters=n_clusters,
        df_resid=df_resid,
        r_squared_within=r_squared_within,
        formula=formula,
        fe_cols=fe_cols
    )
