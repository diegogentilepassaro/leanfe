"""
DuckDB-based fixed effects regression implementation.

Optimized for memory efficiency using in-database operations.
"""

import duckdb
import numpy as np
import polars as pl
from typing import List, Optional, Union

from .common import (
    parse_formula,
    iv_2sls,
    compute_standard_errors,
    build_result
)


def leanfe_duckdb(
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
    Fixed effects regression using DuckDB with optimized memory usage.
    
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
        coefficients, std_errors, n_obs, iterations, vcov_type, 
        is_iv, n_instruments, n_clusters
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
    
    # Make lists mutable
    x_cols = list(x_cols)
    fe_cols = list(fe_cols)
    
    # Build needed columns
    needed_cols = [y_col] + x_cols + fe_cols + instruments
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
    
    # Setup DuckDB connection
    con = duckdb.connect()
    con.execute("SET memory_limit='4GB'")
    con.execute("SET threads=4")
    
    try:
        # Register data source
        if isinstance(data, str):
            col_list = ', '.join(needed_cols)
            con.execute(f"CREATE VIEW raw_data AS SELECT {col_list} FROM read_parquet('{data}')")
        else:
            df = data.select(needed_cols)
            con.register("raw_data", df.to_arrow())
        
        # Sample if requested
        if sample_frac is not None:
            con.execute(f"CREATE TABLE data AS SELECT * FROM raw_data USING SAMPLE {sample_frac * 100}%")
        else:
            con.execute("CREATE TABLE data AS SELECT * FROM raw_data")
        
        # Handle interactions
        if interactions:
            for var, factor, ref in interactions:
                cats = [r[0] for r in con.execute(f"SELECT DISTINCT {factor} FROM data ORDER BY {factor}").fetchall()]
                # Determine reference category
                if ref is None:
                    ref_cat = cats[0]  # Default: first category
                else:
                    # Try to match type
                    ref_cat = ref
                    if cats and not isinstance(cats[0], str):
                        try:
                            ref_cat = type(cats[0])(ref)
                        except (ValueError, TypeError):
                            pass
                    if ref_cat not in cats:
                        raise ValueError(f"Reference category '{ref}' not found in {factor}. Available: {cats}")
                
                for cat in cats:
                    if cat == ref_cat:
                        continue  # Skip reference category
                    col_name = f"{var}_{cat}"
                    con.execute(f"ALTER TABLE data ADD COLUMN {col_name} DOUBLE")
                    con.execute(f"UPDATE data SET {col_name} = CASE WHEN {factor} = '{cat}' THEN {var} ELSE 0 END")
                    x_cols.append(col_name)
        
        # Handle factor variables
        if factor_vars:
            for var, ref in factor_vars:
                cats = [r[0] for r in con.execute(f"SELECT DISTINCT {var} FROM data ORDER BY {var}").fetchall()]
                # Determine reference category
                if ref is None:
                    ref_cat = cats[0]  # Default: first category
                else:
                    # Try to match type
                    ref_cat = ref
                    if cats and not isinstance(cats[0], str):
                        try:
                            ref_cat = type(cats[0])(ref)
                        except (ValueError, TypeError):
                            pass
                    if ref_cat not in cats:
                        raise ValueError(f"Reference category '{ref}' not found in {var}. Available: {cats}")
                
                for cat in cats:
                    if cat == ref_cat:
                        continue  # Skip reference category
                    col_name = f"{var}_{cat}"
                    con.execute(f"ALTER TABLE data ADD COLUMN {col_name} DOUBLE")
                    con.execute(f"UPDATE data SET {col_name} = CASE WHEN {var} = '{cat}' THEN 1 ELSE 0 END")
                    x_cols.append(col_name)
        
        # Drop singletons
        for fe in fe_cols:
            con.execute(f"DELETE FROM data WHERE {fe} IN (SELECT {fe} FROM data GROUP BY {fe} HAVING COUNT(*) = 1)")
        
        n_obs = con.execute("SELECT COUNT(*) FROM data").fetchone()[0]
        cols_to_demean = [y_col] + x_cols + instruments
        
        # Add demeaned columns
        for col in cols_to_demean:
            con.execute(f"ALTER TABLE data ADD COLUMN {col}_dm DOUBLE")
            con.execute(f"UPDATE data SET {col}_dm = {col}")
        
        dm_cols = [f"{col}_dm" for col in cols_to_demean]
        
        # Iterative demeaning
        for it in range(1, n_iter + 1):
            for fe in fe_cols:
                if weights is not None:
                    for col in dm_cols:
                        con.execute(f"""
                            UPDATE data SET {col} = {col} - (
                                SELECT SUM(d2.{col} * d2.{weights}) / SUM(d2.{weights})
                                FROM data d2 WHERE d2.{fe} = data.{fe}
                            )
                        """)
                else:
                    for col in dm_cols:
                        con.execute(f"""
                            WITH fe_means AS (SELECT {fe}, AVG({col}) as mean_val FROM data GROUP BY {fe})
                            UPDATE data SET {col} = data.{col} - fe_means.mean_val
                            FROM fe_means WHERE data.{fe} = fe_means.{fe}
                        """)
            
            if it >= 3:
                max_mean = 0
                for fe in fe_cols:
                    for col in dm_cols:
                        result = con.execute(f"SELECT MAX(ABS(avg_val)) FROM (SELECT AVG({col}) as avg_val FROM data GROUP BY {fe})").fetchone()[0]
                        max_mean = max(max_mean, abs(result or 0))
                if max_mean < demean_tol:
                    break
        
        k = len(x_cols)
        is_iv = len(instruments) > 0
        
        if is_iv:
            # Extract data for IV
            select_cols = [f"{col}_dm" for col in [y_col] + x_cols + instruments]
            if weights is not None:
                select_cols.append(weights)
            if cluster_cols is not None:
                select_cols.extend(cluster_cols)
            
            result_df = con.execute(f"SELECT {', '.join(select_cols)} FROM data").fetchdf()
            Y = result_df[f"{y_col}_dm"].values
            X = result_df[[f"{col}_dm" for col in x_cols]].values
            Z = result_df[[f"{col}_dm" for col in instruments]].values
            w = result_df[weights].values if weights is not None else None
            
            beta, X_hat = iv_2sls(Y, X, Z, w)
            resid = Y - X_hat @ beta
        else:
            # OLS via SQL aggregates
            XtX = np.zeros((k, k))
            Xty = np.zeros(k)
            
            for i, col_i in enumerate(x_cols):
                col_i_dm = f"{col_i}_dm"
                Xty[i] = con.execute(f"SELECT SUM({col_i_dm} * {y_col}_dm) FROM data").fetchone()[0]
                for j in range(i, k):
                    col_j_dm = f"{x_cols[j]}_dm"
                    val = con.execute(f"SELECT SUM({col_i_dm} * {col_j_dm}) FROM data").fetchone()[0]
                    XtX[i, j] = val
                    XtX[j, i] = val
            
            beta = np.linalg.solve(XtX, Xty)
            
            resid_expr = f"{y_col}_dm - (" + " + ".join([f"{b} * {col}_dm" for b, col in zip(beta, x_cols)]) + ")"
            con.execute("ALTER TABLE data ADD COLUMN _resid DOUBLE")
            con.execute(f"UPDATE data SET _resid = {resid_expr}")
        
        # Degrees of freedom
        n_fe_groups = sum(con.execute(f"SELECT COUNT(DISTINCT {fe}) FROM data").fetchone()[0] for fe in fe_cols)
        absorbed_df = n_fe_groups - len(fe_cols)
        df_resid = n_obs - k - absorbed_df
        
        # Compute XtX_inv
        if is_iv:
            if w is not None:
                sqrt_w = np.sqrt(w)
                X_hat_w = X_hat * sqrt_w[:, np.newaxis]
                XtX_inv = np.linalg.inv(X_hat_w.T @ X_hat_w)
            else:
                XtX_inv = np.linalg.inv(X_hat.T @ X_hat)
        else:
            XtX_inv = np.linalg.inv(XtX)
        
        # Standard errors
        if is_iv:
            cluster_ids = None
            if vcov == "cluster":
                if cluster_cols is None:
                    raise ValueError("cluster_cols required for vcov='cluster'")
                if len(cluster_cols) == 1:
                    cluster_ids = result_df[cluster_cols[0]].values
                else:
                    cluster_ids = result_df[cluster_cols].apply(lambda x: '_'.join(x.astype(str)), axis=1).values
            
            se, n_clusters = compute_standard_errors(
                XtX_inv=XtX_inv, resid=resid, n_obs=n_obs, df_resid=df_resid,
                vcov=vcov, X=X_hat, weights=w, cluster_ids=cluster_ids, ssc=ssc
            )
        else:
            n_clusters = None
            if vcov == "iid":
                if weights is not None:
                    sigma2 = con.execute(f"SELECT SUM({weights} * _resid * _resid) / {df_resid} FROM data").fetchone()[0]
                else:
                    sigma2 = con.execute(f"SELECT SUM(_resid * _resid) / {df_resid} FROM data").fetchone()[0]
                se = np.sqrt(sigma2 * np.diag(XtX_inv))
            elif vcov == "HC1":
                meat = np.zeros((k, k))
                for i, col_i in enumerate(x_cols):
                    for j in range(i, k):
                        col_j = x_cols[j]
                        if weights is not None:
                            val = con.execute(f"SELECT SUM({weights} * {col_i}_dm * {col_j}_dm * _resid * _resid) FROM data").fetchone()[0]
                        else:
                            val = con.execute(f"SELECT SUM({col_i}_dm * {col_j}_dm * _resid * _resid) FROM data").fetchone()[0]
                        meat[i, j] = val
                        meat[j, i] = val
                vcov_matrix = XtX_inv @ meat @ XtX_inv
                se = np.sqrt((n_obs / df_resid) * np.diag(vcov_matrix))
            elif vcov == "cluster":
                if cluster_cols is None:
                    raise ValueError("cluster_cols required for vcov='cluster'")
                cluster_expr = cluster_cols[0] if len(cluster_cols) == 1 else f"CONCAT_WS('_', {', '.join(cluster_cols)})"
                if weights is not None:
                    score_exprs = [f"SUM({col}_dm * _resid * {weights}) AS score_{i}" for i, col in enumerate(x_cols)]
                else:
                    score_exprs = [f"SUM({col}_dm * _resid) AS score_{i}" for i, col in enumerate(x_cols)]
                cluster_scores = con.execute(f"SELECT {cluster_expr} AS cluster_id, {', '.join(score_exprs)} FROM data GROUP BY {cluster_expr}").fetchdf()
                n_clusters = len(cluster_scores)
                S = cluster_scores[[f"score_{i}" for i in range(k)]].values
                meat = S.T @ S
                adj = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / df_resid) if ssc else n_clusters / (n_clusters - 1)
                vcov_matrix = adj * XtX_inv @ meat @ XtX_inv
                se = np.sqrt(np.diag(vcov_matrix))
            else:
                raise ValueError(f"Unknown vcov: {vcov}")
    finally:
        con.close()
    
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
        formula=formula,
        fe_cols=fe_cols
    )
