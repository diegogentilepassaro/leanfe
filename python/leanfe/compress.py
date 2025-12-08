"""
YOCO (You Only Compress Once) compression strategy for leanfe.

Implements the optimal data compression strategy from Wong et al. (2021):
"You Only Compress Once: Optimal Data Compression for Estimating Linear Models"

This strategy compresses data by grouping on (regressors + fixed effects),
computing sufficient statistics (n, sum_y, sum_y_sq), and running weighted
least squares on the compressed data.

Used automatically when vcov is "iid" or "HC1" (not cluster) and no IV.
"""

import numpy as np
import polars as pl
from scipy import sparse
from scipy.sparse.linalg import spsolve
from typing import List, Optional, Dict, Tuple


def should_use_compress(vcov: str, has_instruments: bool) -> bool:
    """
    Determine if compression strategy should be used.
    
    Compression is used when:
    - vcov is "iid" or "HC1" (cluster requires different handling)
    - No instrumental variables
    
    Parameters
    ----------
    vcov : str
        Variance-covariance type
    has_instruments : bool
        Whether IV/2SLS is being used
        
    Returns
    -------
    bool
        True if compression should be used
    """
    vcov_ok = vcov.lower() in ("iid", "hc1")
    return vcov_ok and not has_instruments


def compress_polars(
    df: pl.DataFrame,
    y_col: str,
    x_cols: List[str],
    fe_cols: List[str],
    weights: Optional[str] = None
) -> Tuple[pl.DataFrame, int]:
    """
    Compress data using GROUP BY on regressors + fixed effects.
    
    Parameters
    ----------
    df : pl.DataFrame
        Input data
    y_col : str
        Dependent variable column
    x_cols : list of str
        Regressor columns
    fe_cols : list of str
        Fixed effect columns
    weights : str, optional
        Weight column name
        
    Returns
    -------
    tuple
        (compressed_df, n_obs_original)
    """
    group_cols = x_cols + fe_cols
    n_obs_original = len(df)
    
    if weights is not None:
        # Weighted compression
        agg_exprs = [
            pl.col(weights).sum().alias("_n"),
            (pl.col(y_col) * pl.col(weights)).sum().alias("_sum_y"),
            (pl.col(y_col).pow(2) * pl.col(weights)).sum().alias("_sum_y_sq"),
        ]
    else:
        agg_exprs = [
            pl.len().alias("_n"),
            pl.col(y_col).sum().alias("_sum_y"),
            pl.col(y_col).pow(2).sum().alias("_sum_y_sq"),
        ]
    
    compressed = df.group_by(group_cols).agg(agg_exprs)
    
    # Add mean_y and sqrt weights for WLS
    compressed = compressed.with_columns([
        (pl.col("_sum_y") / pl.col("_n")).alias("_mean_y"),
        pl.col("_n").sqrt().alias("_wts"),
    ])
    
    return compressed, n_obs_original


class DuckDBResult:
    """Wrapper for DuckDB numpy result to provide dict-like access."""
    def __init__(self, data: Dict[str, np.ndarray]):
        self._data = data
    
    def __getitem__(self, key: str) -> np.ndarray:
        return self._data[key]
    
    def __len__(self) -> int:
        # Return length of first array
        return len(next(iter(self._data.values())))


def compress_duckdb(
    con,
    y_col: str,
    x_cols: List[str],
    fe_cols: List[str],
    weights: Optional[str] = None
) -> Tuple[DuckDBResult, int]:
    """
    Compress data using SQL GROUP BY.
    
    Parameters
    ----------
    con : duckdb.Connection
        DuckDB connection with 'data' table
    y_col : str
        Dependent variable column
    x_cols : list of str
        Regressor columns
    fe_cols : list of str
        Fixed effect columns
    weights : str, optional
        Weight column name
        
    Returns
    -------
    tuple
        (compressed_data as DuckDBResult, n_obs_original)
    """
    group_cols = x_cols + fe_cols
    group_cols_sql = ", ".join(group_cols)
    
    n_obs_original = con.execute("SELECT COUNT(*) FROM data").fetchone()[0]
    
    if weights is not None:
        query = f"""
        SELECT
            {group_cols_sql},
            SUM({weights}) AS _n,
            SUM({y_col} * {weights}) AS _sum_y,
            SUM(POWER({y_col}, 2) * {weights}) AS _sum_y_sq,
            SUM({y_col} * {weights}) / SUM({weights}) AS _mean_y,
            SQRT(SUM({weights})) AS _wts
        FROM data
        GROUP BY {group_cols_sql}
        """
    else:
        query = f"""
        SELECT
            {group_cols_sql},
            COUNT(*) AS _n,
            SUM({y_col}) AS _sum_y,
            SUM(POWER({y_col}, 2)) AS _sum_y_sq,
            SUM({y_col}) / COUNT(*) AS _mean_y,
            SQRT(COUNT(*)) AS _wts
        FROM data
        GROUP BY {group_cols_sql}
        """
    
    # Use fetchnumpy() to avoid pandas dependency
    result = con.execute(query).fetchnumpy()
    compressed = DuckDBResult(result)
    return compressed, n_obs_original


def _extract_numpy_arrays(compressed_df, x_cols: List[str], backend: str):
    """Extract numpy arrays from compressed dataframe without pandas dependency."""
    if backend == "polars":
        # Use Polars native to_numpy() - no pandas needed
        X_reg = np.column_stack([compressed_df[col].to_numpy() for col in x_cols])
        Y = compressed_df["_mean_y"].to_numpy()
        wts = compressed_df["_wts"].to_numpy()
        
        def get_fe_values(fe):
            return compressed_df[fe].to_numpy()
    else:
        # DuckDB returns DuckDBResult (dict-like with numpy arrays)
        X_reg = np.column_stack([compressed_df[col] for col in x_cols])
        Y = compressed_df["_mean_y"]
        wts = compressed_df["_wts"]
        
        def get_fe_values(fe):
            return compressed_df[fe]
    
    return X_reg, Y, wts, get_fe_values


def build_design_matrix(
    compressed_df,
    x_cols: List[str],
    fe_cols: List[str],
    backend: str = "polars",
    use_sparse: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], int]:
    """
    Build design matrix from compressed data with FE dummies.
    
    Parameters
    ----------
    compressed_df : DataFrame
        Compressed data (polars or pandas)
    x_cols : list of str
        Regressor columns
    fe_cols : list of str
        Fixed effect columns
    backend : str
        "polars" or "duckdb"
    use_sparse : bool
        If True, use sparse matrices for FE dummies (much faster)
        
    Returns
    -------
    tuple
        (X, Y, wts, all_col_names, n_fe_levels)
        X is sparse.csr_matrix if use_sparse=True and fe_cols exist
    """
    # Extract numpy arrays without pandas dependency
    X_reg, Y, wts, get_fe_values = _extract_numpy_arrays(compressed_df, x_cols, backend)
    n_rows = len(Y)
    
    if not fe_cols:
        return X_reg, Y, wts, list(x_cols), 0
    
    if use_sparse:
        # Build sparse FE dummies efficiently using COO format
        fe_col_names = []
        n_fe_levels = 0
        col_offset = 0
        
        # Collect sparse matrix components
        rows_list = []
        cols_list = []
        data_list = []
        
        for fe in fe_cols:
            fe_values = get_fe_values(fe)
            categories = np.unique(fe_values)
            n_cats = len(categories)
            n_fe_levels += n_cats
            
            # Create mapping from category to column index (drop first for identification)
            cat_to_col = {cat: i + col_offset for i, cat in enumerate(categories[1:])}
            
            # Add column names
            for cat in categories[1:]:
                fe_col_names.append(f"{fe}_{cat}")
            
            # Build sparse entries for this FE
            for row_idx, val in enumerate(fe_values):
                if val in cat_to_col:
                    rows_list.append(row_idx)
                    cols_list.append(cat_to_col[val])
                    data_list.append(1.0)
            
            col_offset += n_cats - 1  # -1 because we drop first category
        
        # Create sparse FE matrix
        n_fe_cols = col_offset
        if n_fe_cols > 0:
            X_fe_sparse = sparse.coo_matrix(
                (data_list, (rows_list, cols_list)),
                shape=(n_rows, n_fe_cols)
            ).tocsr()
            
            # Combine: [X_reg | X_fe] as sparse
            X_reg_sparse = sparse.csr_matrix(X_reg)
            X = sparse.hstack([X_reg_sparse, X_fe_sparse], format='csr')
        else:
            X = sparse.csr_matrix(X_reg)
        
        all_col_names = list(x_cols) + fe_col_names
        return X, Y, wts, all_col_names, n_fe_levels
    
    else:
        # Original dense implementation
        fe_dummies = []
        fe_col_names = []
        n_fe_levels = 0
        
        for fe in fe_cols:
            fe_values = get_fe_values(fe)
            categories = np.unique(fe_values)
            n_cats = len(categories)
            n_fe_levels += n_cats
            
            # Create dummy matrix (drop first category for identification)
            for cat in categories[1:]:
                col_name = f"{fe}_{cat}"
                dummy = (fe_values == cat).astype(float)
                fe_dummies.append(dummy)
                fe_col_names.append(col_name)
        
        # Combine regressors and FE dummies
        if fe_dummies:
            X_fe = np.column_stack(fe_dummies)
            X = np.hstack([X_reg, X_fe])
            all_col_names = list(x_cols) + fe_col_names
        else:
            X = X_reg
            all_col_names = list(x_cols)
        
        return X, Y, wts, all_col_names, n_fe_levels


def solve_wls(
    X,  # np.ndarray or sparse matrix
    Y: np.ndarray,
    wts: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve weighted least squares.
    
    Parameters
    ----------
    X : np.ndarray or scipy.sparse matrix
        Design matrix (G x p)
    Y : np.ndarray
        Response vector (G,) - group means
    wts : np.ndarray
        Weights (G,) - sqrt(n_g)
        
    Returns
    -------
    tuple
        (beta, XtX_inv)
    """
    is_sparse = sparse.issparse(X)
    
    if is_sparse:
        # Sparse weighted least squares
        # Weight the design matrix: diag(wts) @ X
        Xw = X.multiply(wts[:, np.newaxis])
        Yw = Y * wts
        
        # X'X is dense (p x p is typically small)
        XtX = (Xw.T @ Xw).toarray()
        Xty = Xw.T @ Yw
        
        # Solve using dense methods (XtX is small)
        try:
            L = np.linalg.cholesky(XtX)
            beta = np.linalg.solve(L.T, np.linalg.solve(L, Xty))
            XtX_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(XtX.shape[0])))
        except np.linalg.LinAlgError:
            beta, _, _, _ = np.linalg.lstsq(XtX, Xty, rcond=None)
            XtX_inv = np.linalg.pinv(XtX)
    else:
        # Dense weighted least squares
        Xw = X * wts[:, np.newaxis]
        Yw = Y * wts
        
        XtX = Xw.T @ Xw
        Xty = Xw.T @ Yw
        
        try:
            L = np.linalg.cholesky(XtX)
            beta = np.linalg.solve(L.T, np.linalg.solve(L, Xty))
            XtX_inv = np.linalg.solve(L.T, np.linalg.solve(L, np.eye(XtX.shape[0])))
        except np.linalg.LinAlgError:
            beta, _, _, _ = np.linalg.lstsq(XtX, Xty, rcond=None)
            XtX_inv = np.linalg.pinv(XtX)
    
    return beta, XtX_inv


def compute_rss_grouped(
    compressed_df,
    X,  # np.ndarray or sparse matrix
    beta: np.ndarray,
    backend: str = "polars"
) -> Tuple[float, np.ndarray]:
    """
    Compute RSS from sufficient statistics.
    
    RSS = sum_g (sum_y_sq_g - 2 * yhat_g * sum_y_g + n_g * yhat_g^2)
    
    Parameters
    ----------
    compressed_df : DataFrame
        Compressed data with _n, _sum_y, _sum_y_sq
    X : np.ndarray or scipy.sparse matrix
        Design matrix
    beta : np.ndarray
        Coefficient vector
    backend : str
        "polars" or "duckdb"
        
    Returns
    -------
    tuple
        (rss_total, rss_g) - total RSS and per-group RSS
    """
    if backend == "polars":
        n_g = compressed_df["_n"].to_numpy()
        sum_y_g = compressed_df["_sum_y"].to_numpy()
        sum_y_sq_g = compressed_df["_sum_y_sq"].to_numpy()
    else:
        # DuckDB returns DuckDBResult (dict-like with numpy arrays)
        n_g = compressed_df["_n"]
        sum_y_g = compressed_df["_sum_y"]
        sum_y_sq_g = compressed_df["_sum_y_sq"]
    
    # Fitted values for each group (handle sparse)
    if sparse.issparse(X):
        yhat_g = np.asarray(X @ beta).flatten()
    else:
        yhat_g = X @ beta
    
    # Per-group RSS
    rss_g = sum_y_sq_g - 2 * yhat_g * sum_y_g + n_g * (yhat_g ** 2)
    rss_total = np.sum(rss_g)
    
    return rss_total, rss_g


def compute_se_compress(
    XtX_inv: np.ndarray,
    rss_total: float,
    rss_g: np.ndarray,
    n_obs: int,
    df_resid: int,
    vcov: str,
    X,  # np.ndarray or sparse matrix
    x_cols: List[str]
) -> np.ndarray:
    """
    Compute standard errors from compressed data.
    
    Parameters
    ----------
    XtX_inv : np.ndarray
        Inverse of X'X
    rss_total : float
        Total residual sum of squares
    rss_g : np.ndarray
        Per-group RSS
    n_obs : int
        Original number of observations
    df_resid : int
        Residual degrees of freedom
    vcov : str
        "iid" or "HC1"
    X : np.ndarray or scipy.sparse matrix
        Design matrix (compressed)
    x_cols : list of str
        Names of regressor columns (to extract subset of SEs)
        
    Returns
    -------
    np.ndarray
        Standard errors for x_cols only
    """
    k_x = len(x_cols)
    
    if vcov == "iid":
        sigma2 = rss_total / df_resid
        se_full = np.sqrt(np.diag(XtX_inv) * sigma2)
    elif vcov.upper() == "HC1":
        # Meat matrix: X' diag(rss_g) X
        if sparse.issparse(X):
            # Sparse: X.T @ diag(rss_g) @ X
            Xw = X.multiply(rss_g[:, np.newaxis])
            meat = (X.T @ Xw).toarray()
        else:
            meat = X.T @ (X * rss_g[:, np.newaxis])
        vcov_matrix = XtX_inv @ meat @ XtX_inv
        # HC1 adjustment
        adjustment = n_obs / df_resid
        se_full = np.sqrt(np.diag(vcov_matrix) * adjustment)
    else:
        raise ValueError(f"vcov must be 'iid' or 'HC1' for compress strategy, got '{vcov}'")
    
    # Return only SEs for x_cols (not FE dummies)
    return se_full[:k_x]


def leanfe_compress_polars(
    df: pl.DataFrame,
    y_col: str,
    x_cols: List[str],
    fe_cols: List[str],
    weights: Optional[str] = None,
    vcov: str = "iid"
) -> Dict:
    """
    Run compressed regression using Polars backend.
    
    Parameters
    ----------
    df : pl.DataFrame
        Input data
    y_col : str
        Dependent variable
    x_cols : list of str
        Regressors
    fe_cols : list of str
        Fixed effects
    weights : str, optional
        Weight column
    vcov : str
        "iid" or "HC1"
        
    Returns
    -------
    dict
        Regression results
    """
    # Compress data
    compressed, n_obs = compress_polars(df, y_col, x_cols, fe_cols, weights)
    n_compressed = len(compressed)
    
    # Build design matrix
    X, Y, wts, all_cols, n_fe_levels = build_design_matrix(
        compressed, x_cols, fe_cols, backend="polars"
    )
    
    # Solve WLS
    beta, XtX_inv = solve_wls(X, Y, wts)
    
    # Compute RSS
    rss_total, rss_g = compute_rss_grouped(compressed, X, beta, backend="polars")
    
    # Degrees of freedom
    p = len(all_cols)
    df_resid = n_obs - p
    
    # Standard errors
    se = compute_se_compress(XtX_inv, rss_total, rss_g, n_obs, df_resid, vcov, X, x_cols)
    
    # Extract coefficients for x_cols only
    beta_x = beta[:len(x_cols)]
    
    return {
        "coefficients": dict(zip(x_cols, beta_x)),
        "std_errors": dict(zip(x_cols, se)),
        "n_obs": n_obs,
        "n_compressed": n_compressed,
        "compression_ratio": n_compressed / n_obs,
        "vcov_type": vcov,
        "strategy": "compress",
        "df_resid": df_resid,
        "rss": rss_total,
    }


def leanfe_compress_duckdb(
    con,
    y_col: str,
    x_cols: List[str],
    fe_cols: List[str],
    weights: Optional[str] = None,
    vcov: str = "iid"
) -> Dict:
    """
    Run compressed regression using DuckDB backend.
    
    Parameters
    ----------
    con : duckdb.Connection
        DuckDB connection with 'data' table
    y_col : str
        Dependent variable
    x_cols : list of str
        Regressors
    fe_cols : list of str
        Fixed effects
    weights : str, optional
        Weight column
    vcov : str
        "iid" or "HC1"
        
    Returns
    -------
    dict
        Regression results
    """
    # Compress data
    compressed, n_obs = compress_duckdb(con, y_col, x_cols, fe_cols, weights)
    n_compressed = len(compressed)
    
    # Build design matrix
    X, Y, wts, all_cols, n_fe_levels = build_design_matrix(
        compressed, x_cols, fe_cols, backend="duckdb"
    )
    
    # Solve WLS
    beta, XtX_inv = solve_wls(X, Y, wts)
    
    # Compute RSS
    rss_total, rss_g = compute_rss_grouped(compressed, X, beta, backend="duckdb")
    
    # Degrees of freedom
    p = len(all_cols)
    df_resid = n_obs - p
    
    # Standard errors
    se = compute_se_compress(XtX_inv, rss_total, rss_g, n_obs, df_resid, vcov, X, x_cols)
    
    # Extract coefficients for x_cols only
    beta_x = beta[:len(x_cols)]
    
    return {
        "coefficients": dict(zip(x_cols, beta_x)),
        "std_errors": dict(zip(x_cols, se)),
        "n_obs": n_obs,
        "n_compressed": n_compressed,
        "compression_ratio": n_compressed / n_obs,
        "vcov_type": vcov,
        "strategy": "compress",
        "df_resid": df_resid,
        "rss": rss_total,
    }
