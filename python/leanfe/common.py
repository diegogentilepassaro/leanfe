"""
Common utilities shared between Polars and DuckDB backends.

Contains formula parsing, IV estimation, and other shared logic.
"""

import re
import numpy as np
from typing import List, Optional, Tuple

from .result import LeanFEResult


def _parse_i_term(term: str) -> Tuple[str, Optional[str]]:
    """
    Parse i() term with optional reference category.
    
    Supports:
    - i(var) -> (var, None) - first category is reference
    - i(var, ref=value) -> (var, "value") - specified reference
    - i(var, ref="value") -> (var, "value") - quoted reference
    
    Returns
    -------
    tuple
        (variable_name, reference_category_or_None)
    """
    # Match i(var) or i(var, ref=value) or i(var, ref="value")
    match = re.match(r'i\((\w+)(?:\s*,\s*ref\s*=\s*["\']?([^"\')\s]+)["\']?)?\)', term)
    if match:
        return match.group(1), match.group(2)
    raise ValueError(f"Invalid i() syntax: {term}. Use i(var) or i(var, ref=value)")


def parse_formula(formula: str) -> Tuple[str, List[str], List[str], List[Tuple[str, Optional[str]]], List[Tuple[str, str, Optional[str]]], List[str]]:
    """
    Parse R-style formula into components.
    
    Supports:
    - Basic: 'y ~ x1 + x2 | fe1 + fe2'
    - Factor variables: 'y ~ x1 + i(region) | fe1'
    - Factor with reference: 'y ~ x1 + i(region, ref=R1) | fe1'
    - Interactions: 'y ~ x1 + treatment:i(region) | fe1'
    - Interactions with reference: 'y ~ x1 + treatment:i(region, ref=R1) | fe1'
    - IV/2SLS: 'y ~ x1 + x2 | fe1 + fe2 | z1 + z2'
    
    Parameters
    ----------
    formula : str
        R-style formula string
        
    Returns
    -------
    tuple
        (y_col, x_cols, fe_cols, factor_vars, interactions, instruments)
        - y_col: dependent variable name
        - x_cols: list of independent variable names
        - fe_cols: list of fixed effect column names
        - factor_vars: list of (var, ref) tuples where ref is None or reference category
        - interactions: list of (var, factor, ref) tuples for var:i(factor) terms
        - instruments: list of instrument variable names (for IV)
    """
    parts = formula.split('|')
    if len(parts) < 2:
        raise ValueError("Formula must include fixed effects: 'y ~ x | fe1 + fe2'")
    if len(parts) > 3:
        raise ValueError("Formula has too many parts. Use: 'y ~ x | fe' or 'y ~ x | fe | z' (IV)")
    
    # Parse left-hand side (y ~ x terms)
    lhs_rhs = parts[0].split('~')
    if len(lhs_rhs) != 2:
        raise ValueError("Formula must have exactly one '~' separating y and x variables")
    
    y_col = lhs_rhs[0].strip()
    
    # Parse x terms
    x_terms = [x.strip() for x in lhs_rhs[1].split('+')]
    x_cols = []
    factor_vars = []
    interactions = []
    
    for term in x_terms:
        if ':i(' in term and term.endswith(')'):
            # Interaction term: var:i(factor) or var:i(factor, ref=value)
            match = re.match(r'(\w+):i\((\w+)(?:\s*,\s*ref\s*=\s*["\']?([^"\')\s]+)["\']?)?\)', term)
            if match:
                interactions.append((match.group(1), match.group(2), match.group(3)))
            else:
                raise ValueError(f"Invalid interaction syntax: {term}")
        elif term.startswith('i(') and term.endswith(')'):
            # Factor variable: i(var) or i(var, ref=value)
            var, ref = _parse_i_term(term)
            factor_vars.append((var, ref))
        else:
            # Regular variable
            x_cols.append(term)
    
    # Parse fixed effects
    fe_cols = [f.strip() for f in parts[1].split('+')]
    
    # Parse instruments (if present)
    instruments = []
    if len(parts) == 3:
        instruments = [z.strip() for z in parts[2].split('+')]
    
    return y_col, x_cols, fe_cols, factor_vars, interactions, instruments


def iv_2sls(
    Y: np.ndarray, 
    X: np.ndarray, 
    Z: np.ndarray, 
    weights: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Two-Stage Least Squares (2SLS) estimation.
    
    Parameters
    ----------
    Y : np.ndarray
        Dependent variable (n,)
    X : np.ndarray
        Endogenous regressors (n, k)
    Z : np.ndarray
        Instruments (n, m) where m >= k
    weights : np.ndarray, optional
        Regression weights (n,)
        
    Returns
    -------
    tuple
        (beta, X_hat) where beta is coefficient vector and X_hat is fitted values from first stage
    """
    if Z.shape[1] < X.shape[1]:
        raise ValueError(f"Under-identified: {Z.shape[1]} instruments for {X.shape[1]} endogenous variables")
    
    if weights is not None:
        sqrt_w = np.sqrt(weights)
        Z_w = Z * sqrt_w[:, np.newaxis]
        X_w = X * sqrt_w[:, np.newaxis]
        Y_w = Y * sqrt_w
        
        # First stage: X = Z @ gamma
        ZtZ = Z_w.T @ Z_w
        ZtX = Z_w.T @ X_w
        gamma = np.linalg.solve(ZtZ, ZtX)
        X_hat = Z @ gamma
        
        # Second stage: Y = X_hat @ beta
        X_hat_w = X_hat * sqrt_w[:, np.newaxis]
        XhtXh = X_hat_w.T @ X_hat_w
        XhtY = X_hat_w.T @ Y_w
        beta = np.linalg.solve(XhtXh, XhtY)
    else:
        # First stage
        ZtZ = Z.T @ Z
        ZtX = Z.T @ X
        gamma = np.linalg.solve(ZtZ, ZtX)
        X_hat = Z @ gamma
        
        # Second stage
        XhtXh = X_hat.T @ X_hat
        XhtY = X_hat.T @ Y
        beta = np.linalg.solve(XhtXh, XhtY)
    
    return beta, X_hat


def compute_standard_errors(
    XtX_inv: np.ndarray,
    resid: np.ndarray,
    n_obs: int,
    df_resid: int,
    vcov: str,
    X: np.ndarray,
    weights: Optional[np.ndarray] = None,
    cluster_ids: Optional[np.ndarray] = None,
    ssc: bool = False
) -> Tuple[np.ndarray, Optional[int]]:
    """
    Compute standard errors for OLS/IV coefficients.
    
    Parameters
    ----------
    XtX_inv : np.ndarray
        Inverse of X'X matrix (k, k)
    resid : np.ndarray
        Residuals (n,)
    n_obs : int
        Number of observations
    df_resid : int
        Residual degrees of freedom
    vcov : str
        Type of variance estimator: "iid", "HC1", or "cluster"
    X : np.ndarray
        Regressor matrix (n, k) - use X_hat for IV
    weights : np.ndarray, optional
        Regression weights (n,)
    cluster_ids : np.ndarray, optional
        Cluster identifiers (n,) - required for vcov="cluster"
    ssc : bool
        Small sample correction for clustered SEs
        
    Returns
    -------
    tuple
        (se, n_clusters) where se is standard errors and n_clusters is cluster count (or None)
    """
    n_clusters = None
    
    if vcov == "iid":
        if weights is not None:
            sigma2 = np.sum(weights * resid**2) / df_resid
        else:
            sigma2 = np.sum(resid**2) / df_resid
        se = np.sqrt(np.diag(XtX_inv) * sigma2)
        
    elif vcov == "HC1":
        if weights is not None:
            meat = X.T @ (X * (weights * resid**2)[:, np.newaxis])
        else:
            meat = X.T @ (X * (resid**2)[:, np.newaxis])
        vcov_matrix = XtX_inv @ meat @ XtX_inv
        adjustment = n_obs / df_resid
        se = np.sqrt(np.diag(vcov_matrix) * adjustment)
        
    elif vcov == "cluster":
        if cluster_ids is None:
            raise ValueError("cluster_ids required for vcov='cluster'")
        
        # Get unique clusters
        unique_clusters = np.unique(cluster_ids)
        n_clusters = len(unique_clusters)
        k = X.shape[1]
        
        # Compute cluster scores: score_g = sum(X_i * e_i) for each cluster g
        scores = np.zeros((n_clusters, k))
        for g, cluster in enumerate(unique_clusters):
            mask = cluster_ids == cluster
            if weights is not None:
                scores[g] = np.sum(X[mask] * (resid[mask] * weights[mask])[:, np.newaxis], axis=0)
            else:
                scores[g] = np.sum(X[mask] * resid[mask][:, np.newaxis], axis=0)
        
        # Meat matrix: S'S
        meat = scores.T @ scores
        
        # Cluster adjustment
        if ssc:
            adjustment = (n_clusters / (n_clusters - 1)) * ((n_obs - 1) / df_resid)
        else:
            adjustment = n_clusters / (n_clusters - 1)
        
        vcov_matrix = XtX_inv @ meat @ XtX_inv * adjustment
        se = np.sqrt(np.diag(vcov_matrix))
        
    else:
        raise ValueError(f"vcov must be 'iid', 'HC1', or 'cluster', got '{vcov}'")
    
    return se, n_clusters


def build_result(
    x_cols: List[str],
    beta: np.ndarray,
    se: np.ndarray,
    n_obs: int,
    iterations: int,
    vcov: str,
    is_iv: bool,
    n_instruments: Optional[int],
    n_clusters: Optional[int],
    df_resid: Optional[int] = None,
    r_squared_within: Optional[float] = None,
    formula: Optional[str] = None,
    fe_cols: Optional[List[str]] = None
) -> LeanFEResult:
    """
    Build LeanFEResult object.
    
    Returns
    -------
    LeanFEResult
        Result object with formatted output
    """
    return LeanFEResult(
        coefficients=dict(zip(x_cols, beta)),
        std_errors=dict(zip(x_cols, se)),
        n_obs=n_obs,
        iterations=iterations,
        vcov_type=vcov,
        is_iv=is_iv,
        n_instruments=n_instruments if is_iv else None,
        n_clusters=n_clusters,
        df_resid=df_resid,
        r_squared_within=r_squared_within,
        formula=formula,
        fe_cols=fe_cols
    )


# Keep old function for backwards compatibility
def build_result_dict(
    x_cols: List[str],
    beta: np.ndarray,
    se: np.ndarray,
    n_obs: int,
    iterations: int,
    vcov: str,
    is_iv: bool,
    n_instruments: Optional[int],
    n_clusters: Optional[int]
) -> dict:
    """
    Build standardized result dictionary.
    
    DEPRECATED: Use build_result() instead which returns LeanFEResult.
    
    Returns
    -------
    dict
        Standardized output with coefficients, std_errors, n_obs, iterations,
        vcov_type, is_iv, n_instruments, n_clusters
    """
    return {
        'coefficients': dict(zip(x_cols, beta)),
        'std_errors': dict(zip(x_cols, se)),
        'n_obs': n_obs,
        'iterations': iterations,
        'vcov_type': vcov,
        'is_iv': is_iv,
        'n_instruments': n_instruments if is_iv else None,
        'n_clusters': n_clusters
    }
