"""
Result class for leanfe regression output.

Provides a nice formatted table output similar to fixest/statsmodels.
"""

import numpy as np
from typing import Dict, List, Optional
from scipy import stats


class LeanFEResult:
    """
    Result object from leanfe regression.
    
    Provides formatted output and easy access to regression results.
    
    Attributes
    ----------
    coefficients : dict
        Coefficient estimates by variable name
    std_errors : dict
        Standard errors by variable name
    t_stats : dict
        t-statistics by variable name
    p_values : dict
        p-values by variable name
    n_obs : int
        Number of observations
    iterations : int
        Number of demeaning iterations
    vcov_type : str
        Type of variance-covariance estimator used
    is_iv : bool
        Whether IV/2SLS was used
    n_instruments : int or None
        Number of instruments (if IV)
    n_clusters : int or None
        Number of clusters (if clustered SEs)
    df_resid : int
        Residual degrees of freedom
    r_squared : float or None
        R-squared (within, if available)
    """
    
    def __init__(
        self,
        coefficients: Dict[str, float],
        std_errors: Dict[str, float],
        n_obs: int,
        iterations: int,
        vcov_type: str,
        is_iv: bool,
        n_instruments: Optional[int],
        n_clusters: Optional[int],
        df_resid: Optional[int] = None,
        r_squared: Optional[float] = None,
        r_squared_within: Optional[float] = None,
        rss: Optional[float] = None,
        tss: Optional[float] = None,
        formula: Optional[str] = None,
        fe_cols: Optional[List[str]] = None
    ):
        self.coefficients = coefficients
        self.std_errors = std_errors
        self.n_obs = n_obs
        self.iterations = iterations
        self.vcov_type = vcov_type
        self.is_iv = is_iv
        self.n_instruments = n_instruments
        self.n_clusters = n_clusters
        self.df_resid = df_resid or (n_obs - len(coefficients))
        self.r_squared = r_squared
        self.r_squared_within = r_squared_within
        self.rss = rss
        self.tss = tss
        self.formula = formula
        self.fe_cols = fe_cols or []
        
        # Compute t-stats and p-values
        self.t_stats = {}
        self.p_values = {}
        for var in coefficients:
            if std_errors[var] > 0:
                t = coefficients[var] / std_errors[var]
                self.t_stats[var] = t
                # Two-tailed p-value
                self.p_values[var] = 2 * (1 - stats.t.cdf(abs(t), self.df_resid))
            else:
                self.t_stats[var] = np.nan
                self.p_values[var] = np.nan
    
    def _significance_stars(self, p: float) -> str:
        """Return significance stars based on p-value."""
        if p < 0.001:
            return "***"
        elif p < 0.01:
            return "**"
        elif p < 0.05:
            return "*"
        elif p < 0.1:
            return "."
        return ""
    
    def summary(self) -> str:
        """Return a formatted summary table."""
        return str(self)
    
    def __repr__(self) -> str:
        """Short representation."""
        n_coef = len(self.coefficients)
        return f"LeanFEResult(n_obs={self.n_obs:,}, n_coef={n_coef}, vcov='{self.vcov_type}')"
    
    def __str__(self) -> str:
        """Formatted regression table output."""
        lines = []
        
        # Header
        lines.append("")
        lines.append("=" * 70)
        lines.append("                         leanfe Regression Results")
        lines.append("=" * 70)
        
        # Model info
        if self.formula:
            lines.append(f"Formula:      {self.formula}")
        lines.append(f"Observations: {self.n_obs:,}")
        if self.fe_cols:
            lines.append(f"Fixed Effects: {', '.join(self.fe_cols)}")
        if self.r_squared_within is not None:
            lines.append(f"R² (within):  {self.r_squared_within:.4f}")
        lines.append(f"Std. Errors:  {self._vcov_description()}")
        if self.n_clusters:
            lines.append(f"Clusters:     {self.n_clusters:,}")
        
        lines.append("-" * 70)
        
        # Column headers
        lines.append(f"{'Variable':<20} {'Estimate':>12} {'Std.Err':>12} {'t-stat':>10} {'p-value':>10}")
        lines.append("-" * 70)
        
        # Coefficients
        for var in self.coefficients:
            coef = self.coefficients[var]
            se = self.std_errors[var]
            t = self.t_stats[var]
            p = self.p_values[var]
            stars = self._significance_stars(p)
            
            # Truncate variable name if too long
            var_display = var[:18] + ".." if len(var) > 20 else var
            
            lines.append(
                f"{var_display:<20} {coef:>12.6f} {se:>12.6f} {t:>10.3f} {p:>9.4f}{stars}"
            )
        
        lines.append("-" * 70)
        lines.append("Signif. codes: 0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1")
        lines.append("=" * 70)
        lines.append("")
        
        return "\n".join(lines)
    
    def _vcov_description(self) -> str:
        """Return human-readable vcov description."""
        if self.vcov_type == "iid":
            return "IID"
        elif self.vcov_type == "HC1":
            return "Heteroskedasticity-robust (HC1)"
        elif self.vcov_type == "cluster":
            return f"Clustered ({self.n_clusters:,} clusters)"
        return self.vcov_type
    
    def coef(self, var: Optional[str] = None):
        """Get coefficient(s). If var is None, returns all as dict."""
        if var is None:
            return self.coefficients.copy()
        return self.coefficients.get(var)
    
    def se(self, var: Optional[str] = None):
        """Get standard error(s). If var is None, returns all as dict."""
        if var is None:
            return self.std_errors.copy()
        return self.std_errors.get(var)
    
    def tstat(self, var: Optional[str] = None):
        """Get t-statistic(s). If var is None, returns all as dict."""
        if var is None:
            return self.t_stats.copy()
        return self.t_stats.get(var)
    
    def pvalue(self, var: Optional[str] = None):
        """Get p-value(s). If var is None, returns all as dict."""
        if var is None:
            return self.p_values.copy()
        return self.p_values.get(var)
    
    def confint(self, level: float = 0.95) -> Dict[str, tuple]:
        """
        Compute confidence intervals.
        
        Parameters
        ----------
        level : float
            Confidence level (default 0.95 for 95% CI)
            
        Returns
        -------
        dict
            Dictionary mapping variable names to (lower, upper) tuples
        """
        alpha = 1 - level
        t_crit = stats.t.ppf(1 - alpha/2, self.df_resid)
        
        ci = {}
        for var in self.coefficients:
            coef = self.coefficients[var]
            se = self.std_errors[var]
            ci[var] = (coef - t_crit * se, coef + t_crit * se)
        return ci
    
    def to_dict(self) -> dict:
        """Convert to dictionary (for backwards compatibility)."""
        return {
            'coefficients': self.coefficients,
            'std_errors': self.std_errors,
            't_stats': self.t_stats,
            'p_values': self.p_values,
            'n_obs': self.n_obs,
            'iterations': self.iterations,
            'vcov_type': self.vcov_type,
            'is_iv': self.is_iv,
            'n_instruments': self.n_instruments,
            'n_clusters': self.n_clusters,
            'df_resid': self.df_resid,
            'r_squared_within': self.r_squared_within
        }
    
    # Allow dict-like access for backwards compatibility
    def __getitem__(self, key):
        return self.to_dict()[key]
    
    def keys(self):
        return self.to_dict().keys()
    
    def values(self):
        return self.to_dict().values()
    
    def items(self):
        return self.to_dict().items()
