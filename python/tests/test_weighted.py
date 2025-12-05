"""Test weighted regression functionality."""

import polars as pl
import numpy as np
from leanfe import fast_feols_polars


def test_weighted_regression():
    """Test basic weighted regression."""
    np.random.seed(42)
    n = 1000
    
    x = np.random.randn(n)
    fe1 = np.random.randint(0, 10, n)
    weights = np.random.uniform(0.5, 2.0, n)
    y = 2.0 * x + np.random.randn(n)
    
    df = pl.DataFrame({
        "y": y,
        "x": x,
        "fe1": fe1,
        "weights": weights
    })
    
    result = fast_feols_polars(df, "y", ["x"], ["fe1"], weights="weights")
    
    assert "coefficients" in result
    assert "x" in result["coefficients"]
    assert abs(result["coefficients"]["x"] - 2.0) < 0.2  # Close to true value


def test_weighted_with_formula():
    """Test weighted regression with formula API."""
    np.random.seed(42)
    n = 1000
    
    df = pl.DataFrame({
        "y": np.random.randn(n),
        "x": np.random.randn(n),
        "fe1": np.random.randint(0, 10, n),
        "weights": np.random.uniform(0.5, 2.0, n)
    })
    
    result = fast_feols_polars(df, formula="y ~ x | fe1", weights="weights")
    
    assert "coefficients" in result
    assert "x" in result["coefficients"]


def test_weighted_with_robust_se():
    """Test weighted regression with HC1 standard errors."""
    np.random.seed(42)
    n = 1000
    
    df = pl.DataFrame({
        "y": np.random.randn(n),
        "x": np.random.randn(n),
        "fe1": np.random.randint(0, 10, n),
        "weights": np.random.uniform(0.5, 2.0, n)
    })
    
    result = fast_feols_polars(df, "y", ["x"], ["fe1"], 
                               weights="weights", vcov="HC1")
    
    assert result["vcov_type"] == "HC1"
    assert "std_errors" in result
