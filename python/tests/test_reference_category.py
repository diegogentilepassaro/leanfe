"""Tests for reference category selection in factor variables and interactions."""

import pytest
import polars as pl
import numpy as np

from leanfe import fast_feols_polars, fast_feols_duckdb


@pytest.fixture
def region_data():
    """Create test data with region factor."""
    np.random.seed(42)
    n = 1000
    
    df = pl.DataFrame({
        "customer_id": np.repeat(np.arange(100), 10),
        "product_id": np.tile(np.arange(50), 20),
        "region": np.random.choice(["R1", "R2", "R3"], n),
        "treatment": np.random.binomial(1, 0.5, n),
    })
    
    # Generate outcome
    df = df.with_columns([
        (10.0 + 
         pl.col("treatment") * 0.5 +
         pl.col("customer_id").cast(pl.Float64) * 0.01 +
         pl.Series(np.random.normal(0, 1, n))
        ).alias("revenue")
    ])
    
    return df


def test_default_reference_first_polars(region_data):
    """Test that default reference is first category (alphabetically)."""
    result = fast_feols_polars(
        region_data,
        formula="revenue ~ treatment + i(region) | customer_id + product_id"
    )
    
    # R1 should be reference (first alphabetically), so we get R2 and R3
    assert "region_R2" in result["coefficients"]
    assert "region_R3" in result["coefficients"]
    assert "region_R1" not in result["coefficients"]


def test_custom_reference_polars(region_data):
    """Test specifying custom reference category in Polars."""
    result = fast_feols_polars(
        region_data,
        formula="revenue ~ treatment + i(region, ref=R2) | customer_id + product_id"
    )
    
    # R2 should be reference, so we get R1 and R3
    assert "region_R1" in result["coefficients"]
    assert "region_R3" in result["coefficients"]
    assert "region_R2" not in result["coefficients"]


def test_custom_reference_duckdb(region_data):
    """Test specifying custom reference category in DuckDB."""
    result = fast_feols_duckdb(
        region_data,
        formula="revenue ~ treatment + i(region, ref=R3) | customer_id + product_id"
    )
    
    # R3 should be reference, so we get R1 and R2
    assert "region_R1" in result["coefficients"]
    assert "region_R2" in result["coefficients"]
    assert "region_R3" not in result["coefficients"]


def test_interaction_custom_reference_polars(region_data):
    """Test specifying custom reference in interaction term."""
    result = fast_feols_polars(
        region_data,
        formula="revenue ~ treatment:i(region, ref=R2) | customer_id + product_id"
    )
    
    # R2 should be reference, so we get treatment_R1 and treatment_R3
    assert "treatment_R1" in result["coefficients"]
    assert "treatment_R3" in result["coefficients"]
    assert "treatment_R2" not in result["coefficients"]


def test_interaction_custom_reference_duckdb(region_data):
    """Test specifying custom reference in interaction term (DuckDB)."""
    result = fast_feols_duckdb(
        region_data,
        formula="revenue ~ treatment:i(region, ref=R1) | customer_id + product_id"
    )
    
    # R1 should be reference, so we get treatment_R2 and treatment_R3
    assert "treatment_R2" in result["coefficients"]
    assert "treatment_R3" in result["coefficients"]
    assert "treatment_R1" not in result["coefficients"]


def test_invalid_reference_raises_error(region_data):
    """Test that invalid reference category raises error."""
    with pytest.raises(ValueError, match="Reference category.*not found"):
        fast_feols_polars(
            region_data,
            formula="revenue ~ treatment + i(region, ref=INVALID) | customer_id + product_id"
        )


def test_quoted_reference_polars(region_data):
    """Test that quoted reference values work."""
    result = fast_feols_polars(
        region_data,
        formula='revenue ~ treatment + i(region, ref="R2") | customer_id + product_id'
    )
    
    # R2 should be reference
    assert "region_R1" in result["coefficients"]
    assert "region_R3" in result["coefficients"]
    assert "region_R2" not in result["coefficients"]
