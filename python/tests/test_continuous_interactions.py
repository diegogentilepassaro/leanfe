"""Test binary treatment × continuous variable interactions."""
import polars as pl
import numpy as np
import pytest

from leanfe.polars_impl import fast_feols_polars
from leanfe.duckdb_impl import fast_feols_duckdb


@pytest.fixture
def binary_continuous_data():
    """Generate test data with binary treatment × continuous variable."""
    np.random.seed(42)
    n = 1000
    
    df = pl.DataFrame({
        'customer_id': np.repeat(np.arange(100), 10),
        'product_id': np.tile(np.arange(50), 20),
        'treatment': np.random.binomial(1, 0.5, n),
        'price': np.random.uniform(10, 100, n),  # Continuous moderator
    })
    
    # Generate outcome with interaction effect
    # True model: y = 10 + 2*treatment + 0.5*price + 1.5*treatment*price + FE + noise
    df = df.with_columns([
        (
            10.0 +
            2.0 * pl.col('treatment') +
            0.5 * pl.col('price') +
            1.5 * pl.col('treatment') * pl.col('price') +
            pl.col('customer_id').cast(pl.Float64) * 0.01 +
            pl.lit(np.random.normal(0, 1, n))
        ).alias('revenue')
    ])
    
    return df


def test_binary_continuous_interaction_polars(binary_continuous_data):
    """Test binary treatment × continuous variable interaction (Polars)."""
    # Create interaction manually
    df = binary_continuous_data.with_columns([
        (pl.col('treatment') * pl.col('price')).alias('treatment_x_price')
    ])
    
    result = fast_feols_polars(
        df,
        formula="revenue ~ treatment + price + treatment_x_price | customer_id + product_id"
    )
    
    # Check all coefficients exist
    assert 'treatment' in result['coefficients']
    assert 'price' in result['coefficients']
    assert 'treatment_x_price' in result['coefficients']
    
    # Check coefficients are close to true values (within 0.3)
    assert abs(result['coefficients']['treatment'] - 2.0) < 0.3
    assert abs(result['coefficients']['price'] - 0.5) < 0.3
    assert abs(result['coefficients']['treatment_x_price'] - 1.5) < 0.3


def test_binary_continuous_interaction_duckdb(binary_continuous_data):
    """Test binary treatment × continuous variable interaction (DuckDB)."""
    # Create interaction manually
    df = binary_continuous_data.with_columns([
        (pl.col('treatment') * pl.col('price')).alias('treatment_x_price')
    ])
    
    result = fast_feols_duckdb(
        df,
        formula="revenue ~ treatment + price + treatment_x_price | customer_id + product_id"
    )
    
    # Check all coefficients exist
    assert 'treatment' in result['coefficients']
    assert 'price' in result['coefficients']
    assert 'treatment_x_price' in result['coefficients']
    
    # Check coefficients are close to true values
    assert abs(result['coefficients']['treatment'] - 2.0) < 0.3
    assert abs(result['coefficients']['price'] - 0.5) < 0.3
    assert abs(result['coefficients']['treatment_x_price'] - 1.5) < 0.3


def test_continuous_main_effect_only(binary_continuous_data):
    """Test continuous variable as main effect (no interaction)."""
    result = fast_feols_polars(
        binary_continuous_data,
        formula="revenue ~ price | customer_id + product_id"
    )
    
    # Should estimate coefficient (captures both main effect and interaction)
    assert 'price' in result['coefficients']
    assert result['coefficients']['price'] > 0


def test_multiple_continuous_variables():
    """Test multiple continuous variables."""
    np.random.seed(42)
    n = 1000
    
    df = pl.DataFrame({
        'customer_id': np.repeat(np.arange(100), 10),
        'product_id': np.tile(np.arange(50), 20),
        'price': np.random.uniform(10, 100, n),
        'quantity': np.random.uniform(1, 10, n),
    })
    
    df = df.with_columns([
        (
            10.0 +
            0.5 * pl.col('price') +
            2.0 * pl.col('quantity') +
            pl.col('customer_id').cast(pl.Float64) * 0.01 +
            pl.lit(np.random.normal(0, 1, n))
        ).alias('revenue')
    ])
    
    result = fast_feols_polars(
        df,
        formula="revenue ~ price + quantity | customer_id + product_id"
    )
    
    # Check both coefficients
    assert 'price' in result['coefficients']
    assert 'quantity' in result['coefficients']
    assert abs(result['coefficients']['price'] - 0.5) < 0.3
    assert abs(result['coefficients']['quantity'] - 2.0) < 0.3
