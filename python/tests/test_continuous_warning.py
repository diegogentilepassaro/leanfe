"""Test continuous treatment variable warning."""
import polars as pl
import numpy as np
import pytest
import warnings

from leanfe.polars_impl import fast_feols_polars
from leanfe.duckdb_impl import fast_feols_duckdb


@pytest.fixture
def continuous_treatment_data():
    """Generate test data with continuous treatment."""
    np.random.seed(42)
    n = 1000
    
    df = pl.DataFrame({
        'customer_id': np.repeat(np.arange(100), 10),
        'product_id': np.tile(np.arange(50), 20),
        'treatment_continuous': np.random.uniform(0, 10, n),  # Continuous
        'treatment_binary': np.random.binomial(1, 0.5, n),    # Binary
    })
    
    df = df.with_columns([
        (
            10.0 +
            0.5 * pl.col('treatment_continuous') +
            1.0 * pl.col('treatment_binary') +
            pl.col('customer_id').cast(pl.Float64) * 0.01 +
            pl.lit(np.random.normal(0, 1, n))
        ).alias('revenue')
    ])
    
    return df


def test_continuous_treatment_warning_polars(continuous_treatment_data):
    """Test that continuous treatment triggers warning in Polars."""
    with pytest.warns(UserWarning, match="Continuous treatment variable"):
        result = fast_feols_polars(
            continuous_treatment_data,
            formula="revenue ~ treatment_continuous | customer_id + product_id"
        )
    
    # Should still run and produce results
    assert 'treatment_continuous' in result['coefficients']
    assert result['coefficients']['treatment_continuous'] > 0


def test_continuous_treatment_warning_duckdb(continuous_treatment_data):
    """Test that continuous treatment triggers warning in DuckDB."""
    with pytest.warns(UserWarning, match="Continuous treatment variable"):
        result = fast_feols_duckdb(
            continuous_treatment_data,
            formula="revenue ~ treatment_continuous | customer_id + product_id"
        )
    
    # Should still run and produce results
    assert 'treatment_continuous' in result['coefficients']
    assert result['coefficients']['treatment_continuous'] > 0


def test_binary_treatment_no_warning_polars(continuous_treatment_data):
    """Test that binary treatment does NOT trigger warning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Turn warnings into errors
        result = fast_feols_polars(
            continuous_treatment_data,
            formula="revenue ~ treatment_binary | customer_id + product_id"
        )
    
    # Should run without warning
    assert 'treatment_binary' in result['coefficients']


def test_mixed_treatments_warning_polars(continuous_treatment_data):
    """Test that mixed treatments (binary + continuous) triggers warning."""
    with pytest.warns(UserWarning, match="Continuous treatment variable"):
        result = fast_feols_polars(
            continuous_treatment_data,
            formula="revenue ~ treatment_binary + treatment_continuous | customer_id + product_id"
        )
    
    # Should still run and produce results for both
    assert 'treatment_binary' in result['coefficients']
    assert 'treatment_continuous' in result['coefficients']


def test_categorical_treatment_no_warning():
    """Test that categorical treatment (few unique values) does NOT trigger warning."""
    np.random.seed(42)
    n = 1000
    
    df = pl.DataFrame({
        'customer_id': np.repeat(np.arange(100), 10),
        'product_id': np.tile(np.arange(50), 20),
        'treatment_categorical': np.random.choice([0.0, 1.0, 2.0], n),  # Only 3 values
    })
    
    df = df.with_columns([
        (
            10.0 +
            1.0 * pl.col('treatment_categorical') +
            pl.col('customer_id').cast(pl.Float64) * 0.01 +
            pl.lit(np.random.normal(0, 1, n))
        ).alias('revenue')
    ])
    
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # Turn warnings into errors
        result = fast_feols_polars(
            df,
            formula="revenue ~ treatment_categorical | customer_id + product_id"
        )
    
    # Should run without warning (only 3 unique values)
    assert 'treatment_categorical' in result['coefficients']
