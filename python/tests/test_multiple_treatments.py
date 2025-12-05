"""Test multiple treatment arms functionality."""
import polars as pl
import numpy as np
import pytest

from leanfe.polars_impl import fast_feols_polars
from leanfe.duckdb_impl import fast_feols_duckdb


@pytest.fixture
def multi_treatment_data():
    """Generate test data with multiple treatment arms."""
    np.random.seed(42)
    n = 10000
    
    df = pl.DataFrame({
        'customer_id': np.repeat(np.arange(1000), 10),
        'product_id': np.tile(np.arange(100), 100),
        'region': np.random.choice(['R1', 'R2', 'R3'], n),
        'treatment_A': np.random.binomial(1, 0.3, n),
        'treatment_B': np.random.binomial(1, 0.4, n),
        'treatment_C': np.random.choice([0, 1, 2], n),  # Categorical treatment
    })
    
    # Generate outcome with known effects
    df = df.with_columns([
        (
            10.0 +
            0.5 * pl.col('treatment_A') +
            0.8 * pl.col('treatment_B') +
            1.2 * pl.col('treatment_C') +
            pl.col('customer_id').cast(pl.Float64) * 0.01 +
            pl.col('product_id').cast(pl.Float64) * 0.02 +
            pl.lit(np.random.normal(0, 1, n))
        ).alias('revenue')
    ])
    
    return df


def test_multiple_binary_treatments_polars(multi_treatment_data):
    """Test multiple binary treatments with Polars."""
    result = fast_feols_polars(
        multi_treatment_data,
        formula="revenue ~ treatment_A + treatment_B | customer_id + product_id"
    )
    
    # Check coefficients exist
    assert 'treatment_A' in result['coefficients']
    assert 'treatment_B' in result['coefficients']
    
    # Check coefficients are reasonable (within 0.3 of true values)
    assert abs(result['coefficients']['treatment_A'] - 0.5) < 0.3
    assert abs(result['coefficients']['treatment_B'] - 0.8) < 0.3
    
    # Check standard errors exist
    assert 'treatment_A' in result['std_errors']
    assert 'treatment_B' in result['std_errors']


def test_multiple_binary_treatments_duckdb(multi_treatment_data):
    """Test multiple binary treatments with DuckDB."""
    result = fast_feols_duckdb(
        multi_treatment_data,
        formula="revenue ~ treatment_A + treatment_B | customer_id + product_id"
    )
    
    # Check coefficients exist
    assert 'treatment_A' in result['coefficients']
    assert 'treatment_B' in result['coefficients']
    
    # Check coefficients are reasonable
    assert abs(result['coefficients']['treatment_A'] - 0.5) < 0.3
    assert abs(result['coefficients']['treatment_B'] - 0.8) < 0.3


def test_mixed_treatment_types_polars(multi_treatment_data):
    """Test mixing binary and categorical treatments with Polars."""
    result = fast_feols_polars(
        multi_treatment_data,
        formula="revenue ~ treatment_A + treatment_B + treatment_C | customer_id + product_id"
    )
    
    # Check all coefficients exist
    assert 'treatment_A' in result['coefficients']
    assert 'treatment_B' in result['coefficients']
    assert 'treatment_C' in result['coefficients']
    
    # Check coefficients are reasonable
    assert abs(result['coefficients']['treatment_A'] - 0.5) < 0.3
    assert abs(result['coefficients']['treatment_B'] - 0.8) < 0.3
    assert abs(result['coefficients']['treatment_C'] - 1.2) < 0.3


def test_multiple_interactions_polars(multi_treatment_data):
    """Test multiple treatment interactions with Polars."""
    result = fast_feols_polars(
        multi_treatment_data,
        formula="revenue ~ treatment_A:i(region) + treatment_B:i(region) | customer_id + product_id"
    )
    
    # Check interaction coefficients exist for both treatments
    # First region (R1) is reference, so only R2 and R3 coefficients
    treatment_a_coeffs = [k for k in result['coefficients'].keys() if k.startswith('treatment_A_')]
    treatment_b_coeffs = [k for k in result['coefficients'].keys() if k.startswith('treatment_B_')]
    
    assert len(treatment_a_coeffs) == 2  # R2 and R3 (R1 is reference)
    assert len(treatment_b_coeffs) == 2  # R2 and R3 (R1 is reference)
    
    # Check standard errors exist
    for coef in treatment_a_coeffs + treatment_b_coeffs:
        assert coef in result['std_errors']


def test_multiple_interactions_duckdb(multi_treatment_data):
    """Test multiple treatment interactions with DuckDB."""
    result = fast_feols_duckdb(
        multi_treatment_data,
        formula="revenue ~ treatment_A:i(region) + treatment_B:i(region) | customer_id + product_id"
    )
    
    # Check interaction coefficients exist for both treatments
    # First region (R1) is reference, so only R2 and R3 coefficients
    treatment_a_coeffs = [k for k in result['coefficients'].keys() if k.startswith('treatment_A_')]
    treatment_b_coeffs = [k for k in result['coefficients'].keys() if k.startswith('treatment_B_')]
    
    assert len(treatment_a_coeffs) == 2  # R2 and R3 (R1 is reference)
    assert len(treatment_b_coeffs) == 2  # R2 and R3 (R1 is reference)


def test_multiple_treatments_with_robust_se(multi_treatment_data):
    """Test multiple treatments with HC1 robust standard errors."""
    result = fast_feols_polars(
        multi_treatment_data,
        formula="revenue ~ treatment_A + treatment_B | customer_id + product_id",
        vcov="HC1"
    )
    
    assert result['vcov_type'] == 'HC1'
    assert 'treatment_A' in result['coefficients']
    assert 'treatment_B' in result['coefficients']
    assert result['std_errors']['treatment_A'] > 0
    assert result['std_errors']['treatment_B'] > 0


def test_multiple_treatments_with_clustered_se(multi_treatment_data):
    """Test multiple treatments with clustered standard errors."""
    result = fast_feols_polars(
        multi_treatment_data,
        formula="revenue ~ treatment_A + treatment_B | customer_id + product_id",
        vcov="cluster",
        cluster_cols=["region"]
    )
    
    assert result['vcov_type'] == 'cluster'
    assert result['n_clusters'] == 3
    assert 'treatment_A' in result['coefficients']
    assert 'treatment_B' in result['coefficients']


def test_multiple_interactions_different_factors(multi_treatment_data):
    """Test interactions with different factor variables."""
    # Add time variable
    df = multi_treatment_data.with_columns([
        (pl.col('customer_id') % 5).alias('time_period')
    ])
    
    result = fast_feols_polars(
        df,
        formula="revenue ~ treatment_A:i(region) + treatment_B:i(time_period) | customer_id + product_id"
    )
    
    # Check treatment_A has 2 coefficients (3 regions - 1 reference)
    treatment_a_coeffs = [k for k in result['coefficients'].keys() if k.startswith('treatment_A_')]
    assert len(treatment_a_coeffs) == 2  # R2 and R3 (R1 is reference)
    
    # Check treatment_B has 4 coefficients (5 time periods - 1 reference)
    treatment_b_coeffs = [k for k in result['coefficients'].keys() if k.startswith('treatment_B_')]
    assert len(treatment_b_coeffs) == 4  # First time period is reference
