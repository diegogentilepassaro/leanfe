import polars as pl
import numpy as np
import pytest
from leanfe.polars_impl import fast_feols_polars

def test_interaction_basic():
    """Test basic interaction: treatment:i(region) - first category dropped as reference"""
    # Create test data with repeated FE groups
    df = pl.DataFrame({
        'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] * 200,
        'treatment': [0, 1, 0, 1, 0, 1] * 200,
        'region': ['A', 'A', 'B', 'B', 'C', 'C'] * 200,
        'fe1': [i % 100 for i in range(1200)],  # Repeated groups
        'fe2': [i % 50 for i in range(1200)]
    })
    
    # Interaction syntax with robust SE
    result = fast_feols_polars(df, formula="y ~ treatment:i(region) | fe1 + fe2", vcov="HC1")
    
    # Verify interaction terms created (first category A is reference, dropped)
    assert 'treatment_B' in result['coefficients']
    assert 'treatment_C' in result['coefficients']
    assert len(result['coefficients']) == 2  # B and C only, A is reference
    
    # Verify all have standard errors
    assert all(se >= 0 for se in result['std_errors'].values())

def test_interaction_robust_se():
    """Test interaction with robust standard errors"""
    df = pl.DataFrame({
        'y': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] * 200,
        'treatment': [0, 1, 0, 1, 0, 1] * 200,
        'region': ['A', 'A', 'B', 'B', 'C', 'C'] * 200,
        'fe1': [i % 100 for i in range(1200)],
        'fe2': [i % 50 for i in range(1200)]
    })
    
    result = fast_feols_polars(df, formula="y ~ treatment:i(region) | fe1 + fe2", vcov="HC1")
    
    assert result['vcov_type'] == 'HC1'
    assert 'treatment_B' in result['std_errors']  # A is reference
    assert 'treatment_C' in result['std_errors']
    assert all(se >= 0 for se in result['std_errors'].values())

def test_interaction_clustered_se():
    """Test interaction with clustered standard errors"""
    np.random.seed(42)
    n = 3000
    
    df = pl.DataFrame({
        'treatment': np.random.binomial(1, 0.5, n),
        'region': np.random.choice(['A', 'B', 'C'], n),
        'fe1': np.repeat(np.arange(500), 6),  # 500 clusters, 6 obs each
        'fe2': np.tile(np.arange(250), 12)
    })
    
    # Generate outcome with treatment effects by region
    df = df.with_columns([
        (
            10.0 +
            (pl.col('treatment') * (pl.col('region') == 'A')).cast(pl.Float64) * 0.5 +
            (pl.col('treatment') * (pl.col('region') == 'B')).cast(pl.Float64) * 1.0 +
            (pl.col('treatment') * (pl.col('region') == 'C')).cast(pl.Float64) * 1.5 +
            pl.col('fe1').cast(pl.Float64) * 0.01 +
            pl.lit(np.random.normal(0, 1, n))
        ).alias('y')
    ])
    
    result = fast_feols_polars(df, formula="y ~ treatment:i(region) | fe1 + fe2", 
                               vcov="cluster", cluster_cols=["fe1"])
    
    assert result['vcov_type'] == 'cluster'
    assert result['n_clusters'] == 500
    assert all(se > 0 for se in result['std_errors'].values())
