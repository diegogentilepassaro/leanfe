import polars as pl
import numpy as np
from leanfe import fast_feols_polars, fast_feols_duckdb

def test_factor_variables_polars():
    """Test factor variable expansion into dummies (first category dropped as reference)"""
    np.random.seed(42)
    n = 1000
    
    df = pl.DataFrame({
        'y': np.random.randn(n),
        'x': np.random.randn(n),
        'region': np.random.choice(['A', 'B', 'C'], n),
        'fe1': np.random.randint(0, 10, n),
        'fe2': np.random.randint(0, 5, n)
    })
    
    # Test with factor variable (expanded to dummies, first category dropped)
    result = fast_feols_polars(df, formula="y ~ x + i(region) | fe1 + fe2")
    
    # Should have coefficient for x plus 2 region dummies (A is reference)
    assert 'x' in result['coefficients']
    assert 'region_B' in result['coefficients']
    assert 'region_C' in result['coefficients']
    assert len(result['coefficients']) == 3  # x + 2 region dummies
    
    print("✓ Polars factor variables test passed")
    print(f"  Coefficients: {list(result['coefficients'].keys())}")
    print(f"  (region_A is reference category)")

def test_factor_variables_duckdb():
    """Test factor variable expansion into dummies with DuckDB"""
    np.random.seed(42)
    n = 1000
    
    df = pl.DataFrame({
        'y': np.random.randn(n),
        'x': np.random.randn(n),
        'region': np.random.choice(['A', 'B', 'C'], n),
        'fe1': np.random.randint(0, 10, n),
        'fe2': np.random.randint(0, 5, n)
    })
    
    # Test with factor variable (expanded to dummies, first category dropped)
    result = fast_feols_duckdb(df, formula="y ~ x + i(region) | fe1 + fe2")
    
    # Should have coefficient for x plus 2 region dummies (A is reference)
    assert 'x' in result['coefficients']
    assert 'region_B' in result['coefficients']
    assert 'region_C' in result['coefficients']
    assert len(result['coefficients']) == 3
    
    print("✓ DuckDB factor variables test passed")
    print(f"  Coefficients: {list(result['coefficients'].keys())}")
    print(f"  (region_A is reference category)")

if __name__ == "__main__":
    test_factor_variables_polars()
    test_factor_variables_duckdb()
    print("\n✓ All factor variable tests passed!")

