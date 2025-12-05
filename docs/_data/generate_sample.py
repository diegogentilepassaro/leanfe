"""
Synthetic panel data generation for leanfe documentation.

Generates reproducible synthetic datasets with realistic panel structure
for tutorials and benchmarks. All examples use 1M+ observations to
demonstrate leanfe's efficiency from the first example.

Usage:
    from _data.generate_sample import generate_panel_data
    
    # Tutorial data (1M obs)
    df = generate_panel_data(n_obs=1_000_000, seed=42)
    
    # Benchmark data (10M obs, high cardinality FEs)
    df = generate_panel_data(n_obs=10_000_000, fe_cardinality="high", seed=42)
"""

import numpy as np
import polars as pl
from typing import Literal


# Cardinality configurations for fixed effects
FE_CARDINALITY_CONFIG = {
    "low": {
        "n_units": 100,      # e.g., 100 firms
        "n_products": 50,    # e.g., 50 product categories
        "n_regions": 4,      # e.g., 4 regions
    },
    "medium": {
        "n_units": 1_000,    # e.g., 1,000 firms
        "n_products": 500,   # e.g., 500 products
        "n_regions": 10,     # e.g., 10 regions
    },
    "high": {
        "n_units": 10_000,   # e.g., 10,000 firms
        "n_products": 5_000, # e.g., 5,000 products
        "n_regions": 50,     # e.g., 50 regions
    },
}


def generate_panel_data(
    n_obs: int = 1_000_000,
    n_time_periods: int = 20,
    fe_cardinality: Literal["low", "medium", "high"] = "medium",
    treatment_effect: float = 2.5,
    include_iv: bool = True,
    seed: int = 42,
) -> pl.DataFrame:
    """
    Generate synthetic panel data for fixed effects regression examples.
    
    Creates a realistic firm-year panel with:
    - Outcome variable (y) with known treatment effect
    - Treatment indicator with staggered adoption
    - Continuous covariates
    - Multiple fixed effects (unit, product, time, region)
    - Optional instrumental variable
    - Observation weights
    
    Parameters
    ----------
    n_obs : int
        Number of observations. Default 1,000,000.
    n_time_periods : int
        Number of time periods. Default 20.
    fe_cardinality : {"low", "medium", "high"}
        Fixed effect cardinality level. Default "medium".
    treatment_effect : float
        True treatment effect coefficient. Default 2.5.
    include_iv : bool
        Whether to include instrumental variable. Default True.
    seed : int
        Random seed for reproducibility. Default 42.
    
    Returns
    -------
    pl.DataFrame
        Synthetic panel data with columns:
        - y: outcome variable
        - treatment: treatment indicator (0/1)
        - treated_post: post-treatment indicator for DiD
        - x1, x2: continuous covariates
        - region: categorical region (R1, R2, ...)
        - customer_id: unit fixed effect (high cardinality)
        - product_id: product fixed effect (high cardinality)
        - time_id: time period fixed effect
        - weight: observation weights
        - instrument: instrumental variable (if include_iv=True)
        - event_time: time relative to treatment (for event studies)
    
    Examples
    --------
    >>> df = generate_panel_data(n_obs=1_000_000, seed=42)
    >>> df.shape
    (1000000, 12)
    
    >>> # High cardinality for benchmarks
    >>> df = generate_panel_data(n_obs=10_000_000, fe_cardinality="high", seed=42)
    """
    np.random.seed(seed)
    
    # Get cardinality config
    config = FE_CARDINALITY_CONFIG[fe_cardinality]
    n_units = config["n_units"]
    n_products = config["n_products"]
    n_regions = config["n_regions"]
    
    # Generate fixed effect IDs
    customer_id = np.random.randint(1, n_units + 1, size=n_obs)
    product_id = np.random.randint(1, n_products + 1, size=n_obs)
    time_id = np.random.randint(1, n_time_periods + 1, size=n_obs)
    region_id = np.random.randint(1, n_regions + 1, size=n_obs)
    region = np.array([f"R{r}" for r in region_id])
    
    # Generate unit-level fixed effects (unobserved heterogeneity)
    unit_fe = np.random.normal(0, 1, n_units + 1)[customer_id]
    product_fe = np.random.normal(0, 0.5, n_products + 1)[product_id]
    time_fe = np.random.normal(0, 0.3, n_time_periods + 1)[time_id]
    
    # Generate covariates
    x1 = np.random.normal(0, 1, n_obs)
    x2 = np.random.normal(0, 1, n_obs)
    
    # Generate staggered treatment adoption
    # Each unit has a random treatment start time (some never treated)
    treatment_start = np.random.choice(
        list(range(5, n_time_periods + 1)) + [999],  # 999 = never treated
        size=n_units + 1,
        p=[0.8 / (n_time_periods - 4)] * (n_time_periods - 4) + [0.2]  # 20% never treated
    )
    unit_treatment_start = treatment_start[customer_id]
    treated_post = (time_id >= unit_treatment_start).astype(float)
    
    # Binary treatment indicator (ever treated)
    treatment = (unit_treatment_start < 999).astype(float)
    
    # Event time (for event studies)
    event_time = np.where(
        unit_treatment_start < 999,
        time_id - unit_treatment_start,
        np.nan
    )
    
    # Generate instrumental variable (correlated with treatment, not with error)
    if include_iv:
        # Instrument is correlated with treatment propensity but not outcome
        instrument = 0.5 * treatment + np.random.normal(0, 0.5, n_obs)
    
    # Generate outcome with known DGP
    # y = treatment_effect * treated_post + 1.5*x1 + 0.8*x2 + unit_fe + product_fe + time_fe + error
    error = np.random.normal(0, 1, n_obs)
    y = (
        treatment_effect * treated_post
        + 1.5 * x1
        + 0.8 * x2
        + unit_fe
        + product_fe
        + time_fe
        + error
    )
    
    # Generate weights (positive, varying)
    weight = np.abs(np.random.normal(1, 0.3, n_obs))
    weight = np.maximum(weight, 0.1)  # Ensure positive
    
    # Build DataFrame
    data = {
        "y": y,
        "treatment": treatment,
        "treated_post": treated_post,
        "x1": x1,
        "x2": x2,
        "region": region,
        "customer_id": customer_id,
        "product_id": product_id,
        "time_id": time_id,
        "weight": weight,
        "event_time": event_time,
    }
    
    if include_iv:
        data["instrument"] = instrument
    
    return pl.DataFrame(data)


def generate_benchmark_data(
    n_obs: int = 5_000_000,
    fe_cardinality: Literal["low", "medium", "high"] = "high",
    seed: int = 42,
    save_parquet: bool = False,
    output_path: str | None = None,
) -> pl.DataFrame:
    """
    Generate large synthetic data for benchmarking.
    
    Wrapper around generate_panel_data optimized for benchmark scenarios.
    
    Parameters
    ----------
    n_obs : int
        Number of observations. Default 5,000,000.
    fe_cardinality : {"low", "medium", "high"}
        Fixed effect cardinality. Default "high" for benchmarks.
    seed : int
        Random seed. Default 42.
    save_parquet : bool
        Whether to save to parquet file. Default False.
    output_path : str, optional
        Path for parquet file. Default "_data/benchmark_{n_obs}.parquet".
    
    Returns
    -------
    pl.DataFrame
        Benchmark dataset.
    """
    df = generate_panel_data(
        n_obs=n_obs,
        fe_cardinality=fe_cardinality,
        seed=seed,
    )
    
    if save_parquet:
        if output_path is None:
            output_path = f"_data/benchmark_{n_obs}.parquet"
        df.write_parquet(output_path)
        print(f"Saved benchmark data to {output_path}")
    
    return df


# Convenience functions for common dataset sizes
def get_tutorial_data(seed: int = 42) -> pl.DataFrame:
    """Get standard 1M observation dataset for tutorials."""
    return generate_panel_data(n_obs=1_000_000, seed=seed)


def get_small_example_data(seed: int = 42) -> pl.DataFrame:
    """Get smaller 100K dataset for quick examples (still demonstrates scale)."""
    return generate_panel_data(n_obs=100_000, fe_cardinality="low", seed=seed)


if __name__ == "__main__":
    # Generate and display sample data
    print("Generating tutorial data (1M observations)...")
    df = generate_panel_data(n_obs=1_000_000, seed=42)
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns}")
    print("\nSample rows:")
    print(df.head(5))
    print("\nData types:")
    print(df.dtypes)
