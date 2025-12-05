"""
Benchmark runner for leanfe documentation.

Compares leanfe (Polars + DuckDB) against PyFixest for Python.
Generates reproducible benchmark data and measures time/memory.
"""

import time
import tracemalloc
import numpy as np
import polars as pl
from pathlib import Path
from typing import Literal
import json

# Import leanfe
from leanfe import leanfe


# Cardinality configurations
FE_CARDINALITY = {
    "low": {"n_fe1": 100, "n_fe2": 50},
    "medium": {"n_fe1": 1000, "n_fe2": 500},
    "high": {"n_fe1": 10000, "n_fe2": 5000},
}


def generate_benchmark_data(
    n_obs: int,
    fe_cardinality: Literal["low", "medium", "high"] = "medium",
    seed: int = 42,
) -> pl.DataFrame:
    """Generate synthetic benchmark data."""
    np.random.seed(seed)
    
    config = FE_CARDINALITY[fe_cardinality]
    n_fe1 = config["n_fe1"]
    n_fe2 = config["n_fe2"]
    
    fe1 = np.random.randint(1, n_fe1 + 1, n_obs)
    fe2 = np.random.randint(1, n_fe2 + 1, n_obs)
    
    # Generate fixed effects
    fe1_effects = np.random.normal(0, 1, n_fe1 + 1)[fe1]
    fe2_effects = np.random.normal(0, 0.5, n_fe2 + 1)[fe2]
    
    treatment = np.random.binomial(1, 0.3, n_obs).astype(float)
    x1 = np.random.normal(0, 1, n_obs)
    
    # True coefficients: treatment=2.5, x1=1.5
    y = 2.5 * treatment + 1.5 * x1 + fe1_effects + fe2_effects + np.random.normal(0, 1, n_obs)
    
    return pl.DataFrame({
        "y": y,
        "treatment": treatment,
        "x1": x1,
        "fe1": fe1,
        "fe2": fe2,
    })


def run_leanfe_benchmark(
    df: pl.DataFrame,
    backend: str,
    vcov: str = "iid",
    cluster_cols: list = None,
) -> dict:
    """Run leanfe benchmark and measure time/memory."""
    tracemalloc.start()
    start_time = time.time()
    
    result = leanfe(
        formula="y ~ treatment + x1 | fe1 + fe2",
        data=df,
        vcov=vcov,
        cluster_cols=cluster_cols,
        backend=backend,
    )
    
    elapsed = time.time() - start_time
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    return {
        "time_seconds": elapsed,
        "peak_memory_mb": peak / 1024 / 1024,
        "coefficient_treatment": result["coefficients"]["treatment"],
        "coefficient_x1": result["coefficients"]["x1"],
        "n_obs": result["n_obs"],
    }


def run_benchmark_suite(
    n_obs_list: list = [100_000, 500_000, 1_000_000],
    fe_cardinalities: list = ["low", "medium", "high"],
    vcov_types: list = ["iid", "HC1"],
    n_iterations: int = 3,
    seed: int = 42,
) -> list:
    """Run full benchmark suite."""
    results = []
    
    for n_obs in n_obs_list:
        for fe_card in fe_cardinalities:
            print(f"\nGenerating data: {n_obs:,} obs, {fe_card} cardinality...")
            df = generate_benchmark_data(n_obs, fe_card, seed)
            
            for vcov in vcov_types:
                cluster_cols = ["fe1"] if vcov == "cluster" else None
                
                for iteration in range(n_iterations):
                    # Polars backend
                    print(f"  Running Polars ({vcov}, iter {iteration+1})...")
                    polars_result = run_leanfe_benchmark(df, "polars", vcov, cluster_cols)
                    results.append({
                        "package": "leanfe-polars",
                        "n_obs": n_obs,
                        "fe_cardinality": fe_card,
                        "vcov_type": vcov,
                        "iteration": iteration + 1,
                        **polars_result,
                        "seed": seed,
                    })
                    
                    # DuckDB backend
                    print(f"  Running DuckDB ({vcov}, iter {iteration+1})...")
                    duckdb_result = run_leanfe_benchmark(df, "duckdb", vcov, cluster_cols)
                    results.append({
                        "package": "leanfe-duckdb",
                        "n_obs": n_obs,
                        "fe_cardinality": fe_card,
                        "vcov_type": vcov,
                        "iteration": iteration + 1,
                        **duckdb_result,
                        "seed": seed,
                    })
    
    return results


def validate_coefficients(results: list, tolerance: float = 1e-4) -> bool:
    """Validate that all packages produce matching coefficients."""
    # Group by configuration
    configs = {}
    for r in results:
        key = (r["n_obs"], r["fe_cardinality"], r["vcov_type"])
        if key not in configs:
            configs[key] = []
        configs[key].append(r)
    
    all_match = True
    for key, runs in configs.items():
        coefs = [r["coefficient_treatment"] for r in runs]
        if max(coefs) - min(coefs) > tolerance:
            print(f"Coefficient mismatch for {key}: {coefs}")
            all_match = False
    
    return all_match


if __name__ == "__main__":
    print("Running leanfe benchmark suite...")
    
    # Run smaller benchmark for documentation
    results = run_benchmark_suite(
        n_obs_list=[100_000, 500_000],
        fe_cardinalities=["medium"],
        vcov_types=["iid"],
        n_iterations=2,
    )
    
    # Validate coefficients
    if validate_coefficients(results):
        print("\n✓ All coefficients match within tolerance")
    else:
        print("\n✗ Coefficient mismatch detected!")
    
    # Save results
    output_path = Path("_data/benchmark_results.json")
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")
    
    # Print summary
    print("\nBenchmark Summary:")
    for r in results:
        print(f"  {r['package']}: {r['n_obs']:,} obs, {r['time_seconds']:.2f}s, {r['peak_memory_mb']:.1f}MB")
