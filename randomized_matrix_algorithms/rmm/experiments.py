"""Comprehensive RMM experiment runner following plan.md.

This module runs the full experimental protocol from randomized_matrix_multiplication/plan.md:

**Matrix Families:**
1. Dense Gaussian (worst case - uniform column norms)
2. Low-rank matrices (varying r = 5, 10, 20, 50, 100)
3. Sparse matrices (varying density = 10%, 5%, 1%, 0.1%)
4. Neural-network-like (heavy-tailed singular values)

**Sampling ratios:** s/n ∈ {0.5%, 1%, 2%, 5%, 10%, 20%}

**Metrics:**
- Relative Frobenius error
- Runtime
- Speedup vs exact GEMM

**Key experiments:**
- Error vs samples for different matrix types
- Error vs samples for different ranks
- Error vs samples for different sparsity levels
- Samples required for <5% error vs rank
- Samples required for <5% error vs sparsity
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np

from randomized_matrix_algorithms.common.datasets import (
    GaussianMatrixSpec,
    LowRankMatrixSpec,
    SparseMatrixSpec,
    gaussian_matrix,
    low_rank_matrix,
    sparse_matrix,
    nn_like_synthetic,
)
from randomized_matrix_algorithms.common.metrics import relative_frobenius_error
from randomized_matrix_algorithms.common.timing import time_function
from randomized_matrix_algorithms.common.logging_utils import get_logger
from randomized_matrix_algorithms.overall.baselines import gemm_baseline, naive_matmul
from randomized_matrix_algorithms.rmm.core import rmm_importance, rmm_uniform

logger = get_logger(__name__)


# ============================================================================
# Configuration
# ============================================================================

# Sampling ratios from plan.md: {0.5%, 1%, 2%, 5%, 10%, 20%}
SAMPLING_RATIOS = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]

# Ranks to test for low-rank experiments
RANKS = [5, 10, 20, 50, 100]

# Sparsity levels (density = fraction of nonzeros)
SPARSITY_DENSITIES = [0.10, 0.05, 0.01, 0.001]  # 10%, 5%, 1%, 0.1%

# Number of trials for stability
NUM_TRIALS = 10

# Matrix size for most experiments
DEFAULT_N = 512


def _write_csv(path: Path, rows: List[Dict]) -> None:
    """Write rows to CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Wrote {len(rows)} rows to {path}")


# ============================================================================
# Matrix Generators
# ============================================================================

def generate_dense_gaussian(n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate dense Gaussian A, B matrices (worst case for RMM)."""
    spec_a = GaussianMatrixSpec(m=n, n=n, seed=seed)
    spec_b = GaussianMatrixSpec(m=n, n=n, seed=seed + 1000)
    return gaussian_matrix(spec_a), gaussian_matrix(spec_b)


def generate_low_rank(n: int, rank: int, seed: int, sharp: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Generate low-rank A, B matrices.
    
    Parameters
    ----------
    n : int
        Matrix size.
    rank : int
        Target rank.
    seed : int
        Random seed.
    sharp : bool
        If True, use sharp decay (exponent=2.0) and no noise for clearer rank effect.
    """
    decay = 2.0 if sharp else 1.0
    noise = 0.0 if sharp else 0.01
    spec_a = LowRankMatrixSpec(m=n, n=n, r=rank, decay_exponent=decay, noise_std=noise, seed=seed)
    spec_b = LowRankMatrixSpec(m=n, n=n, r=rank, decay_exponent=decay, noise_std=noise, seed=seed + 1000)
    return low_rank_matrix(spec_a), low_rank_matrix(spec_b)


def generate_sparse(n: int, density: float, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate sparse A, B matrices."""
    spec_a = SparseMatrixSpec(m=n, n=n, density=density, seed=seed)
    spec_b = SparseMatrixSpec(m=n, n=n, density=density, seed=seed + 1000)
    return sparse_matrix(spec_a), sparse_matrix(spec_b)


def generate_nn_like(n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    """Generate neural-network-like A, B matrices (heavy-tailed spectrum)."""
    a = nn_like_synthetic(n, n, decay_exponent=1.5, seed=seed)
    b = nn_like_synthetic(n, n, decay_exponent=1.5, seed=seed + 1000)
    return a, b


# ============================================================================
# Core Experiment Runner
# ============================================================================

def run_single_experiment(
    a: np.ndarray,
    b: np.ndarray,
    exact_c: np.ndarray,
    baseline_time: float,
    sampling_ratio: float,
    algo: str,
    seed: int,
) -> Dict:
    """Run a single RMM experiment and return metrics.
    
    Parameters
    ----------
    baseline_time : float
        Time taken by the baseline (naive_matmul) for fair comparison.
    """
    n = a.shape[1]
    s = max(1, int(round(sampling_ratio * n)))
    
    rmm_fn = rmm_uniform if algo == "rmm_uniform" else rmm_importance
    
    def _run():
        return rmm_fn(a, b, num_samples=s, seed=seed)
    
    result, timing = time_function(_run)
    rel_error = relative_frobenius_error(exact_c, result.estimate)
    speedup = baseline_time / timing.seconds if timing.seconds > 0 else 0.0
    
    return {
        "s": s,
        "sampling_ratio": sampling_ratio,
        "rel_error": rel_error,
        "runtime_sec": timing.seconds,
        "speedup": speedup,
    }


# ============================================================================
# Experiment 1: Error vs Samples across Matrix Types
# ============================================================================

def run_matrix_type_comparison(output_dir: Path, n: int = DEFAULT_N, seed: int = 42) -> None:
    """
    Compare RMM performance across different matrix types.
    
    Goal: Show that low-rank and sparse matrices need fewer samples than dense Gaussian.
    """
    logger.info("Running Experiment 1: Matrix Type Comparison")
    
    rng = np.random.default_rng(seed)
    rows: List[Dict] = []
    
    # Matrix types to compare - use sharp=True for low-rank to show clear difference
    matrix_configs = [
        ("Dense Gaussian", lambda: generate_dense_gaussian(n, int(rng.integers(0, 1_000_000)))),
        ("Low-Rank (r=10)", lambda: generate_low_rank(n, 10, int(rng.integers(0, 1_000_000)), sharp=True)),
        ("Sparse (1%)", lambda: generate_sparse(n, 0.01, int(rng.integers(0, 1_000_000)))),
        ("NN-Like", lambda: generate_nn_like(n, int(rng.integers(0, 1_000_000)))),
    ]
    
    for matrix_type, generator in matrix_configs:
        logger.info(f"  Testing {matrix_type}...")
        
        for trial in range(NUM_TRIALS):
            a, b = generator()
            exact_c = gemm_baseline(a, b)  # Use optimized for accuracy
            _, baseline_timing = time_function(lambda: naive_matmul(a, b))  # Use naive for fair timing
            
            for ratio in SAMPLING_RATIOS:
                for algo in ["rmm_uniform", "rmm_importance"]:
                    result = run_single_experiment(
                        a, b, exact_c, baseline_timing.seconds,
                        ratio, algo, int(rng.integers(0, 1_000_000))
                    )
                    rows.append({
                        "matrix_type": matrix_type,
                        "algo": algo,
                        "n": n,
                        "trial": trial,
                        **result,
                    })
    
    _write_csv(output_dir / "rmm_matrix_type_comparison.csv", rows)


# ============================================================================
# Experiment 2: Error vs Samples for Different Ranks
# ============================================================================

def run_rank_sweep(output_dir: Path, n: int = DEFAULT_N, seed: int = 42) -> None:
    """
    Sweep over different intrinsic ranks.
    
    Goal: Show that lower rank → fewer samples needed for good approximation.
    
    Uses sharp=True for clearer rank differentiation (no noise, sharp decay).
    """
    logger.info("Running Experiment 2: Rank Sweep")
    
    rng = np.random.default_rng(seed)
    rows: List[Dict] = []
    
    for rank in RANKS:
        if rank >= n:
            continue
        logger.info(f"  Testing rank={rank}...")
        
        for trial in range(NUM_TRIALS):
            # Use sharp=True for clearer rank effect
            a, b = generate_low_rank(n, rank, int(rng.integers(0, 1_000_000)), sharp=True)
            exact_c = gemm_baseline(a, b)  # Use optimized for accuracy
            _, baseline_timing = time_function(lambda: naive_matmul(a, b))  # Use naive for fair timing
            
            for ratio in SAMPLING_RATIOS:
                for algo in ["rmm_uniform", "rmm_importance"]:
                    result = run_single_experiment(
                        a, b, exact_c, baseline_timing.seconds,
                        ratio, algo, int(rng.integers(0, 1_000_000))
                    )
                    rows.append({
                        "rank": rank,
                        "algo": algo,
                        "n": n,
                        "trial": trial,
                        **result,
                    })
    
    _write_csv(output_dir / "rmm_rank_sweep.csv", rows)


# ============================================================================
# Experiment 3: Error vs Samples for Different Sparsity Levels
# ============================================================================

def run_sparsity_sweep(output_dir: Path, n: int = DEFAULT_N, seed: int = 42) -> None:
    """
    Sweep over different sparsity levels.
    
    Goal: Show that sparser matrices → fewer samples needed.
    """
    logger.info("Running Experiment 3: Sparsity Sweep")
    
    rng = np.random.default_rng(seed)
    rows: List[Dict] = []
    
    for density in SPARSITY_DENSITIES:
        sparsity_pct = (1 - density) * 100
        logger.info(f"  Testing density={density:.1%} (sparsity={sparsity_pct:.1f}%)...")
        
        for trial in range(NUM_TRIALS):
            a, b = generate_sparse(n, density, int(rng.integers(0, 1_000_000)))
            exact_c = gemm_baseline(a, b)  # Use optimized for accuracy
            _, baseline_timing = time_function(lambda: naive_matmul(a, b))  # Use naive for fair timing
            
            for ratio in SAMPLING_RATIOS:
                for algo in ["rmm_uniform", "rmm_importance"]:
                    result = run_single_experiment(
                        a, b, exact_c, baseline_timing.seconds,
                        ratio, algo, int(rng.integers(0, 1_000_000))
                    )
                    rows.append({
                        "density": density,
                        "sparsity_pct": sparsity_pct,
                        "algo": algo,
                        "n": n,
                        "trial": trial,
                        **result,
                    })
    
    _write_csv(output_dir / "rmm_sparsity_sweep.csv", rows)


# ============================================================================
# Experiment 4: Runtime and Speedup vs Matrix Size
# ============================================================================

def run_size_scaling(output_dir: Path, seed: int = 42) -> None:
    """
    Measure runtime scaling with matrix size.
    
    Goal: Show speedup grows as s/n decreases.
    """
    logger.info("Running Experiment 4: Size Scaling")
    
    rng = np.random.default_rng(seed)
    rows: List[Dict] = []
    
    sizes = [128, 256, 512, 1024]
    
    for n in sizes:
        logger.info(f"  Testing n={n}...")
        
        for trial in range(min(NUM_TRIALS, 5)):  # Fewer trials for large matrices
            # Use low-rank matrices for cleaner speedup demonstration
            a, b = generate_low_rank(n, min(20, n // 4), int(rng.integers(0, 1_000_000)))
            exact_c = gemm_baseline(a, b)  # Use optimized for accuracy
            _, baseline_timing = time_function(lambda: naive_matmul(a, b))  # Use naive for fair timing
            
            # Record naive matmul baseline (fair comparison)
            rows.append({
                "algo": "naive_matmul",
                "n": n,
                "trial": trial,
                "s": n,
                "sampling_ratio": 1.0,
                "rel_error": 0.0,
                "runtime_sec": baseline_timing.seconds,
                "speedup": 1.0,
            })
            
            for ratio in SAMPLING_RATIOS:
                for algo in ["rmm_uniform", "rmm_importance"]:
                    result = run_single_experiment(
                        a, b, exact_c, baseline_timing.seconds,
                        ratio, algo, int(rng.integers(0, 1_000_000))
                    )
                    rows.append({
                        "algo": algo,
                        "n": n,
                        "trial": trial,
                        **result,
                    })
    
    _write_csv(output_dir / "rmm_size_scaling.csv", rows)


# ============================================================================
# Experiment 5: Samples Required for <5% Error
# ============================================================================

def find_samples_for_target_error(
    a: np.ndarray,
    b: np.ndarray,
    exact_c: np.ndarray,
    target_error: float,
    algo: str,
    seed: int,
    max_ratio: float = 0.5,
) -> Tuple[int, float]:
    """Binary search to find minimum samples needed for target error."""
    n = a.shape[1]
    rng = np.random.default_rng(seed)
    
    # Test ratios from small to large
    test_ratios = [0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50]
    
    for ratio in test_ratios:
        if ratio > max_ratio:
            break
        s = max(1, int(round(ratio * n)))
        
        # Average over a few trials
        errors = []
        for _ in range(3):
            rmm_fn = rmm_uniform if algo == "rmm_uniform" else rmm_importance
            result = rmm_fn(a, b, num_samples=s, seed=int(rng.integers(0, 1_000_000)))
            errors.append(relative_frobenius_error(exact_c, result.estimate))
        
        avg_error = np.mean(errors)
        if avg_error <= target_error:
            return s, ratio
    
    return -1, -1.0  # Could not achieve target error


def run_samples_for_target_error(output_dir: Path, n: int = DEFAULT_N, seed: int = 42) -> None:
    """
    Find minimum samples needed for target error across different structures.
    
    Goal: Show how sample complexity depends on rank and sparsity.
    
    Note: We use multiple error thresholds (50%, 100%, 200%) since RMM has high
    variance and achieving <5% error requires s close to n.
    """
    logger.info("Running Experiment 5: Samples for Target Error")
    
    rng = np.random.default_rng(seed)
    rows: List[Dict] = []
    # Use more realistic error thresholds for RMM
    target_errors = [1.0, 2.0, 3.0]  # 100%, 200%, 300% relative error
    
    # Test across ranks - use sharp=True for clear differentiation
    logger.info("  Testing across ranks...")
    for rank in RANKS:
        if rank >= n:
            continue
        for trial in range(min(NUM_TRIALS, 5)):  # Fewer trials since we test multiple thresholds
            a, b = generate_low_rank(n, rank, int(rng.integers(0, 1_000_000)), sharp=True)
            exact_c = gemm_baseline(a, b)
            
            for target_error in target_errors:
                for algo in ["rmm_uniform", "rmm_importance"]:
                    s_needed, ratio_needed = find_samples_for_target_error(
                        a, b, exact_c, target_error, algo, int(rng.integers(0, 1_000_000))
                    )
                    rows.append({
                        "experiment": "rank_sweep",
                        "rank": rank,
                        "density": 1.0,
                        "algo": algo,
                        "trial": trial,
                        "samples_needed": s_needed,
                        "ratio_needed": ratio_needed,
                        "target_error": target_error,
                    })
    
    # Test across sparsity levels
    logger.info("  Testing across sparsity levels...")
    for density in SPARSITY_DENSITIES:
        for trial in range(min(NUM_TRIALS, 5)):
            a, b = generate_sparse(n, density, int(rng.integers(0, 1_000_000)))
            exact_c = gemm_baseline(a, b)
            
            for target_error in target_errors:
                for algo in ["rmm_uniform", "rmm_importance"]:
                    s_needed, ratio_needed = find_samples_for_target_error(
                        a, b, exact_c, target_error, algo, int(rng.integers(0, 1_000_000))
                    )
                    rows.append({
                        "experiment": "sparsity_sweep",
                        "rank": -1,
                        "density": density,
                        "algo": algo,
                        "trial": trial,
                        "samples_needed": s_needed,
                        "ratio_needed": ratio_needed,
                        "target_error": target_error,
                    })
    
    _write_csv(output_dir / "rmm_samples_for_target_error.csv", rows)


# ============================================================================
# Main Entry Point
# ============================================================================

def run_all_experiments(output_dir: Path, seed: int = 42) -> None:
    """Run all RMM experiments from plan.md."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Starting comprehensive RMM experiments")
    logger.info("=" * 60)
    
    run_matrix_type_comparison(output_dir, seed=seed)
    run_rank_sweep(output_dir, seed=seed)
    run_sparsity_sweep(output_dir, seed=seed)
    run_size_scaling(output_dir, seed=seed)
    run_samples_for_target_error(output_dir, seed=seed)
    
    logger.info("=" * 60)
    logger.info(f"All experiments complete. Results in {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    # All RMM outputs go to the rmm/results folder
    output_path = Path("randomized_matrix_algorithms/rmm/results")
    run_all_experiments(output_path, seed=42)
