"""Unified benchmark harness for comparing all matrix multiplication methods.

Following overall_combined/plan.md, this module implements:
1. Multiple workloads: NN-like, Recsys-like, Dense Gaussian, Low-Rank synthetic
2. All methods: NumPy GEMM, Naive GEMM, Strassen, RMM (uniform/importance), LR-GEMM (RSVD/det)
3. Comprehensive parameter sweeps across ranks, sampling ratios, sparsities

Outputs CSVs to ``overall/results/`` for joint comparison plots.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from randomized_matrix_algorithms.common.datasets import (
    gaussian_matrix,
    low_rank_matrix,
    sparse_matrix,
    nn_like_synthetic,
    GaussianMatrixSpec,
    LowRankMatrixSpec,
    SparseMatrixSpec,
)
from randomized_matrix_algorithms.common.logging_utils import get_logger
from randomized_matrix_algorithms.common.metrics import relative_frobenius_error
from randomized_matrix_algorithms.common.timing import time_function
from randomized_matrix_algorithms.overall.baselines import (
    gemm_baseline,
    naive_matmul,
    strassen,
)
from randomized_matrix_algorithms.rmm.core import rmm_uniform, rmm_importance
from randomized_matrix_algorithms.low_rank_gemm.core import (
    low_rank_gemm_rsvd,
    low_rank_gemm_deterministic,
    compute_factors_rsvd,
    factorized_multiply_from_factors,
)

logger = get_logger(__name__)

FloatArray = NDArray[np.floating]

# ============================================================================
# Configuration
# ============================================================================

# Matrix sizes to test
SIZES = [256, 512, 1024]

# Sampling ratios for RMM
SAMPLING_RATIOS = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]

# Ranks for Low-Rank GEMM
RANKS = [8, 16, 32, 64, 128]

# Intrinsic ranks for synthetic low-rank matrices
INTRINSIC_RANKS = [10, 20, 50]

# Sparsity densities
SPARSITY_DENSITIES = [0.10, 0.05, 0.01]

# Number of trials
NUM_TRIALS = 3

# Strassen threshold
STRASSEN_THRESHOLD = 64


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
# Matrix Generators for Different Workloads
# ============================================================================

def generate_nn_like_workload(n: int, seed: int) -> Tuple[FloatArray, FloatArray, str]:
    """Generate NN-like matrices (heavy-tailed spectrum)."""
    a = nn_like_synthetic(n, n, decay_exponent=1.5, seed=seed)
    b = nn_like_synthetic(n, n, decay_exponent=1.5, seed=seed + 1000)
    return a, b, "NN-Like"


def generate_recsys_workload(n: int, intrinsic_rank: int, seed: int) -> Tuple[FloatArray, FloatArray, str]:
    """Generate Recsys-like matrices (low-rank user-item style)."""
    # User-item matrix: low-rank with some noise
    spec_a = LowRankMatrixSpec(m=n, n=n, r=intrinsic_rank, decay_exponent=1.5, noise_std=0.01, seed=seed)
    spec_b = LowRankMatrixSpec(m=n, n=n, r=intrinsic_rank, decay_exponent=1.5, noise_std=0.01, seed=seed + 1000)
    a = low_rank_matrix(spec_a)
    b = low_rank_matrix(spec_b)
    return a, b, f"Recsys (r={intrinsic_rank})"


def generate_dense_gaussian_workload(n: int, seed: int) -> Tuple[FloatArray, FloatArray, str]:
    """Generate dense Gaussian matrices (worst case for approximation)."""
    spec_a = GaussianMatrixSpec(m=n, n=n, seed=seed)
    spec_b = GaussianMatrixSpec(m=n, n=n, seed=seed + 1000)
    a = gaussian_matrix(spec_a)
    b = gaussian_matrix(spec_b)
    return a, b, "Dense Gaussian"


def generate_low_rank_workload(n: int, intrinsic_rank: int, seed: int) -> Tuple[FloatArray, FloatArray, str]:
    """Generate exactly low-rank matrices (best case for LR-GEMM)."""
    spec_a = LowRankMatrixSpec(m=n, n=n, r=intrinsic_rank, decay_exponent=2.0, noise_std=0.0, seed=seed)
    spec_b = LowRankMatrixSpec(m=n, n=n, r=intrinsic_rank, decay_exponent=2.0, noise_std=0.0, seed=seed + 1000)
    a = low_rank_matrix(spec_a)
    b = low_rank_matrix(spec_b)
    return a, b, f"Low-Rank (r={intrinsic_rank})"


def generate_sparse_workload(n: int, density: float, seed: int) -> Tuple[FloatArray, FloatArray, str]:
    """Generate sparse matrices."""
    spec_a = SparseMatrixSpec(m=n, n=n, density=density, seed=seed)
    spec_b = SparseMatrixSpec(m=n, n=n, density=density, seed=seed + 1000)
    a = sparse_matrix(spec_a)
    b = sparse_matrix(spec_b)
    sparsity_pct = (1 - density) * 100
    return a, b, f"Sparse ({sparsity_pct:.0f}%)"


# ============================================================================
# Method Runners
# ============================================================================

@dataclass
class MethodResult:
    """Result from running a single method."""
    method: str
    config: str  # e.g., "r=32" or "s/n=5%"
    runtime_sec: float
    rel_error: float
    speedup_vs_numpy: float
    speedup_vs_naive: float


def run_numpy_gemm(a: FloatArray, b: FloatArray, exact_c: FloatArray) -> MethodResult:
    """Run NumPy's optimized GEMM."""
    def _run():
        return gemm_baseline(a, b)
    
    result, timing = time_function(_run)
    rel_error = relative_frobenius_error(exact_c, result)
    
    return MethodResult(
        method="NumPy GEMM",
        config="optimized",
        runtime_sec=timing.seconds,
        rel_error=rel_error,
        speedup_vs_numpy=1.0,
        speedup_vs_naive=0.0,  # Will be filled later
    )


def run_naive_gemm(a: FloatArray, b: FloatArray, exact_c: FloatArray) -> MethodResult:
    """Run naive outer-product GEMM."""
    def _run():
        return naive_matmul(a, b)
    
    result, timing = time_function(_run)
    rel_error = relative_frobenius_error(exact_c, result)
    
    return MethodResult(
        method="Naive GEMM",
        config="outer-product",
        runtime_sec=timing.seconds,
        rel_error=rel_error,
        speedup_vs_numpy=0.0,
        speedup_vs_naive=1.0,
    )


def run_strassen(a: FloatArray, b: FloatArray, exact_c: FloatArray, threshold: int = 64) -> MethodResult:
    """Run Strassen's algorithm."""
    def _run():
        return strassen(a, b, threshold=threshold)
    
    result, timing = time_function(_run)
    rel_error = relative_frobenius_error(exact_c, result)
    
    return MethodResult(
        method="Strassen",
        config=f"thresh={threshold}",
        runtime_sec=timing.seconds,
        rel_error=rel_error,
        speedup_vs_numpy=0.0,
        speedup_vs_naive=0.0,
    )


def run_rmm_uniform(
    a: FloatArray, b: FloatArray, exact_c: FloatArray, 
    sampling_ratio: float, seed: int
) -> MethodResult:
    """Run RMM with uniform sampling."""
    n = a.shape[1]
    s = max(1, int(round(sampling_ratio * n)))
    
    def _run():
        return rmm_uniform(a, b, num_samples=s, seed=seed)
    
    result, timing = time_function(_run)
    rel_error = relative_frobenius_error(exact_c, result.estimate)
    
    return MethodResult(
        method="RMM-Uniform",
        config=f"s/n={sampling_ratio*100:.1f}%",
        runtime_sec=timing.seconds,
        rel_error=rel_error,
        speedup_vs_numpy=0.0,
        speedup_vs_naive=0.0,
    )


def run_rmm_importance(
    a: FloatArray, b: FloatArray, exact_c: FloatArray, 
    sampling_ratio: float, seed: int
) -> MethodResult:
    """Run RMM with importance sampling."""
    n = a.shape[1]
    s = max(1, int(round(sampling_ratio * n)))
    
    def _run():
        return rmm_importance(a, b, num_samples=s, seed=seed)
    
    result, timing = time_function(_run)
    rel_error = relative_frobenius_error(exact_c, result.estimate)
    
    return MethodResult(
        method="RMM-Importance",
        config=f"s/n={sampling_ratio*100:.1f}%",
        runtime_sec=timing.seconds,
        rel_error=rel_error,
        speedup_vs_numpy=0.0,
        speedup_vs_naive=0.0,
    )


def run_lrgemm_rsvd(
    a: FloatArray, b: FloatArray, exact_c: FloatArray, 
    rank: int, seed: int
) -> MethodResult:
    """Run Low-Rank GEMM with RSVD factorization."""
    if rank >= min(a.shape[0], a.shape[1], b.shape[1]):
        return MethodResult(
            method="LR-GEMM-RSVD",
            config=f"r={rank}",
            runtime_sec=float('inf'),
            rel_error=float('inf'),
            speedup_vs_numpy=0.0,
            speedup_vs_naive=0.0,
        )
    
    def _run():
        return low_rank_gemm_rsvd(a, b, r=rank, p=10, q=1, seed=seed)
    
    result, timing = time_function(_run)
    rel_error = relative_frobenius_error(exact_c, result.estimate)
    
    return MethodResult(
        method="LR-GEMM-RSVD",
        config=f"r={rank}",
        runtime_sec=timing.seconds,
        rel_error=rel_error,
        speedup_vs_numpy=0.0,
        speedup_vs_naive=0.0,
    )


def run_lrgemm_det(
    a: FloatArray, b: FloatArray, exact_c: FloatArray, 
    rank: int
) -> MethodResult:
    """Run Low-Rank GEMM with deterministic SVD."""
    if rank >= min(a.shape[0], a.shape[1], b.shape[1]):
        return MethodResult(
            method="LR-GEMM-Det",
            config=f"r={rank}",
            runtime_sec=float('inf'),
            rel_error=float('inf'),
            speedup_vs_numpy=0.0,
            speedup_vs_naive=0.0,
        )
    
    def _run():
        return low_rank_gemm_deterministic(a, b, r=rank)
    
    result, timing = time_function(_run)
    rel_error = relative_frobenius_error(exact_c, result.estimate)
    
    return MethodResult(
        method="LR-GEMM-Det",
        config=f"r={rank}",
        runtime_sec=timing.seconds,
        rel_error=rel_error,
        speedup_vs_numpy=0.0,
        speedup_vs_naive=0.0,
    )


# ============================================================================
# Main Experiment: Comprehensive Method Comparison
# ============================================================================

def run_comprehensive_comparison(output_dir: Path, seed: int = 42) -> None:
    """Run comprehensive comparison of all methods across all workloads.
    
    This is the main experiment that generates data for joint comparison plots.
    """
    logger.info("=" * 60)
    logger.info("Running Comprehensive Method Comparison")
    logger.info("=" * 60)
    
    rng = np.random.default_rng(seed)
    rows: List[Dict] = []
    
    for n in SIZES:
        logger.info(f"\n--- Matrix Size N={n} ---")
        
        # Define workloads for this size
        workloads = [
            lambda s=n, sd=int(rng.integers(0, 1_000_000)): generate_nn_like_workload(s, sd),
            lambda s=n, sd=int(rng.integers(0, 1_000_000)): generate_dense_gaussian_workload(s, sd),
        ]
        
        # Add low-rank workloads
        for ir in INTRINSIC_RANKS:
            if ir < n:
                workloads.append(
                    lambda s=n, r=ir, sd=int(rng.integers(0, 1_000_000)): generate_low_rank_workload(s, r, sd)
                )
                workloads.append(
                    lambda s=n, r=ir, sd=int(rng.integers(0, 1_000_000)): generate_recsys_workload(s, r, sd)
                )
        
        # Add sparse workloads
        for density in SPARSITY_DENSITIES:
            workloads.append(
                lambda s=n, d=density, sd=int(rng.integers(0, 1_000_000)): generate_sparse_workload(s, d, sd)
            )
        
        for workload_fn in workloads:
            for trial in range(NUM_TRIALS):
                a, b, workload_name = workload_fn()
                exact_c = gemm_baseline(a, b)
                
                logger.info(f"  {workload_name}, trial {trial+1}/{NUM_TRIALS}")
                
                # Run baseline methods
                numpy_result = run_numpy_gemm(a, b, exact_c)
                naive_result = run_naive_gemm(a, b, exact_c)
                
                numpy_time = numpy_result.runtime_sec
                naive_time = naive_result.runtime_sec
                
                # Store baseline results
                for result in [numpy_result, naive_result]:
                    result.speedup_vs_numpy = numpy_time / result.runtime_sec if result.runtime_sec > 0 else 0
                    result.speedup_vs_naive = naive_time / result.runtime_sec if result.runtime_sec > 0 else 0
                    rows.append({
                        "workload": workload_name,
                        "n": n,
                        "trial": trial,
                        "method": result.method,
                        "config": result.config,
                        "runtime_sec": result.runtime_sec,
                        "rel_error": result.rel_error,
                        "speedup_vs_numpy": result.speedup_vs_numpy,
                        "speedup_vs_naive": result.speedup_vs_naive,
                    })
                
                # Run Strassen (only for square matrices)
                try:
                    strassen_result = run_strassen(a, b, exact_c, STRASSEN_THRESHOLD)
                    strassen_result.speedup_vs_numpy = numpy_time / strassen_result.runtime_sec if strassen_result.runtime_sec > 0 else 0
                    strassen_result.speedup_vs_naive = naive_time / strassen_result.runtime_sec if strassen_result.runtime_sec > 0 else 0
                    rows.append({
                        "workload": workload_name,
                        "n": n,
                        "trial": trial,
                        "method": strassen_result.method,
                        "config": strassen_result.config,
                        "runtime_sec": strassen_result.runtime_sec,
                        "rel_error": strassen_result.rel_error,
                        "speedup_vs_numpy": strassen_result.speedup_vs_numpy,
                        "speedup_vs_naive": strassen_result.speedup_vs_naive,
                    })
                except Exception as e:
                    logger.warning(f"Strassen failed: {e}")
                
                # Run RMM methods
                for ratio in SAMPLING_RATIOS:
                    for rmm_fn, method_name in [
                        (run_rmm_uniform, "RMM-Uniform"),
                        (run_rmm_importance, "RMM-Importance"),
                    ]:
                        result = rmm_fn(a, b, exact_c, ratio, int(rng.integers(0, 1_000_000)))
                        result.speedup_vs_numpy = numpy_time / result.runtime_sec if result.runtime_sec > 0 else 0
                        result.speedup_vs_naive = naive_time / result.runtime_sec if result.runtime_sec > 0 else 0
                        rows.append({
                            "workload": workload_name,
                            "n": n,
                            "trial": trial,
                            "method": result.method,
                            "config": result.config,
                            "runtime_sec": result.runtime_sec,
                            "rel_error": result.rel_error,
                            "speedup_vs_numpy": result.speedup_vs_numpy,
                            "speedup_vs_naive": result.speedup_vs_naive,
                        })
                
                # Run LR-GEMM methods
                for rank in RANKS:
                    if rank >= n:
                        continue
                    
                    for lrgemm_fn in [run_lrgemm_rsvd, run_lrgemm_det]:
                        if lrgemm_fn == run_lrgemm_rsvd:
                            result = lrgemm_fn(a, b, exact_c, rank, int(rng.integers(0, 1_000_000)))
                        else:
                            result = lrgemm_fn(a, b, exact_c, rank)
                        
                        if result.runtime_sec < float('inf'):
                            result.speedup_vs_numpy = numpy_time / result.runtime_sec if result.runtime_sec > 0 else 0
                            result.speedup_vs_naive = naive_time / result.runtime_sec if result.runtime_sec > 0 else 0
                            rows.append({
                                "workload": workload_name,
                                "n": n,
                                "trial": trial,
                                "method": result.method,
                                "config": result.config,
                                "runtime_sec": result.runtime_sec,
                                "rel_error": result.rel_error,
                                "speedup_vs_numpy": result.speedup_vs_numpy,
                                "speedup_vs_naive": result.speedup_vs_naive,
                            })
    
    _write_csv(output_dir / "overall_comparison.csv", rows)


# ============================================================================
# Experiment 2: Scaling with Matrix Size
# ============================================================================

def run_scaling_experiment(output_dir: Path, seed: int = 42) -> None:
    """Run scaling experiment to show how methods scale with matrix size."""
    logger.info("=" * 60)
    logger.info("Running Scaling Experiment")
    logger.info("=" * 60)
    
    rng = np.random.default_rng(seed)
    rows: List[Dict] = []
    
    sizes = [128, 256, 512, 1024]
    fixed_rank = 32
    fixed_ratio = 0.05
    
    for n in sizes:
        logger.info(f"  Size N={n}")
        
        for trial in range(NUM_TRIALS):
            # Use NN-like workload as representative
            a, b, workload_name = generate_nn_like_workload(n, int(rng.integers(0, 1_000_000)))
            exact_c = gemm_baseline(a, b)
            
            # Baselines
            numpy_result = run_numpy_gemm(a, b, exact_c)
            naive_result = run_naive_gemm(a, b, exact_c)
            
            numpy_time = numpy_result.runtime_sec
            naive_time = naive_result.runtime_sec
            
            # All methods with fixed configs
            methods_to_run = [
                ("NumPy GEMM", lambda: run_numpy_gemm(a, b, exact_c)),
                ("Naive GEMM", lambda: run_naive_gemm(a, b, exact_c)),
                ("Strassen", lambda: run_strassen(a, b, exact_c, STRASSEN_THRESHOLD)),
                ("RMM-Uniform", lambda: run_rmm_uniform(a, b, exact_c, fixed_ratio, int(rng.integers(0, 1_000_000)))),
                ("RMM-Importance", lambda: run_rmm_importance(a, b, exact_c, fixed_ratio, int(rng.integers(0, 1_000_000)))),
            ]
            
            if fixed_rank < n:
                methods_to_run.append(
                    ("LR-GEMM-RSVD", lambda: run_lrgemm_rsvd(a, b, exact_c, fixed_rank, int(rng.integers(0, 1_000_000))))
                )
                methods_to_run.append(
                    ("LR-GEMM-Det", lambda: run_lrgemm_det(a, b, exact_c, fixed_rank))
                )
            
            for method_name, method_fn in methods_to_run:
                try:
                    result = method_fn()
                    result.speedup_vs_numpy = numpy_time / result.runtime_sec if result.runtime_sec > 0 else 0
                    result.speedup_vs_naive = naive_time / result.runtime_sec if result.runtime_sec > 0 else 0
                    rows.append({
                        "n": n,
                        "trial": trial,
                        "method": result.method,
                        "config": result.config,
                        "runtime_sec": result.runtime_sec,
                        "rel_error": result.rel_error,
                        "speedup_vs_numpy": result.speedup_vs_numpy,
                        "speedup_vs_naive": result.speedup_vs_naive,
                    })
                except Exception as e:
                    logger.warning(f"{method_name} failed at N={n}: {e}")
    
    _write_csv(output_dir / "scaling_comparison.csv", rows)


# ============================================================================
# Experiment 3: Best Config Under Error Budget
# ============================================================================

def run_error_budget_experiment(output_dir: Path, seed: int = 42) -> None:
    """Find best configurations under different error budgets."""
    logger.info("=" * 60)
    logger.info("Running Error Budget Experiment")
    logger.info("=" * 60)
    
    rng = np.random.default_rng(seed)
    rows: List[Dict] = []
    
    n = 512
    error_budgets = [0.01, 0.05, 0.10, 0.20, 0.50]  # 1%, 5%, 10%, 20%, 50%
    
    # Test different workloads
    workload_generators = [
        ("NN-Like", lambda: generate_nn_like_workload(n, int(rng.integers(0, 1_000_000)))),
        ("Dense Gaussian", lambda: generate_dense_gaussian_workload(n, int(rng.integers(0, 1_000_000)))),
        ("Low-Rank (r=20)", lambda: generate_low_rank_workload(n, 20, int(rng.integers(0, 1_000_000)))),
        ("Sparse (95%)", lambda: generate_sparse_workload(n, 0.05, int(rng.integers(0, 1_000_000)))),
    ]
    
    for workload_name, workload_fn in workload_generators:
        logger.info(f"  Workload: {workload_name}")
        
        a, b, _ = workload_fn()
        exact_c = gemm_baseline(a, b)
        
        numpy_result = run_numpy_gemm(a, b, exact_c)
        naive_result = run_naive_gemm(a, b, exact_c)
        numpy_time = numpy_result.runtime_sec
        naive_time = naive_result.runtime_sec
        
        # Collect all method results
        all_results: List[Dict] = []
        
        # RMM methods
        for ratio in SAMPLING_RATIOS:
            for rmm_fn in [run_rmm_uniform, run_rmm_importance]:
                result = rmm_fn(a, b, exact_c, ratio, int(rng.integers(0, 1_000_000)))
                all_results.append({
                    "method": result.method,
                    "config": result.config,
                    "runtime_sec": result.runtime_sec,
                    "rel_error": result.rel_error,
                    "speedup_vs_naive": naive_time / result.runtime_sec if result.runtime_sec > 0 else 0,
                })
        
        # LR-GEMM methods
        for rank in RANKS:
            if rank >= n:
                continue
            for lrgemm_fn in [run_lrgemm_rsvd, run_lrgemm_det]:
                if lrgemm_fn == run_lrgemm_rsvd:
                    result = lrgemm_fn(a, b, exact_c, rank, int(rng.integers(0, 1_000_000)))
                else:
                    result = lrgemm_fn(a, b, exact_c, rank)
                
                if result.runtime_sec < float('inf'):
                    all_results.append({
                        "method": result.method,
                        "config": result.config,
                        "runtime_sec": result.runtime_sec,
                        "rel_error": result.rel_error,
                        "speedup_vs_naive": naive_time / result.runtime_sec if result.runtime_sec > 0 else 0,
                    })
        
        # For each error budget, find best method
        for budget in error_budgets:
            # Filter methods that meet the budget
            valid = [r for r in all_results if r["rel_error"] <= budget]
            
            if valid:
                # Find fastest among valid
                best = min(valid, key=lambda x: x["runtime_sec"])
                rows.append({
                    "workload": workload_name,
                    "n": n,
                    "error_budget": budget,
                    "error_budget_pct": budget * 100,
                    "best_method": best["method"],
                    "best_config": best["config"],
                    "actual_error": best["rel_error"],
                    "runtime_sec": best["runtime_sec"],
                    "speedup_vs_naive": best["speedup_vs_naive"],
                })
            else:
                rows.append({
                    "workload": workload_name,
                    "n": n,
                    "error_budget": budget,
                    "error_budget_pct": budget * 100,
                    "best_method": "None",
                    "best_config": "N/A",
                    "actual_error": float('inf'),
                    "runtime_sec": float('inf'),
                    "speedup_vs_naive": 0.0,
                })
    
    _write_csv(output_dir / "error_budget_comparison.csv", rows)


# ============================================================================
# Main Entry Point
# ============================================================================

def run_all_experiments(output_dir: Path, seed: int = 42) -> None:
    """Run all overall comparison experiments."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Starting Overall Comparison Experiments")
    logger.info("=" * 60)
    
    run_comprehensive_comparison(output_dir, seed=seed)
    run_scaling_experiment(output_dir, seed=seed)
    run_error_budget_experiment(output_dir, seed=seed)
    
    logger.info("=" * 60)
    logger.info(f"All experiments complete. Results in {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    output_path = Path("randomized_matrix_algorithms/overall/results")
    run_all_experiments(output_path, seed=42)
