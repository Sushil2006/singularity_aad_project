"""RSVD experiment runner following randomized_SVD/plan.md.

Experiments implemented:
1) Rank sweep across matrix families (error vs k, runtime vs full SVD).
2) Oversampling / power-iteration sweep (effect of p, q on error and runtime).
3) Size scaling (runtime and speedup vs matrix size for fixed k, p, q).

Outputs are CSVs under ``randomized_matrix_algorithms/results``:
- ``rsvd_rank_sweep.csv``
- ``rsvd_hyperparam_sweep.csv``
- ``rsvd_size_scaling.csv``
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np

from randomized_matrix_algorithms.common.datasets import (
    GaussianMatrixSpec,
    LowRankMatrixSpec,
    SparseMatrixSpec,
    gaussian_matrix,
    low_rank_matrix,
    sparse_matrix,
    sparse_low_rank_matrix,
    nn_like_synthetic,
)
from randomized_matrix_algorithms.common.logging_utils import get_logger
from randomized_matrix_algorithms.common.metrics import relative_frobenius_error
from randomized_matrix_algorithms.common.timing import time_function
from randomized_matrix_algorithms.rsvd.core import rsvd, truncated_svd

logger = get_logger(__name__)

# Default grids from plan.md
RANKS = [16, 32, 64, 128, 256]
OVERSAMPLING_VALUES = [0, 5, 10, 20]
POWER_ITERS = [0, 1, 2]
NUM_TRIALS = 5  # keep modest for quick regeneration; adjust upward when running final paper plots
DEFAULT_N = 1024


def _write_csv(path: Path, rows: List[Dict]) -> None:
    """Write rows to CSV with a header derived from the first row."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Wrote %d rows to %s", len(rows), path)


# ============================================================================
# Matrix generators (aligned with plan.md families)
# ============================================================================

def generate_dense_gaussian(n: int, seed: int) -> np.ndarray:
    return gaussian_matrix(GaussianMatrixSpec(m=n, n=n, seed=seed))


def generate_low_rank(n: int, rank: int, seed: int, sharp: bool = False) -> np.ndarray:
    decay = 2.0 if sharp else 1.0
    noise = 0.0 if sharp else 0.01
    return low_rank_matrix(
        LowRankMatrixSpec(
            m=n,
            n=n,
            r=rank,
            decay_exponent=decay,
            noise_std=noise,
            seed=seed,
        )
    )


def generate_sparse(n: int, density: float, seed: int, low_rank_r: int | None = None) -> np.ndarray:
    if low_rank_r is not None:
        return sparse_low_rank_matrix(n, n, r=low_rank_r, density=density, decay_exponent=2.0, seed=seed)
    return sparse_matrix(SparseMatrixSpec(m=n, n=n, density=density, low_rank_r=None, seed=seed))


def generate_nn_like(n: int, seed: int) -> np.ndarray:
    # Strong decay + smaller intrinsic rank to ensure structured behavior
    return nn_like_synthetic(n, n, r=n // 32, decay_exponent=2.5, noise_std=0.0, seed=seed)


# ============================================================================
# Core evaluation helpers
# ============================================================================

@dataclass
class BenchmarkResult:
    rel_error: float
    baseline_error: float
    runtime_sec: float
    speedup: float


def evaluate_rsvd(
    a: np.ndarray,
    rank: int,
    oversampling: int,
    power_iter: int,
    seed: int,
) -> Tuple[BenchmarkResult, float]:
    """Run RSVD vs full SVD on a single matrix.

    Returns (RSVD metrics, baseline_runtime_sec).
    """

    # Baseline exact SVD (full) to get optimal rank-k truncation and its error vs A.
    def _baseline():
        return truncated_svd(a, k=rank)

    exact, baseline_timing = time_function(_baseline)
    exact_mat = exact.as_matrix()

    def _approx():
        return rsvd(a, k=rank, p=oversampling, q=power_iter, seed=seed)

    approx, rsvd_timing = time_function(_approx)
    approx_mat = approx.as_matrix()

    # Errors are vs the original matrix to show approximation quality.
    rel_err_rsvd = relative_frobenius_error(a, approx_mat)
    rel_err_baseline = relative_frobenius_error(a, exact_mat)
    speedup = baseline_timing.seconds / rsvd_timing.seconds if rsvd_timing.seconds > 0 else 0.0
    return (
        BenchmarkResult(
            rel_error=rel_err_rsvd,
            baseline_error=rel_err_baseline,
            runtime_sec=rsvd_timing.seconds,
            speedup=speedup,
        ),
        baseline_timing.seconds,
    )


# ============================================================================
# Experiment 1: Rank sweep across matrix families
# ============================================================================

def run_rank_sweep(output_dir: Path, n: int = DEFAULT_N, seed: int = 1234) -> None:
    logger.info("Running RSVD rank sweep (n=%d)", n)

    rng = np.random.default_rng(seed)
    rows: List[Dict] = []

    matrix_configs: List[Tuple[str, Callable[[], np.ndarray]]] = [
        ("Dense Gaussian", lambda: generate_dense_gaussian(n, int(rng.integers(0, 1_000_000)))),
        ("Low-Rank (r=20)", lambda: generate_low_rank(n, 20, int(rng.integers(0, 1_000_000)), sharp=True)),
        ("Sparse Low-Rank (1%)", lambda: generate_sparse(n, 0.01, int(rng.integers(0, 1_000_000)), low_rank_r=20)),
        ("NN-Like", lambda: generate_nn_like(n, int(rng.integers(0, 1_000_000)))),
    ]

    for matrix_type, gen in matrix_configs:
        logger.info("  Matrix type: %s", matrix_type)
        for trial in range(NUM_TRIALS):
            a = gen()
            for rank in RANKS:
                if rank >= min(a.shape):
                    continue  # skip invalid ranks for smaller sizes
                metrics, baseline_time = evaluate_rsvd(
                    a,
                    rank=rank,
                    oversampling=10,
                    power_iter=1,
                    seed=int(rng.integers(0, 1_000_000)),
                )

                rows.append(
                    {
                        "matrix_type": matrix_type,
                        "algo": "rsvd",
                        "m": a.shape[0],
                        "n": a.shape[1],
                        "rank": rank,
                        "oversampling": 10,
                        "power_iter": 1,
                        "trial": trial,
                        "rel_error": metrics.rel_error,
                        "baseline_error": metrics.baseline_error,
                        "runtime_sec": metrics.runtime_sec,
                        "baseline_runtime_sec": baseline_time,
                        "speedup": metrics.speedup,
                    }
                )
    _write_csv(output_dir / "rsvd_rank_sweep.csv", rows)


# ============================================================================
# Experiment 2: Oversampling / power-iteration sweep
# ============================================================================

def run_hyperparam_sweep(output_dir: Path, n: int = DEFAULT_N, rank: int = 64, seed: int = 5678) -> None:
    logger.info("Running RSVD hyperparameter sweep (p, q)")
    rng = np.random.default_rng(seed)
    rows: List[Dict] = []

    # Use two contrasting matrix types to show effect of q (slow vs fast decay)
    matrix_configs: List[Tuple[str, Callable[[], np.ndarray]]] = [
        ("Dense Gaussian", lambda: generate_dense_gaussian(n, int(rng.integers(0, 1_000_000)))),
        ("Low-Rank (r=rank)", lambda: generate_low_rank(n, rank, int(rng.integers(0, 1_000_000)), sharp=False)),
        ("NN-Like", lambda: generate_nn_like(n, int(rng.integers(0, 1_000_000)))),
    ]

    for matrix_type, gen in matrix_configs:
        logger.info("  Matrix type: %s", matrix_type)
        for trial in range(NUM_TRIALS):
            a = gen()
            for p in OVERSAMPLING_VALUES:
                for q_val in POWER_ITERS:
                    metrics, baseline_time = evaluate_rsvd(
                        a,
                        rank=rank,
                        oversampling=p,
                        power_iter=q_val,
                        seed=int(rng.integers(0, 1_000_000)),
                    )
                    rows.append(
                        {
                            "matrix_type": matrix_type,
                            "m": a.shape[0],
                            "n": a.shape[1],
                            "rank": rank,
                            "oversampling": p,
                            "power_iter": q_val,
                            "trial": trial,
                            "rel_error": metrics.rel_error,
                            "baseline_error": metrics.baseline_error,
                            "runtime_sec": metrics.runtime_sec,
                            "baseline_runtime_sec": baseline_time,
                            "speedup": metrics.speedup,
                        }
                    )
    _write_csv(output_dir / "rsvd_hyperparam_sweep.csv", rows)


# ============================================================================
# Experiment 3: Size scaling
# ============================================================================

def run_size_scaling(
    output_dir: Path,
    sizes: Sequence[int] = (256, 512, 1024, 2048),
    rank: int = 64,
    oversampling: int = 10,
    power_iter: int = 1,
    seed: int = 9012,
) -> None:
    logger.info("Running RSVD size scaling")
    rng = np.random.default_rng(seed)
    rows: List[Dict] = []

    for n in sizes:
        logger.info("  Size n=%d", n)
        for trial in range(NUM_TRIALS):
            a = generate_dense_gaussian(n, int(rng.integers(0, 1_000_000)))
            # Adjust rank if too large for the matrix
            rank_eff = min(rank, n - 1)

            metrics, baseline_time = evaluate_rsvd(
                a,
                rank=rank_eff,
                oversampling=oversampling,
                power_iter=power_iter,
                seed=int(rng.integers(0, 1_000_000)),
            )

            rows.append(
                {
                    "matrix_type": "Dense Gaussian",
                    "m": n,
                    "n": n,
                    "rank": rank_eff,
                    "oversampling": oversampling,
                    "power_iter": power_iter,
                    "trial": trial,
                    "rel_error": metrics.rel_error,
                    "baseline_error": metrics.baseline_error,
                    "runtime_sec": metrics.runtime_sec,
                    "baseline_runtime_sec": baseline_time,
                    "speedup": metrics.speedup,
                }
            )
    _write_csv(output_dir / "rsvd_size_scaling.csv", rows)


# ============================================================================
# Entry point
# ============================================================================

def run_all(output_dir: Path | None = None) -> None:
    """Run all RSVD experiments and emit CSVs."""

    base_dir = Path("randomized_matrix_algorithms/results") if output_dir is None else Path(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    run_rank_sweep(base_dir)
    run_hyperparam_sweep(base_dir)
    run_size_scaling(base_dir)


if __name__ == "__main__":  # pragma: no cover - manual execution
    run_all()
