# Randomized Matrix Algorithms

This submodule implements and evaluates randomized matrix algorithms for approximate matrix multiplication and low-rank approximation, following the project guidelines and the detailed plans in:

- `IMPLEMENTATION_AND_REPORT_PLAN.md`
- `PHASED_TODO_PLAN.md`
- `randomized_matrix_multiplication/plan.md`
- `randomized_SVD/plan.md`
- `low_rank_approx_matrix_mul/plan.md`
- `overall_combined/plan.md`

## Overview

We study and compare the following methods:

- **Baseline GEMM**: standard dense matrix multiplication (NumPy `A @ B`).
- **Strassen’s algorithm**: classical asymptotically faster exact method for square matrices.
- **Randomized Matrix Multiplication (RMM)**: outer-product sampling with uniform and importance sampling.
- **Randomized SVD (RSVD)**: low-rank approximation via random projections.
- **Two-sided low-rank GEMM**: approximate `C = AB` using low-rank factorizations of both `A` and `B`.

All algorithms are implemented **from scratch** at the algorithmic level, using Python 3 with:

- `numpy` for dense arrays and BLAS-backed matrix multiplication as a primitive.
- `scipy` for sparse matrices and small exact SVDs where needed.
- `torch` only for training/loading small real neural-network models and datasets (e.g., MNIST/CIFAR-10) and extracting weight matrices.

## Layout

- `common/`: dataset generators/loaders, metrics, timing utilities, logging helpers, experiment configs.
- `rmm/`: RMM core algorithms and experiment runners.
- `rsvd/`: RSVD core algorithms and experiment runners.
- `low_rank_gemm/`: two-sided low-rank GEMM (RSVD-based and deterministic) and experiments.
- `overall/`: baselines (GEMM, Strassen) and unified comparison experiments.
- `plots/`: scripts to generate all figures for the report.

## Running the project (high-level)

Detailed instructions will be added as the implementation progresses, but the final workflow will include:

1. Installing dependencies (NumPy, SciPy, PyTorch, Matplotlib, etc.).
2. Running unit tests for each module.
3. Running experiment drivers in `rmm/`, `rsvd/`, `low_rank_gemm/`, and `overall/` to generate CSV/JSON results.
4. Running plotting scripts in `plots/` to regenerate all figures for the report.

See `IMPLEMENTATION_AND_REPORT_PLAN.md` and `PHASED_TODO_PLAN.md` for the frozen high-level design and phased TODO list.
