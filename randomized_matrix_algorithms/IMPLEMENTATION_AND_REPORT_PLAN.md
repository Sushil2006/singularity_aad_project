# Randomized Matrix Algorithms Project: Implementation & Report Master Plan

This document freezes the **end-to-end plan** for the randomized matrix algorithms project and combines:

- The **course / project guidelines** (from-scratch implementation, report format, etc.).
- The **algorithm- and experiment-specific plans** from the four `plan.md` files:
  - Randomized Matrix Multiplication (RMM)
  - Randomized SVD (RSVD)
  - Two-Sided Low-Rank GEMM
  - Overall combined comparisons.

This file is the *single source of truth* for how to structure the repo, implement algorithms, run experiments, and write the research-style report.

---

## 1. Global Course Guidelines (Must-Follow)

### 1.1 Code Implementation Rules

- **From-Scratch Mandate**
  - Implement the **core logic of all algorithms yourself**.
  - Allowed:
    - Standard language features and data structures: lists, arrays, dictionaries, sets, basic math and linear algebra primitives.
    - Low-level numerical kernels (e.g., basic matrix multiplication, dot products) as long as we **do not call a library that already implements the *full* target algorithm** (RMM, RSVD, low-rank GEMM, Strassen) for us.
  - Not allowed:
    - Calling external functions that *implement the algorithm* directly (e.g., `networkx.dijkstra`, `scipy.spatial.ConvexHull`, or a library "randomized SVD" / "low-rank GEMM" routine).
    - For SVD: acceptable to use a **basic dense SVD routine** for *small helper problems* (e.g., exact SVD of the reduced matrix in RSVD), because the randomized part is implemented from scratch and the course guidelines target **algorithmic-level reuse**, not standard linear-algebra subroutines.

- **Programming Language**
  - Primary language: **Python 3.x**.
  - Implementation from scratch using:
    - `numpy` for arrays and basic dense matrix operations (BLAS-backed matmul allowed as a low-level primitive).
    - `scipy` for sparse matrix representations and (when needed) exact SVD on small reduced matrices.
    - `torch` (PyTorch) **only** for training/loading small real neural-network models and extracting their weight matrices; never for implementing RMM/RSVD/low-rank GEMM logic.

- **Repository Requirements**
  - `README.md`:
    - How to install dependencies.
    - How to run unit tests.
    - How to reproduce experiments for each method.
    - How to regenerate all plots and tables for the report.
  - **Well-commented, modular code**:
    - Each algorithm in a separate file/module.
    - Helper utilities in their own modules.
    - No giant “god files”.
  - **Test / Benchmarking Harness**:
    - Scripts (or a small CLI) to:
      - Run unit tests / sanity checks.
      - Run experiment suites and dump results to CSV / JSON.
      - Generate all figures used in the report.
  - **Docstrings**:
    - Every public function must have a docstring.
    - Each docstring must:
      - Explain the purpose of the function.
      - Specify input argument types and shapes.
      - Specify output types and shapes.
  - **Modularization**:
    - Separate modules for:
      - RMM core algorithm(s).
      - RSVD core algorithm.
      - Two-sided low-rank GEMM.
      - Baseline GEMM and Strassen.
      - Dataset/workload generation.
      - Experiment runners.
      - Plotting.

### 1.2 Project Report Structure

Report to be written in LaTeX (research paper style). Structure:

1. **Title Page & Abstract**
   - Descriptive title.
   - Author(s), affiliation, date.
   - 150–250 word abstract summarizing the goal, methods (RMM, RSVD, low-rank GEMM, Strassen, GEMM), datasets, and key findings.

2. **Introduction**
   - Define the overarching problem:
     - Efficient approximate matrix multiplication and low-rank approximation.
   - Real-world relevance:
     - Neural networks, recommender systems, kernels, large-scale linear algebra.
   - Objectives:
     - Implement from-scratch RMM, RSVD, and low-rank GEMM.
     - Benchmark vs exact GEMM and Strassen on realistic workloads.
     - Characterize when each method is best.

3. **Algorithm Descriptions**
   - For **each algorithm** (RMM, RSVD, two-sided low-rank GEMM, Strassen, baseline GEMM):
     - Theoretical explanation.
     - Pseudocode.
     - Correct time and space complexity.
   - Use plan.md contents as the backbone (Sections 2, 3, 4, and 1 in subplans).

4. **Implementation Details**
   - Language and libraries (Python, NumPy, optional SciPy/PyTorch).
   - Data structures (dense arrays vs sparse, how probabilities and indices are stored).
   - Design choices:
     - Error handling strategy, logging, configuration.
     - Offline vs online timing for RMM and low-rank GEMM.
   - Challenges:
     - Numerical stability, reproducibility, memory constraints.

5. **Experimental Setup**
  - Environment:
    - CPU/GPU specs, RAM.
    - OS version, Python version.
    - Key library versions.
  - Datasets:
    - Synthetic matrix families.
    - **Real datasets (mandatory):** at least one real neural-network training dataset (e.g., MNIST or CIFAR-10 via PyTorch) and at least one real recommender dataset (e.g., MovieLens-100K) so that results are not purely synthetic.
  - Unified experimental protocol (from overall_combined/plan.md).

6. **Results & Analysis**
   - Per-method results:
     - RMM: error vs samples, effect of sparsity and rank.
     - RSVD: error vs rank and oversampling, runtime vs full SVD.
     - Low-rank GEMM: error vs rank, speedup vs rank, relation to structure.
   - Overall comparisons:
     - Error vs runtime scatter plots, speedup vs error curves.
     - Best-configuration tables under fixed error budgets.
   - Discuss:
     - How empirical results relate to theory.
     - Where each method is practically useful.

7. **Conclusion**
   - Summarize main findings.
   - Limitations (e.g., only dense CPU implementation, limited sizes).
   - Future work (GPU implementation, more real datasets, other randomized methods).

8. **References**
   - Use `randomized_matrix_algorithms/citations.txt` as the core reference list.
   - Include standard randomized linear algebra literature.

---

## 2. Codebase Architecture (Frozen)

We organize the randomized matrix algorithms as a Python package under `randomized_matrix_algorithms/`:

### 2.1 Top-Level Layout

- `randomized_matrix_algorithms/`
  - `citations.txt`
  - `IMPLEMENTATION_AND_REPORT_PLAN.md` (this file)
  - `randomized_matrix_multiplication/plan.md`
  - `randomized_SVD/plan.md`
  - `low_rank_approx_matrix_mul/plan.md`
  - `overall_combined/plan.md`
  - `rmm/` – RMM implementation and experiments.
  - `rsvd/` – RSVD implementation and experiments.
  - `low_rank_gemm/` – two-sided low-rank GEMM implementation and experiments.
  - `overall/` – combined baselines and cross-method comparisons.
  - `common/` – shared utilities (datasets, metrics, timing, logging, config).
  - `plots/` – scripts to generate all figures.
  - (Optional later) `RESULTS.md` per submodule documenting experiments.

### 2.2 Common Utilities (`common/`)

- `datasets.py`
  - Synthetic matrix families:
    - Dense Gaussian.
    - Synthetic low-rank (orthonormal U, V, decaying singulars, optional noise).
    - Sparse matrices with configurable sparsity.
    - Neural-network-like matrices (heavy-tailed singular values; real or synthetic).
  - Workload-level generators:
    - NN layer workload (W1, W2, X, etc.).
    - Recommender-style user–item workload (M, B).

- `metrics.py`
  - Relative Frobenius error.
  - Optionally spectral norm error (with a simple power method).
  - Task-specific metrics:
    - Top-k overlap for recsys.
    - Accuracy / loss deltas for NN workloads.

- `timing.py`
  - Context managers / helpers for wall-clock timing.
  - Distinguish offline vs online phases.

- `logging_utils.py`
  - Structured logging of experiment configs and metrics.

- `config.py`
  - Dataclasses / typed configs for experiment parameters:
    - Matrix sizes, ranks, sampling ratios, #trials, seeds.

### 2.3 Randomized Matrix Multiplication (`rmm/`)

- `core.py`
  - From-scratch implementations:
    - `rmm_uniform(A, B, s, rng)`.
    - `rmm_importance(A, B, s, rng)`.
  - Responsibilities:
    - Validate dimensions and types.
    - Compute probabilities (`p_k`) and sample indices.
    - Form the Monte Carlo estimator:
      - `\tilde C = (1/s) * sum_i (1/p_{k_i}) a_{k_i} b_{k_i}^T`.
    - Return the approximate product and possibly metadata (indices, probabilities).

- `experiments.py`
  - Implements `randomized_matrix_multiplication/plan.md`:
    - Runs RMM on all matrix families and sizes.
    - Sweeps `s/n` ∈ {0.5%, 1%, 2%, 5%, 10%, 20%}.
    - Repeats T times per configuration.
    - Saves metrics: `relErr_F`, runtime, speedup, etc. to CSV.

### 2.4 Randomized SVD (`rsvd/`)

- `core.py`
  - From-scratch RSVD core (random projection + QR + small SVD):
    - `rsvd(A, k, p=10, q=0, rng)`.
    - Returns (`U_k`, `S_k`, `Vt_k`).
    - The randomized part (Ω, power iterations, QR) is implemented manually.
    - The final small SVD can use a standard dense SVD routine.

- `experiments.py`
  - Implements `randomized_SVD/plan.md`:
    - Compare RSVD vs full SVD.
    - Sweep `k`, `p`, `q`.
    - Evaluate error and runtime on the shared matrix families.

### 2.5 Two-Sided Low-Rank GEMM (`low_rank_gemm/`)

- `core.py`
  - `low_rank_gemm_rsvd(A, B, r, p=10, q=0, rng)`:
    - Offline:
      - RSVD(A) → `(U_A, Σ_A, V_A^T)`.
      - RSVD(B) → `(U_B, Σ_B, V_B^T)`.
    - Online:
      - Compute `M_1 = V_A^T U_B`, `M_2 = Σ_A M_1 Σ_B`, and `\tilde C = U_A M_2 V_B^T`.
  - `low_rank_gemm_det(A, B, r)`:
    - Uses exact truncated SVD or known synthetic factors to form rank-r approximations.

- `experiments.py`
  - Implements `low_rank_approx_matrix_mul/plan.md`:
    - Evaluate error vs rank, speedup vs rank across all matrix families.
    - Separate offline vs online costs.

### 2.6 Overall Combined Comparisons (`overall/`)

- `baselines.py`
  - Exact GEMM via `A @ B`.
  - Strassen’s algorithm for square matrices (from scratch, recursive).

- `experiments.py`
  - Implements `overall_combined/plan.md`:
    - Unified protocol for NN workload, recsys workload, generic dense.
    - For each method (GEMM, Strassen, RMM, low-rank GEMM RSVD/det):
      - Sweep quality knobs (s, r, recursion threshold).
      - Collect runtime, speedup, error, and domain metrics.

### 2.7 Plotting (`plots/`)

- `plot_rmm.py`: RMM-specific figures.
- `plot_rsvd.py`: RSVD figures.
- `plot_low_rank_gemm.py`: Two-sided GEMM figures.
- `plot_overall.py`: joint comparisons (error–runtime scatter, speedup–error, scaling with N, etc.).

Each plotting script reads CSV/JSON results produced by `experiments.py` modules and saves `.png`/`.pdf` figures for inclusion in the LaTeX report.

---

## 3. Dataset and Workload Plan (Frozen)

### 3.1 Core Matrix Families

Used for RMM, RSVD, and low-rank GEMM experiments.

- **Dense Gaussian matrices**
  - `A, B` with i.i.d. `N(0, 1)` entries.
  - Sizes:
    - Square: `N ∈ {512, 1024, 2048}`.
    - Rectangular when needed to mimic NN layers.

- **Synthetic low-rank matrices**
  - Construct `A = U_r Σ_r V_r^T` (and similarly for `B`) with:
    - Orthonormal `U_r, V_r`.
    - Diagonal `Σ_r` with controlled decay (`σ_i ∼ i^{-α}` or exponential decay).
    - Optional additive Gaussian noise.
  - Effective ranks `r_true ∈ {5, 10, 20, 50, 100}`.

- **Sparse matrices**
  - Generate dense matrices then sparsify or use sparse formats.
  - Sparsity levels: {10%, 5%, 1%, 0.1%} nonzeros.
  - Maintain some low-rank structure when desired.

- **Neural-network-like matrices**
  - Option 1 (synthetic): heavy-tailed singular spectrum (e.g., `σ_i ∼ i^{-α}`) to mimic NN weights.
  - Option 2 (real, **required in final experiments**): train or load a small real model (e.g., MLP/CNN on MNIST or CIFAR-10 using PyTorch) and extract actual fully-connected/embedding weight matrices.

### 3.2 Workloads for Overall Comparison

Per `overall_combined/plan.md`:

- **Neural Network Layers**
  - Use `W_1 ∈ R^{4096×1024}`, `W_2 ∈ R^{1024×4096}` or similar.
  - Workload: `H = W_1 X`, `Y = W_2 H` for random or real mini-batches `X`.
  - Metrics:
    - Matrix-level: `relErr_F` for `W_1 X`, `W_2 H`.
    - Task-level: accuracy / loss drop vs baseline.

- **Recommender-style User–Item Matrix**
  - Synthetic: `M = U V^T + noise` with low-rank `U, V`.
  - Real: **MovieLens-100K (or larger variant)** to supply an actual user–item rating matrix.
  - Workload: `S = M B` for some user/item feature matrix `B`.
  - Metrics:
    - `relErr_F(S)`.
    - Top-k ranking overlap per user.

- **Generic Dense Benchmark**
  - Dense Gaussian `A, B` and optionally exactly low-rank `A, B`.
  - Workload: `C = AB`.
  - Metrics: runtime, `relErr_F`, scaling with N.

---

## 4. Experimental Protocol (Unified)

- **Matrix sizes**: for each workload, use sizes suggested in `overall_combined/plan.md`.
- **Metrics**:
  - Runtime (offline vs online separated when applicable).
  - Speedup = `T_baseline_GEMM / T_method`.
  - Accuracy / error: `relErr_F`, and when relevant spectral error.
  - Task-specific metrics (top-k overlap, accuracy drop).
- **Sweeping parameters**:
  - RMM: sampling ratio `s/n` in `{0.5%, 1%, 2%, 5%, 10%, 20%}`.
  - Low-rank GEMM: rank `r ∈ {16, 32, 64, 128, 256}`.
  - RSVD: various `k`, oversampling `p`, power iterations `q`.
  - Strassen: recursion threshold.
- **Visualization**:
  - Error vs s / r.
  - Speedup vs s / r.
  - Error vs runtime scatter per workload.
  - Speedup vs error curves.
  - Scaling with matrix size N.

---

## 5. Implementation & Report Task Breakdown (High Level)

This section is mirrored as a machine-readable task list elsewhere (e.g., TODOs in the IDE). High-level phases:

1. **Repository bootstrapping & README**
2. **Common utilities (datasets, metrics, timing, logging, config)**
3. **Baselines (GEMM, Strassen)**
4. **RMM implementation + experiments + plots + writeup**
5. **RSVD implementation + experiments + plots + writeup**
6. **Two-sided low-rank GEMM implementation + experiments + plots + writeup**
7. **Overall combined experiments + plots + integrative writeup**
8. **Final report polishing & references**

Detailed tasklists will be maintained via the IDE TODO system and aligned with this document.
