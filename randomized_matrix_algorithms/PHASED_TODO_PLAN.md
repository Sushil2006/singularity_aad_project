# Phased TODO / Task Plan – Randomized Matrix Algorithms Project

This file lists the **concrete, phased tasks** for implementing, benchmarking, and reporting on the randomized matrix algorithms project. It mirrors the high-level plan in `IMPLEMENTATION_AND_REPORT_PLAN.md` but is focused on actionable items.

Checkboxes are for your manual tracking; they are not automated.

---

## Phase 1 – Repository Bootstrapping & README

- [ ] **P1.1 – Create Python package skeleton**
  - [ ] Create directories: `common/`, `rmm/`, `rsvd/`, `low_rank_gemm/`, `overall/`, `plots/` under `randomized_matrix_algorithms/`.
  - [ ] Add minimal `__init__.py` for each package.

- [ ] **P1.2 – Top-level README**
  - [ ] Write `README.md` at project root (or update existing) to describe:
    - Project overview and goals.
    - High-level description of RMM, RSVD, low-rank GEMM, Strassen, GEMM.
    - Dependencies (`numpy`, `scipy`, `torch`, `matplotlib`, etc.).
    - How to set up a virtualenv/conda env.
  - [ ] Add basic instructions on how to run unit tests and a small demo experiment.

---

## Phase 2 – Common Utilities (Datasets, Metrics, Timing, Logging, Config)

- [ ] **P2.1 – `common/datasets.py`**
  - [ ] Implement generators/loaders for synthetic **matrix families**:
    - [ ] Dense Gaussian matrices.
    - [ ] Synthetic low-rank matrices with controllable rank and spectrum.
    - [ ] Sparse matrices with configurable sparsity (10%, 5%, 1%, 0.1%).
    - [ ] Neural-network-like synthetic matrices (heavy-tailed singular values).
  - [ ] Implement **workload-level datasets**:
    - [ ] Neural-net workloads based on small real models trained/loaded via PyTorch on MNIST or CIFAR-10.
    - [ ] Recommender-style matrices including **MovieLens-100K** (or variant) as a real user–item matrix.
  - [ ] Add full type hints and detailed docstrings (shape/dtype, behavior, errors).
  - [ ] Add robust error handling and logging.

- [ ] **P2.2 – `common/metrics.py`**
  - [ ] Implement functions for:
    - [ ] Relative Frobenius error.
    - [ ] Optional spectral norm error (via simple power iteration).
    - [ ] Top-k overlap metrics for recommender workloads.
    - [ ] Accuracy/loss comparison metrics for NN workloads.
  - [ ] Add type hints, docstrings, and validation of inputs.

- [ ] **P2.3 – `common/timing.py` and `common/logging_utils.py`**
  - [ ] Implement context managers/helpers for precise wall-clock timing.
  - [ ] Provide a clean API to separate **offline** (factorization / setup) vs **online** (actual multiply) times.
  - [ ] Implement structured logging utilities (e.g., JSON logs or standard logging with experiment IDs).

- [ ] **P2.4 – `common/config.py`**
  - [ ] Define dataclasses / typed config objects for experiment parameters:
    - [ ] Matrix sizes and shapes.
    - [ ] Ranks `r`, sampling ratios `s/n`.
    - [ ] Number of trials, seeds.
  - [ ] Provide helpers to serialize configs to/from JSON/YAML if needed.

---

## Phase 3 – Baselines (GEMM and Strassen)

- [ ] **P3.1 – `overall/baselines.py`**
  - [ ] Implement a simple wrapper for baseline GEMM using `A @ B` with timing hooks.
  - [ ] Implement **Strassen’s algorithm from scratch** (recursive, square matrices):
    - [ ] Base-case threshold parameter for dropping back to GEMM.
    - [ ] Input validation, shape checks, and detailed error messages.

- [ ] **P3.2 – Baseline Unit Tests & Micro-benchmarks**
  - [ ] Write tests to confirm GEMM and Strassen output correctness for small matrices.
  - [ ] Add simple timing micro-benchmarks to validate timing harness behavior.

---

## Phase 4 – Randomized Matrix Multiplication (RMM)

- [ ] **P4.1 – `rmm/core.py`**
  - [ ] Implement `rmm_uniform(A, B, s, rng)`:
    - [ ] Validate that `A` and `B` have compatible shapes and numeric dtypes.
    - [ ] Implement uniform sampling with `p_k = 1/n`.
    - [ ] Compute estimator `\tilde C` using sampled outer products.
  - [ ] Implement `rmm_importance(A, B, s, rng)`:
    - [ ] Compute norms `||a_k||`, `||b_k||` and probabilities `p_k ∝ ||a_k|| · ||b_k||`.
    - [ ] Handle zero-norm columns robustly (raise clear error if probabilities are degenerate).
    - [ ] Implement the same Monte Carlo estimator with importance correction.
  - [ ] Add detailed type hints, docstrings, and logging.

- [ ] **P4.2 – RMM Unit Tests**
  - [ ] Small-matrix Monte Carlo sanity checks:
    - [ ] Check empirical unbiasedness of `\tilde C` vs exact `AB`.
    - [ ] Check variance scaling approximately as `1/s`.
  - [ ] Edge-case tests: very small `s`, degenerate columns, etc.

- [ ] **P4.3 – `rmm/experiments.py`**
  - [ ] Implement experiment loops according to `randomized_matrix_multiplication/plan.md`:
    - [ ] Matrix families: Gaussian, low-rank, sparse, NN-like.
    - [ ] Sizes: as in the plan (e.g., `N ∈ {512, 1024, 2048}` etc.).
    - [ ] Sampling ratios `s/n ∈ {0.5%, 1%, 2%, 5%, 10%, 20%}`.
    - [ ] Repeat runs (e.g., 10 trials) and record metrics.
  - [ ] Write all results to CSV/JSON with full config metadata.

- [ ] **P4.4 – `plots/plot_rmm.py`**
  - [ ] Generate all RMM figures from the plan:
    - [ ] Error vs `s` across matrix types, ranks, and sparsities.
    - [ ] Runtime vs `n`, speedup vs `s`.
    - [ ] Samples needed for `<5%` error vs rank and sparsity.

- [ ] **P4.5 – RMM Report Section**
  - [ ] Draft LaTeX section for RMM:
    - [ ] Theory (estimator, unbiasedness, variance, optimal `p_k`).
    - [ ] Complexity.
    - [ ] Experimental results and key observations.
  - [ ] Integrate citations from `citations.txt` (Drineas, Ipsen & Wentworth).

---

## Phase 5 – Randomized SVD (RSVD)

- [ ] **P5.1 – `rsvd/core.py`**
  - [ ] Implement RSVD core:
    - [ ] Random test matrix `Ω`, `Y = AΩ`, optional power iterations.
    - [ ] QR factorization to get `Q`.
    - [ ] Small SVD on `B = Q^T A`.
    - [ ] Return `U_k, S_k, Vt_k`.
  - [ ] Add type hints, docstrings, validation, and logging.

- [ ] **P5.2 – RSVD Unit Tests**
  - [ ] Compare RSVD vs full SVD on small matrices.
  - [ ] Verify error decays with `k`, and improves with oversampling / power iterations.

- [ ] **P5.3 – `rsvd/experiments.py`**
  - [ ] Implement experiments per `randomized_SVD/plan.md`:
    - [ ] Matrix families: Gaussian, synthetic low-rank, sparse, NN-like.
    - [ ] Sweep `k`, oversampling `p`, power iterations `q`.
    - [ ] Log relative errors and runtime vs full SVD.

- [ ] **P5.4 – `plots/plot_rsvd.py`**
  - [ ] Plot error vs rank, error vs oversampling, runtime vs size for RSVD vs full SVD.

- [ ] **P5.5 – RSVD Report Section**
  - [ ] Write LaTeX section for RSVD theory, complexity, and experiments.
  - [ ] Cite Halko, Martinsson, Tropp from `citations.txt`.

---

## Phase 6 – Two-Sided Low-Rank GEMM

- [ ] **P6.1 – `low_rank_gemm/core.py`**
  - [ ] Implement RSVD-based two-sided low-rank GEMM:
    - [ ] Offline RSVD factorizations for `A` and `B`.
    - [ ] Online factorized multiply `U_A Σ_A (V_A^T U_B) Σ_B V_B^T`.
  - [ ] Implement deterministic low-rank GEMM variant using exact truncated SVD or ground-truth factors.
  - [ ] Add type hints, docstrings, and careful shape/error handling.

- [ ] **P6.2 – Low-Rank GEMM Unit Tests**
  - [ ] Test on synthetic exactly low-rank `A, B` (varying ranks and noise levels).
  - [ ] Confirm that error is small when `r` ≥ true rank and that runtime is reduced vs GEMM for moderate `N`.

- [ ] **P6.3 – `low_rank_gemm/experiments.py`**
  - [ ] Implement experiments per `low_rank_approx_matrix_mul/plan.md`:
    - [ ] Matrix families: Gaussian, synthetic low-rank, sparse/structured, NN-like (including real NN weights).
    - [ ] Sweep ranks `r` and log `relErr_F`, runtime, speedup, and optional memory.

- [ ] **P6.4 – `plots/plot_low_rank_gemm.py`**
  - [ ] Generate plots:
    - [ ] Error vs rank `r`.
    - [ ] Speedup vs `r`.
    - [ ] Error–speedup tradeoff curves.
    - [ ] Error vs sparsity and vs intrinsic rank.

- [ ] **P6.5 – Low-Rank GEMM Report Section**
  - [ ] Write LaTeX section covering:
    - [ ] Algorithm derivation, error intuition, and complexity.
    - [ ] Results and interpretation across matrix families.
  - [ ] Cite Metere (Low-Rank GEMM), LoRA (Hu et al.), and Koren–Bell–Volinsky.

---

## Phase 7 – Overall Combined Experiments & Comparisons

- [ ] **P7.1 – `overall/experiments.py`**
  - [ ] Implement unified benchmark harness following `overall_combined/plan.md`:
    - [ ] Neural net workload (real NN model + approximate matmuls).
    - [ ] Recsys workload (MovieLens + synthetic or learned user features).
    - [ ] Generic dense benchmarks.
  - [ ] Integrate all methods: GEMM, Strassen, RMM (uniform + importance), RSVD-based low-rank GEMM, deterministic low-rank GEMM.

- [ ] **P7.2 – `plots/plot_overall.py`**
  - [ ] Generate joint plots and tables:
    - [ ] Error vs runtime scatter plots per workload.
    - [ ] Speedup vs error curves.
    - [ ] “Best config under X% error” tables.
    - [ ] Scaling with matrix size `N` for representative configurations.

- [ ] **P7.3 – Overall Comparative Report Section**
  - [ ] Write LaTeX section synthesizing results:
    - [ ] When each method is best (structure, error budget, offline/online).
    - [ ] How empirical tradeoffs match theoretical expectations.

---

## Phase 8 – Full Report Writing & Polishing

- [ ] **P8.1 – Structure & Algorithms**
  - [ ] Write Introduction and algorithm description sections (using all `plan.md` and the master plan).

- [ ] **P8.2 – Experimental Setup**
  - [ ] Describe environment, datasets (synthetic + real), protocols, and metrics.

- [ ] **P8.3 – Results & Analysis**
  - [ ] Integrate all figures and tables.
  - [ ] Provide detailed interpretation of trends and anomalies.

- [ ] **P8.4 – Conclusion & References**
  - [ ] Write Conclusion and Future Work.
  - [ ] Finalize bibliography using `citations.txt` and any additional sources.

---

## Phase 9 – Documentation & Notes

- [ ] **P9.1 – RESULTS / Notes Markdown Files**
  - [ ] For each submodule (`rmm/`, `rsvd/`, `low_rank_gemm/`, `overall/`, `common/`):
    - [ ] Create or update a `RESULTS.md` or notes file documenting:
      - Architecture and main design choices.
      - Experiments run and key outcomes.
      - Bugs encountered and how they were fixed.

This phased TODO plan should be kept in sync with `IMPLEMENTATION_AND_REPORT_PLAN.md` as the project evolves.
