# Visualization Guide: Data, Scales, Colors, Legends, and Graph Types

This guide codifies how to pick data slices, axis scales, color/marker schemes, legend layout, and plot types so that conclusions from the randomized matrix algorithms project are immediately visible. It is derived from:
- `IMPLEMENTATION_AND_REPORT_PLAN.md`
- `PHASED_TODO_PLAN.md`
- `randomized_matrix_multiplication/plan.md`
- `randomized_SVD/plan.md`
- `low_rank_approx_matrix_mul/plan.md`
- `overall_combined/plan.md`
- `rmm/RMM_PLOT_SPEC.md` and current `plots/plot_rmm.py`

Use this as the single reference when regenerating figures or adding new ones.

---

## 1) Data Selection and Aggregation Rules
- **Matrix families (keep consistent across methods):** Dense Gaussian (worst case), synthetic low-rank (sharp decay, no noise when isolating rank effects), sparse (10%, 5%, 1%, 0.1% density with non-uniform column sparsity), NN-like (real or synthetic heavy-tailed spectra), recsys (MovieLens-style), and NN workloads (real FC/embedding weights).
- **Size grids:** Dense benchmarks use `n ∈ {128, 256, 512, 1024, 2048}` (square); NN/recsys workloads follow `overall_combined/plan.md` (layers 1024–4096; users/items ~5k–10k). Fix `n=512` when isolating sampling-ratio effects for RMM to avoid clutter.
- **Parameter grids:**  
  - RMM: sampling ratios `{0.5%, 1%, 2%, 5%, 10%, 20%}`.  
  - RSVD/low-rank GEMM: ranks `{16, 32, 64, 128, 256}` (adapt if dimensions smaller), oversampling `p≈10`, power iters `q ∈ {0,1,2}`.  
  - Strassen: vary recursion threshold; report only if it beats GEMM for that size.
- **Trials and aggregation:** Run ≥5 trials (10 preferred). Plot means with std-dev error bars. Exclude failed configs (NaNs or ratio_needed ≤ 0) from aggregates.
- **Offline vs online:** Always log offline (factorization/sampling setup) separately from online (matmul) time. Plots that claim runtime/speedup must state whether they include offline cost.
- **Target-error extraction:** For “samples/rank needed for <X% error”, pick the largest target error with non-empty data (as in `plot_rmm.py`), but prefer 5% or 1% thresholds when available for interpretability.

---

## 2) Shared Visual Encoding Conventions
- **Algorithms (global palette):**  
  - GEMM baseline: neutral gray (`#4d4d4d`), marker `^`.  
  - Strassen: teal (`#1b9e77`), marker `v`.  
  - RMM-uniform: deep blue (`#2166ac`), marker `o`.  
  - RMM-importance: red (`#b2182b`), marker `s`.  
  - Low-rank GEMM (RSVD factors): orange (`#f46d43`), marker `D`.  
  - Low-rank GEMM (deterministic factors): gold (`#fdae61`), marker `P`.  
  - RSVD: purple (`#7b3294`), marker `X`.
- **Matrix-family palette (when sweeping structure):** Dense Gaussian=red, NN-like=purple, Low-Rank=green, Sparse=blue; use dark→light for worse→better structure.
- **Rank/sparsity gradients:** Use monotone colormaps (viridis/plasma) where “better structure” maps to darker/bolder tones (low rank / high sparsity).
- **Axis scales:**  
  - Errors: y log-scale when spanning orders of magnitude; x linear for sampling ratio or rank.  
  - Runtime: log-log (size vs ms) with base-2 x ticks for size scaling.  
  - Speedup: linear y; x linear (ratio or rank) unless span >10×, then semi-log x.  
  - Error–runtime scatter: log y for runtime, log or linear x depending on error spread (log if error spans >10×).  
  - Sparsity: report as % sparsity (100·(1−density)) on a linear x axis.
- **Legends:** Order worst→best (e.g., Dense → NN-like → Low-Rank → Sparse) or baseline→approximate (GEMM → Strassen → RMM → LR-GEMM RSVD → LR-GEMM det). Keep legends inside empty corners; avoid covering curves. Use short labels and include parameter in label when filtering (e.g., “RMM (s/n=5%)”).
- **Error bars:** Always plot std-dev bars; capsize=3–4; linewidth≥2 for readability.
- **Annotations:** Call out key frontier points (e.g., “8× speedup @ 3% error, r=64”) to make conclusions explicit.
- **Gridlines and limits:** Light grid (`alpha≈0.3`). Clamp x/y limits to remove dead space (see per-figure specs below).

---

## 3) Figure Templates and Choices

### 3.1 RMM (sampling-based GEMM)
Follow `rmm/RMM_PLOT_SPEC.md` numbers unless new data dictates otherwise.
- **Fig 1: Error vs sampling ratio by matrix type**  
  - Data: `n=512`, matrix types {Dense, NN-like, Low-Rank(r=10), Sparse(1%)}; both algos.  
  - Axes: x linear 0–22 (%), y log 0.5–50.  
  - Colors/markers: per matrix-type palette; separate subplots for uniform vs importance.  
  - Message: structured matrices sit lower than dense; importance improves variance.
- **Fig 2: Error vs sampling ratio by rank**  
  - Data: low-rank matrices, ranks {5,10,20,50,100}, no noise, decay exponent 2.0.  
  - Axes: x linear 0–22 (%), y log 0.5–50. Dark→light = lower→higher rank.  
  - Message: lower rank needs fewer samples.
- **Fig 3: Error vs sampling ratio by sparsity**  
  - Data: densities {10%, 5%, 1%, 0.1%}, non-uniform sparsity.  
  - Axes: x linear 0–22 (%), y log 0.1–100. Dark→light = denser→sparser.  
  - Message: sparser matrices drop error faster, esp. for importance sampling.
- **Fig 4: Runtime vs size**  
  - Data: sizes {128…2048}, fix sampling ratio 5%.  
  - Axes: x log2 scale; y log(ms). Lines: GEMM gray, RMM-U blue, RMM-IS red.  
  - Message: RMM diverges from GEMM as n grows.
- **Fig 5: Speedup vs sampling ratio**  
  - Data: all ratios; average speedup over trials.  
  - Axes: x linear (%), y linear; reference line y=1.  
  - Message: speedup increases as s/n decreases; uniform slightly faster than importance.
- **Fig 6: Samples for <target error vs rank**  
  - Data: from `rmm_samples_for_target_error.csv`, rank sweep entries.  
  - Axes: x linear rank, y linear sampling ratio (%) with error bars.  
  - Message: monotone increase with rank; importance dominates uniform.
- **Fig 7: Samples for <target error vs sparsity**  
  - Data: same CSV, sparsity sweep entries.  
  - Axes: x sparsity (%) descending, y linear sampling ratio (%).  
  - Message: more sparsity → fewer samples.

### 3.2 RSVD (low-rank approximation)
- **Error vs target rank k (per matrix family)**  
  - Data: ranks matching {16…256}; include oversampling `p=10`, power iters `q ∈ {0,1,2}`.  
  - Axes: x linear k, y log relative Frobenius or spectral error.  
  - Colors: k gradient (dark low k), markers differentiate q; annotate “full SVD” as a gray horizontal line.  
  - Message: error decays with k; power iters tighten error for slow spectra.
- **Error vs oversampling p**  
  - Data: fix k (e.g., 64) and q; sweep p {0,5,10,20}.  
  - Axes: x linear p, y log error.  
  - Message: diminishing returns after ~10.
- **Runtime vs size (RSVD vs full SVD)**  
  - Data: sizes {512, 1024, 2048}, fixed k (e.g., 64).  
  - Axes: x log2 size, y log runtime (ms).  
  - Colors: RSVD purple, full SVD gray; include offline vs online if applicable.  
  - Message: RSVD sublinear slope relative to full SVD for fixed k.
- **Accuracy–runtime scatter**  
  - Data: all (k,p,q) configs per family.  
  - Axes: x log error, y log runtime; color by family, marker by q.  
  - Message: Pareto frontier; highlight configs on the knee.

### 3.3 Two-Sided Low-Rank GEMM
- **Error vs rank r (per matrix family)**  
  - Data: ranks {16…256}; include both RSVD-factored and deterministic factors.  
  - Axes: x linear r, y log relErr_F(C).  
  - Colors: RSVD orange, det gold; markers differentiate family if overlaid.  
  - Message: error decays until intrinsic rank; det is upper bound.
- **Speedup vs rank r (online-only and with offline bands)**  
  - Data: same ranks; compute speedup with/without offline cost.  
  - Axes: x linear r, y linear speedup; add y=1 reference.  
  - Message: speedup drops as r grows; offline cost matters for one-off multiplies.
- **Error vs speedup trade-off**  
  - Data: pairs (relErr_F, speedup) for each r.  
  - Axes: x log error, y linear speedup.  
  - Message: choose r on Pareto frontier; annotate best points for 1%, 5%, 10% error.
- **Error vs structure (intrinsic rank or sparsity)**  
  - Data: synthetic spectra with varying decay, sparsity levels.  
  - Axes: x intrinsic rank or sparsity (%), y log error at fixed r.  
  - Message: faster spectral decay / higher sparsity reduces needed r.

### 3.4 Overall Combined (workload-level)
- **Error–runtime scatter per workload**  
  - Data: representative sizes per workload; include GEMM, Strassen, RMM (several s), LR-GEMM RSVD/det (several r).  
  - Axes: x log relErr_F (or task metric delta), y log runtime.  
  - Encoding: color by algorithm palette; marker shape by workload (NN, recsys, dense).  
  - Message: show trade-off frontiers; GEMM/Strassen at zero error anchors.
- **Speedup vs error curves**  
  - Data: same grid, aggregated over trials.  
  - Axes: x log error, y linear speedup, y=1 reference.  
  - Message: method-specific curves; annotate “best under 5% error” points.
- **Best-config table (per workload, per error budget)**  
  - Columns: workload, error budget (1/5/10%), method, param (s or r), speedup, notes (task metric drop).  
  - Message: immediate prescription of which method to pick.
- **Scaling with N**  
  - Data: fix a good config per method (e.g., RMM s/n=5%, LR-GEMM r=64).  
  - Axes: x log2 N, y log runtime; optionally a second panel for speedup vs N.  
  - Message: where Strassen overtakes GEMM; randomized methods’ speedup grows with N.

---

## 4) Legend, Ordering, and Labeling Rules
- Sort legends to reflect conceptual order (baseline→approximate) or structural difficulty (dense→NN→low-rank→sparse).
- Keep labels short; include the key knob inline: “RMM-U (s/n=5%)”, “LR-GEMM (r=64, RSVD)”.
- When multiple panels share axes, use a single shared legend (outside, centered) to reduce clutter.
- Title lines state the causal message: e.g., “Sparser matrices need fewer samples (<5% error)”.

---

## 5) Checklist Before Publishing a Plot
- [ ] Axes scaled per guidance; limits tight; ticks human-readable (percentages as 0–20, not 0–0.2).
- [ ] Colors/markers match the global palettes; legends ordered meaningfully.
- [ ] Means + std-dev shown; number of trials noted in caption if <10.
- [ ] Offline vs online costs clarified in caption or label.
- [ ] Key frontier points annotated; baseline GEMM/Strassen clearly visible.
- [ ] File saved under `randomized_matrix_algorithms/results/figures/` with descriptive name (`figX_*.png`).

Adhering to these rules keeps every figure self-explanatory, highlights the trade-offs (error vs speed, structure vs effort), and makes it easy to compare methods and matrix families without re-reading the report text.
