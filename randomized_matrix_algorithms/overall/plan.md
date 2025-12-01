# 1. Methods to Compare

## 1.1 Baselines

### Naive / BLAS GEMM

Your usual A @ B (NumPy / PyTorch / BLAS).

This is the gold standard for accuracy and the baseline for runtime.

### Strassen’s Algorithm

Implement a simple recursive Strassen for square matrices (e.g., up to some threshold).

Only useful for reasonably large, square-ish dense matrices.

Good to show as a classical asymptotic improvement vs GEMM, even if in practice it’s often not better for moderate sizes.

---

## 1.2 Approximate / Structured Methods

### RMM (Randomized Matrix Multiplication)

Outer-product sampling with importance sampling.

Tunable knob: number of samples $s$ (or sampling ratio $s/n$).

### Low-Rank GEMM + RSVD

Use RSVD to get $A \approx U_A \Sigma_A V_A^\top$, $B \approx U_B \Sigma_B V_B^\top$.

Then use two-sided low-rank GEMM: $\tilde{C} = U_A M_2 V_B^\top$.

Tunable knob: rank $r$.

### Low-Rank GEMM (deterministic)

Same GEMM as above, but the factorization is:

Either exact truncated SVD (using a library SVD),

Or the true low-rank factors when you synthetically generate low-rank matrices.

This is your “upper bound” on what low-rank GEMM could do if factorization were perfect and cheap.

Tunable knob: rank $r$ again.

For each method, you have:

- A quality knob (s or r).
- A baseline to compare against (standard GEMM).

---

# 2. Practical Datasets / Workloads

Make the comparison feel “real” by using three-ish very interpretable workloads:

## 2.1 Neural Network Layers

Train or load a small MLP or transformer on MNIST/CIFAR/synthetic.

Extract a couple of big weight matrices:

E.g. $W_1 \in \mathbb{R}^{4096 \times 1024}$, $W_2 \in \mathbb{R}^{1024 \times 4096}$.

Workload:

Multiply those weights by random batches $X$:

$H = W_1 X$, $Y = W_2 H$.

Also evaluate task-level effect (accuracy) when you replace GEMM with approximate methods.

Here, matrices are “neural-net-like”: heavy-tailed, approximately low-rank.

## 2.2 Recommender-Style User–Item Matrix

Construct or load a user–item rating / interaction matrix $M \in \mathbb{R}^{n_u \times n_i}$:

Synthetic: $M = UV^\top$ + small noise, $U \in \mathbb{R}^{n_u \times r_0}$, $V \in \mathbb{R}^{n_i \times r_0}$.

Or real: something like MovieLens (if you feel like it).

Workload:

Score all items for many users: $S = MB$ where $B$ is a matrix of user feature vectors.

Here matrices are tall, potentially sparse, and low-rank-ish.

## 2.3 Generic Dense Matrix Benchmark

Dense Gaussian matrices:

Generate $A, B$ with i.i.d. $\mathcal{N}(0,1)$ entries.

Possibly also a synthetic exactly low-rank pair (to show the best-case low-rank behavior).

Workload is simply $C = AB$.

This gives you a “no structure” baseline and an “ideal low-rank” scenario.

---

# 3. Unified Experimental Protocol

Do this for each workload.

## 3.1 Fix Matrix Sizes

Pick a couple of sizes per workload, e.g.:

- Neural net: layer dims = 1024, 2048, 4096.
- Recsys: number of users/items = 5k, 10k.
- Dense: $N = 512, 1024, 2048$ square.

This lets you also comment on how things scale with N.

## 3.2 Metrics

Use the same metrics across all methods:

### Runtime

Wall-clock time for the multiplication.

For RMM & low-rank GEMM:

separate offline (factorization / sampling setup) and online (actual multiply).

### Speedup

$$\text{speedup} = \frac{T_{\text{baseline GEMM}}}{T_{\text{method}}}.$$

### Accuracy / Error

Relative Frobenius error:

$$\text{relErr}_F = \frac{\|C_{\text{full}} - \tilde{C}\|_F}{\|C_{\text{full}}\|_F}.$$

For recsys: top-k ranking overlap (e.g. same top-10 items).

For neural net: accuracy / loss change vs baseline on a validation set.

### Memory / parameter count (optional but nice)

Number of stored parameters / floats vs baseline.

## 3.3 Sweeping Parameters

For each method:

### RMM

Vary sampling ratio $s/n$ (or just absolute $s$):

e.g. $s/n \in \{0.5\%, 1\%, 2\%, 5\%, 10\%, 20\%\}$.

Measure error and runtime at each point.

### Low-Rank GEMM (RSVD)

Vary rank $r$:

e.g. $r \in \{16, 32, 64, 128, 256\}$.

For each $r$, do RSVD + factorized multiply.

### Low-Rank GEMM (deterministic)

Same rank grid $r$.

Use deterministic SVD or ground-truth factors; skip RSVD cost.

This isolates:

“How good can low-rank GEMM be if the factors are perfect?”

### Strassen

Vary recursion threshold (when to drop back to naive GEMM).

Mostly report runtime vs N; error is zero (exact algorithm).

---

# 4. Joint Plots & Tables

This is where everything comes together nicely.

## 4.1 Error vs Runtime Scatter (per workload)

For a fixed workload and matrix size:

Each point = (runtime, error) for one configuration of one algorithm.

Color = algorithm (RMM, LR-GEMM-RSVD, LR-GEMM-det, Strassen).

Label a few representative points.

What you see:

RMM points sweeping from “very fast, high error” (tiny s) to “slower, low error” (larger s).

Low-Rank GEMM points sweeping similarly as rank r grows.

Strassen sitting near GEMM in error (0) but with (maybe slightly lower) runtime at larger sizes.

GEMM itself: single point at (baseline time, 0 error).

This makes the tradeoff curve visually obvious.

## 4.2 Speedup vs Error Curves

For each method, plot:

x-axis: error (log-scale is nice).

y-axis: speedup over GEMM.

You then get:

A frontier curve per method:

RMM: good for very sparse/structured workloads.

Low-rank GEMM: especially good when matrices are truly low-rank.

Strassen: single point, exact but speedup only at large N.

You can annotate key points like:

“Configuration A: 10× speedup, 3% error” (NN weights, LR-GEMM with r = 64).

“Configuration B: 4× speedup, 1% error” (recsys matrix, RMM with s/n = 2%).

## 4.3 “Best Config under X% Error” Table

Make a simple table per workload:

For an error budget of, say, 1%, 5%, 10%:

List, for each method:

Required parameter (s or r),

Runtime,

Speedup.

Example row:

| Workload | Error ≤ 5% | Method | Param | Speedup | Notes |
|----------|------------|--------|-------|---------|-------|
| NN layer | ✓ | LR-GEMM-RSVD | r = 64 | 8× | Accuracy drop < 0.5% |
| NN layer | ✓ | RMM | s/n=5% | 3× | Slightly higher activation noise |
| Recsys | ✓ | LR-GEMM-det | r = 32 | 12× | Top-10 list unchanged 95% of users |
| Gaussian N | ✓ | LR-GEMM-RSVD | r = 256 | 1.5× | Needs high rank for low error |

This is incredibly readable in your report and in the talk.

## 4.4 Scaling with Matrix Size

For at least one workload (say NN layer):

Fix method & parameter (e.g. LR-GEMM-RSVD with r = 64).

Plot:

Runtime vs N for GEMM vs LR-GEMM vs RMM vs Strassen.

Speedup vs N.

This visually shows:

Strassen only becoming interesting at large N.

Randomized methods’ speedup growing with size (since N/r factor expands).

---

# 5. Putting It in Words (Easy Interpretation)

In the report / slides, your big-picture story can be:

## Exact methods (GEMM, Strassen)

Zero error, but cost $O(N^3)$.

Strassen gives theoretical improvement but helps mostly at huge sizes and only for dense, square matrices.

## RMM (sampling outer products)

Best when:

- Matrices are sparse or have very non-uniform column norms.
- Approximate results are okay.

Speedup comes from:

Evaluating only a small fraction of outer products.

Works nicely in:

- Graph / recsys-style sparse matrices,
- Maybe some NN settings, but error is more “noisy.”

## Low-Rank GEMM (with RSVD)

Best when:

- Matrices are numerically low-rank.
- You can afford a one-time factorization.

Speedup comes from:

Replacing large GEMM with products on rank-r factors.

Works great for:

- Neural-network weight matrices,
- Recommender user–item matrices,
- Kernel / covariance matrices.

### RSVD vs deterministic SVD

Deterministic SVD gives slightly better factors but is much more expensive.

RSVD gives nearly identical downstream GEMM quality at far lower factorization cost.

## Overall

There is no single winner.

The winning algorithm depends on:

- Structure: sparse vs dense, low-rank vs full-rank.
- Error budget: how much approximation is allowed.
- Online vs offline cost: do you reuse matrices many times?

Your comparative section then becomes:

“Given a matrix with these properties, here is the best algorithm and why.”

Which is exactly what your instructor wants: you’ve linked theory (complexity + structure) and practice (actual speedups on realistic workloads).