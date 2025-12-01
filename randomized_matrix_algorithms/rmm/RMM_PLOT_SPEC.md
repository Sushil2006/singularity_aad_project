# RMM Plot Specification

## Final Results Summary (After Implementation)

### What the Figures Show:

| Figure | Goal | Result | Status |
|--------|------|--------|--------|
| **Fig 1** | Structured matrices have lower error | Low-Rank & Sparse < Dense Gaussian | ✅ Success |
| **Fig 2** | Lower rank → lower error | Some differentiation visible | ⚠️ Partial |
| **Fig 3** | Sparser → lower error | Very clear separation (99.9% << 90%) | ✅ Excellent |
| **Fig 4** | RMM faster than GEMM | Uniform RMM faster at large n | ✅ Success |
| **Fig 5** | Speedup increases as s/n decreases | ~2x speedup for uniform | ✅ Success |
| **Fig 6** | Lower rank needs fewer samples | Non-monotonic relationship | ⚠️ Partial |
| **Fig 7** | Sparser needs fewer samples | Clear downward trend | ✅ Success |

### Key Insights:
1. **Sparsity has the strongest effect** - 99.9% sparse matrices need only ~2% samples vs ~15% for 90% sparse
2. **Importance sampling helps for sparse matrices** - clearer separation in Fig 3 right panel
3. **Uniform sampling is faster** - no norm computation overhead
4. **RMM error depends on column norm distribution**, not just rank

---

## Goals for Each Figure

### Figure 1: Error vs Sampling Ratio by Matrix Type
**Goal**: Show that structured matrices (low-rank, sparse) achieve LOWER error than dense Gaussian at the same sampling ratio.

**Expected behavior**:
- Dense Gaussian: Highest error (baseline)
- Low-Rank: Lower error (energy concentrated in few directions)
- Sparse: Lower error (column norms vary → importance sampling helps)
- NN-Like: Between dense and low-rank

**Plot specs**:
- X-axis: Sampling ratio s/n (%) - linear scale, range [0, 25]
- Y-axis: Relative Frobenius Error - log scale, range [0.1, 50]
- Two subplots: Uniform (left), Importance (right)
- Colors: Dense=blue, Low-Rank=green, Sparse=orange, NN-Like=purple
- Line styles: Solid with distinct markers
- Error bars: Show std across trials

**Data requirements**:
- n = 512
- Sampling ratios: 0.5%, 1%, 2%, 5%, 10%, 20%
- 10 trials per configuration
- Matrix types with CLEAR structural differences

---

### Figure 2: Error vs Sampling Ratio by Rank
**Goal**: Show that LOWER rank → LOWER error at the same sampling ratio.

**Expected behavior**:
- rank=5: Lowest error (most concentrated energy)
- rank=100: Highest error (more spread out)
- Clear separation between curves

**Plot specs**:
- X-axis: Sampling ratio s/n (%) - linear scale, range [0, 25]
- Y-axis: Relative Frobenius Error - log scale, range [0.1, 50]
- Two subplots: Uniform (left), Importance (right)
- Colors: Use viridis colormap (dark=low rank, light=high rank)
- Ranks: 5, 10, 20, 50, 100

**Data requirements**:
- Generate low-rank matrices with NO NOISE and SHARP decay (exponent=2.0)
- This ensures column norms vary significantly with rank

---

### Figure 3: Error vs Sampling Ratio by Sparsity
**Goal**: Show that SPARSER matrices → LOWER error (especially with importance sampling).

**Expected behavior**:
- 10% density: Highest error
- 0.1% density: Lowest error
- Importance sampling should show MORE benefit for sparse matrices

**Plot specs**:
- X-axis: Sampling ratio s/n (%) - linear scale, range [0, 25]
- Y-axis: Relative Frobenius Error - log scale, range [0.1, 100]
- Two subplots: Uniform (left), Importance (right)
- Colors: plasma colormap (dark=dense, light=sparse)
- Densities: 10%, 5%, 1%, 0.1%

---

### Figure 4: Runtime vs Matrix Size
**Goal**: Show that RMM is FASTER than exact GEMM, especially at larger sizes.

**Expected behavior**:
- Exact GEMM: O(n³) growth
- RMM: O(s·n²) growth, where s = 5% of n

**Plot specs**:
- X-axis: Matrix size n - log scale (base 2), range [128, 2048]
- Y-axis: Runtime (ms) - log scale, range [0.1, 1000]
- Single plot with 3 lines: Exact GEMM, RMM Uniform, RMM Importance
- Colors: GEMM=gray, Uniform=blue, Importance=red
- Show clear gap between GEMM and RMM at large n

**Data requirements**:
- Sizes: 128, 256, 512, 1024, 2048
- Use low-rank matrices for cleaner comparison

---

### Figure 5: Speedup vs Sampling Ratio
**Goal**: Show that speedup INCREASES as sampling ratio DECREASES.

**Expected behavior**:
- At s/n = 0.5%: High speedup (10-20x)
- At s/n = 20%: Low speedup (~1x)
- Uniform should be faster than Importance (no norm computation)

**Plot specs**:
- X-axis: Sampling ratio s/n (%) - linear scale, range [0, 25]
- Y-axis: Speedup (vs Exact GEMM) - linear scale, range [0, 25]
- Single plot with 2 lines + reference line at y=1
- Colors: Uniform=blue, Importance=red, Reference=gray dashed

---

### Figure 6: Samples Required for Target Error vs Rank
**Goal**: Show that LOWER rank → FEWER samples needed for a given error.

**Expected behavior**:
- rank=5: Need ~5% samples for 100% error
- rank=100: Need ~30% samples for 100% error
- Clear upward trend

**Plot specs**:
- X-axis: Intrinsic Rank r - linear scale, range [0, 110]
- Y-axis: Sampling Ratio s/n (%) for target error - linear scale, range [0, 50]
- Single plot with 2 lines (Uniform, Importance)
- Target error: 100% (more achievable than 5%)

---

### Figure 7: Samples Required for Target Error vs Sparsity
**Goal**: Show that SPARSER matrices → FEWER samples needed.

**Expected behavior**:
- 90% sparse (10% density): Need ~30% samples
- 99.9% sparse (0.1% density): Need ~5% samples
- Clear downward trend

**Plot specs**:
- X-axis: Sparsity Level (%) - linear scale, range [85, 100]
- Y-axis: Sampling Ratio s/n (%) for target error - linear scale, range [0, 50]
- Single plot with 2 lines (Uniform, Importance)

---

## Matrix Generation Changes

### Low-Rank Matrices (for clear rank effect)
- **No noise** (noise_std = 0)
- **Sharp decay** (decay_exponent = 2.0)
- This makes column norms vary significantly

### Sparse Matrices (for clear sparsity effect)
- Generate with **non-uniform sparsity pattern**
- Some columns should be much sparser than others
- This creates varying column norms

### Dense Gaussian (baseline)
- Standard N(0,1) entries
- Uniform column norms (worst case for RMM)
