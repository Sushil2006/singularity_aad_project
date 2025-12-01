# Overall Comparison Results & Documentation

## Overview

This module provides a unified benchmark comparing all matrix multiplication methods:
- **Exact Methods**: NumPy GEMM, Naive GEMM, Strassen
- **Approximate Methods**: RMM (Uniform/Importance), LR-GEMM (RSVD/Deterministic)

Tested across multiple workloads:
- NN-Like (heavy-tailed spectrum)
- Dense Gaussian (worst case)
- Low-Rank (best case for LR-GEMM)
- Recsys-style (low-rank with noise)
- Sparse matrices (various densities)

---

## Key Findings

### 1. Speedup Heatmap (Method vs Workload)

| Method | Dense Gaussian | NN-Like | Low-Rank | Sparse |
|--------|---------------|---------|----------|--------|
| **RMM-Uniform** | 430x | 470x | 327-403x | 345-425x |
| **RMM-Importance** | 288x | 301x | 225-246x | 219-258x |
| **LR-GEMM-RSVD** | 135x | 172x | 123-134x | 118-155x |
| **LR-GEMM-Det** | 2-4x | 4x | 2-4x | 2-4x |
| **Strassen** | 21x | 19x | 16-21x | 16-19x |

**Key Insight**: RMM-Uniform provides the highest speedups (300-470x) but with high error (~100%). LR-GEMM-RSVD provides moderate speedups (120-170x) with much lower error when matrices are truly low-rank.

### 2. Error vs Runtime Tradeoff

- **Exact methods** (NumPy, Naive, Strassen): Zero error, varying runtime
- **RMM methods**: High speedup (100-350x) but high error (50-300%)
- **LR-GEMM-RSVD**: Moderate speedup (10-130x) with error depending on intrinsic rank
- **LR-GEMM-Det**: Low speedup (2-4x) due to SVD cost, but optimal low-rank error

### 3. Scaling with Matrix Size

At N=1024:
- **RMM-Uniform**: ~340x speedup
- **RMM-Importance**: ~190x speedup
- **LR-GEMM-RSVD**: ~55x speedup
- **Strassen**: ~20x speedup

Speedup increases with matrix size for all approximate methods.

### 4. Best Method Under Error Budget

| Workload | ≤1% Error | ≤5% Error | ≤10% Error | ≤50% Error |
|----------|-----------|-----------|------------|------------|
| NN-Like | LR-GEMM-RSVD (8x) | LR-GEMM-RSVD (23x) | - | LR-GEMM-RSVD (56x) |
| Dense Gaussian | None | None | None | None |
| Low-Rank (r=20) | LR-GEMM-RSVD (10x) | LR-GEMM-RSVD (19x) | - | LR-GEMM-RSVD (54x) |
| Sparse (95%) | None | None | None | None |

**Key Insight**: LR-GEMM-RSVD is the only method that can achieve low error (<5%) with meaningful speedup. RMM methods are too noisy for low error budgets.

---

## When to Use Each Method

### NumPy GEMM
- **Use when**: You need exact results and maximum accuracy
- **Pros**: Highly optimized, zero error
- **Cons**: O(N³) complexity

### Strassen
- **Use when**: Large square matrices, exact results needed
- **Pros**: Asymptotically faster O(N^2.81)
- **Cons**: Only helps at large N, overhead at small sizes

### RMM (Uniform/Importance)
- **Use when**: Very high error tolerance (>50%), maximum speed needed
- **Pros**: Extremely fast (300-470x speedup)
- **Cons**: High error, not suitable for precision-critical applications

### LR-GEMM-RSVD
- **Use when**: Matrices are approximately low-rank, moderate error acceptable
- **Pros**: Good speedup (10-170x) with controllable error
- **Cons**: Requires low-rank structure, offline factorization cost

### LR-GEMM-Deterministic
- **Use when**: Optimal low-rank approximation needed, offline cost acceptable
- **Pros**: Best possible low-rank error
- **Cons**: Slow due to full SVD

---

## Files Generated

### CSVs (in `results/`):
- `overall_comparison.csv` - Full comparison across all methods/workloads (2475 rows)
- `scaling_comparison.csv` - Scaling with matrix size (84 rows)
- `error_budget_comparison.csv` - Best configs under error budgets (20 rows)

### Figures (in `results/figures/`):
1. `scatter_error_runtime_*.png` - Error vs runtime scatter per workload (5 plots)
2. `speedup_vs_error_curves.png` - Speedup vs error tradeoff curves
3. `method_comparison_by_workload.png` - Bar chart comparison
4. `method_workload_heatmap.png` - Speedup heatmap
5. `scaling_with_size.png` - Runtime and speedup vs N
6. `error_budget_comparison.png` - Best speedup under error budgets

---

## Usage

```python
from randomized_matrix_algorithms.overall.experiments import run_all_experiments
from randomized_matrix_algorithms.plots.plot_overall import generate_all_plots
from pathlib import Path

# Run experiments
output_dir = Path("randomized_matrix_algorithms/overall/results")
run_all_experiments(output_dir, seed=42)

# Generate plots
generate_all_plots(output_dir, output_dir / "figures")
```

---

## Conclusions

1. **No single winner**: The best method depends on matrix structure and error tolerance
2. **RMM is fastest but noisiest**: 300-470x speedup but ~100% error
3. **LR-GEMM-RSVD is the sweet spot**: 10-170x speedup with controllable error for low-rank matrices
4. **Dense Gaussian is hardest**: No approximate method achieves low error
5. **Speedup scales with size**: All methods benefit from larger matrices
