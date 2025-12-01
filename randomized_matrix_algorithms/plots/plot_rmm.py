"""Comprehensive plotting for RMM experiments following plan.md and RMM_PLOT_SPEC.md.

Generates all figures required by the experimental plan with proper:
- Axis scales and ranges for clear differentiation
- Distinct colors and markers for each category
- Clear labels and titles explaining what to infer
- Error bars showing variance across trials

Key theoretical predictions to demonstrate:
- Error decays as ~1/√s
- Importance sampling reduces variance vs uniform
- Structured matrices (low-rank, sparse) need fewer samples
- Lower rank → lower error at same sampling ratio
- Sparser matrices → lower error at same sampling ratio
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# Set up matplotlib for publication-quality figures
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'lines.linewidth': 2,
    'lines.markersize': 8,
    'errorbar.capsize': 3,
})

# Color schemes for different plot types
ALGO_COLORS = {
    "rmm_uniform": "#2166ac",      # Blue
    "rmm_importance": "#b2182b",   # Red
    "naive_matmul": "#7f7f7f",     # Gray (baseline)
}

ALGO_MARKERS = {
    "rmm_uniform": "o",
    "rmm_importance": "s",
    "naive_matmul": "^",
}

ALGO_LABELS = {
    "rmm_uniform": "RMM (Uniform Sampling)",
    "rmm_importance": "RMM (Importance Sampling)",
    "naive_matmul": "Naive MatMul (Baseline)",
}

# Distinct colors for matrix types
MATRIX_TYPE_COLORS = {
    "Dense Gaussian": "#1f77b4",   # Blue
    "Low-Rank (r=10)": "#2ca02c",  # Green
    "Sparse (1%)": "#ff7f0e",      # Orange
    "NN-Like": "#9467bd",          # Purple
}

MATRIX_TYPE_MARKERS = {
    "Dense Gaussian": "o",
    "Low-Rank (r=10)": "s",
    "Sparse (1%)": "^",
    "NN-Like": "D",
}


def load_csv(path: Path) -> List[Dict[str, str]]:
    """Load CSV rows as list of dicts."""
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _format_log_axis(ax, axis='y'):
    """Format log axis to show more tick labels with actual numbers."""
    if axis == 'y':
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))
        ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=10))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:g}'))
    else:
        ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))
        ax.xaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=10))
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:g}'))


def _group_and_aggregate(
    rows: List[Dict], 
    group_key: str, 
    x_key: str, 
    y_key: str,
    filter_fn=None
) -> Dict[str, Tuple[List[float], List[float], List[float]]]:
    """Group rows and compute mean/std for each x value.
    
    Returns dict mapping group_key value to (x_vals, y_means, y_stds).
    """
    if filter_fn:
        rows = [r for r in rows if filter_fn(r)]
    
    # Group by (group_key, x_key)
    groups: Dict[Tuple[str, float], List[float]] = {}
    for r in rows:
        gk = r[group_key]
        xv = float(r[x_key])
        yv = float(r[y_key])
        key = (gk, xv)
        if key not in groups:
            groups[key] = []
        groups[key].append(yv)
    
    # Aggregate by group_key
    result: Dict[str, Tuple[List[float], List[float], List[float]]] = {}
    group_keys = sorted(set(k[0] for k in groups.keys()))
    
    for gk in group_keys:
        x_vals = sorted(set(k[1] for k in groups.keys() if k[0] == gk))
        y_means = [np.mean(groups[(gk, x)]) for x in x_vals]
        y_stds = [np.std(groups[(gk, x)]) for x in x_vals]
        result[gk] = (x_vals, y_means, y_stds)
    
    return result


# ============================================================================
# Figure 1: Error vs Sampling Ratio across Matrix Types
# ============================================================================

def plot_error_vs_sampling_ratio_by_matrix_type(
    csv_path: Path, output_path: Path
) -> None:
    """
    Figure 1: Error vs sampling ratio for different matrix types.
    
    Goal: Show that structured matrices (low-rank, sparse) achieve LOWER error 
    than dense Gaussian at the same sampling ratio.
    
    Expected: Dense Gaussian > NN-Like > Low-Rank ≈ Sparse
    """
    rows = load_csv(csv_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Get unique matrix types in desired order
    matrix_type_order = ["Dense Gaussian", "NN-Like", "Low-Rank (r=10)", "Sparse (1%)"]
    matrix_types = [mt for mt in matrix_type_order if mt in set(r["matrix_type"] for r in rows)]
    
    # Colors for clear differentiation
    colors = {
        "Dense Gaussian": "#d62728",   # Red (worst)
        "NN-Like": "#9467bd",          # Purple
        "Low-Rank (r=10)": "#2ca02c",  # Green (good)
        "Sparse (1%)": "#1f77b4",      # Blue (good)
    }
    markers = {
        "Dense Gaussian": "o",
        "NN-Like": "D",
        "Low-Rank (r=10)": "s",
        "Sparse (1%)": "^",
    }
    
    for ax_idx, algo in enumerate(["rmm_uniform", "rmm_importance"]):
        ax = axes[ax_idx]
        algo_rows = [r for r in rows if r["algo"] == algo]
        
        for mtype in matrix_types:
            mtype_rows = [r for r in algo_rows if r["matrix_type"] == mtype]
            
            # Group by sampling ratio
            ratio_to_errs: Dict[float, List[float]] = {}
            for r in mtype_rows:
                ratio = float(r["sampling_ratio"])
                err = float(r["rel_error"])
                if ratio not in ratio_to_errs:
                    ratio_to_errs[ratio] = []
                ratio_to_errs[ratio].append(err)
            
            ratios = sorted(ratio_to_errs.keys())
            means = [np.mean(ratio_to_errs[r]) for r in ratios]
            stds = [np.std(ratio_to_errs[r]) for r in ratios]
            
            ax.plot(
                [r * 100 for r in ratios], means,
                marker=markers.get(mtype, "o"), 
                color=colors.get(mtype, "gray"),
                label=mtype, linewidth=2.5, markersize=8
            )
        
        ax.set_xlabel("Sampling Ratio s/n (%)", fontsize=12)
        ax.set_ylabel("Relative Frobenius Error", fontsize=12)
        ax.set_title(f"{ALGO_LABELS[algo]}", fontsize=13, fontweight="bold")
        ax.set_yscale("log")
        ax.set_xlim(0, 22)
        ax.set_ylim(0.5, 50)
        # Add more y-axis tick labels
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:g}'))
        ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=[2, 3, 4, 5, 6, 7, 8, 9], numticks=10))
        ax.yaxis.set_minor_formatter(ticker.FuncFormatter(lambda x, p: f'{x:g}' if x in [2, 5, 20] else ''))
        ax.legend(loc="upper right", fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, which='both')
    
    fig.suptitle(
        "Figure 1: RMM Error vs Sampling Ratio by Matrix Type\n"
        "Goal: Structured matrices (Low-Rank, Sparse) should have LOWER error than Dense Gaussian",
        fontsize=12, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ============================================================================
# Figure 2: Error vs Sampling Ratio for Different Ranks
# ============================================================================

def plot_error_vs_sampling_ratio_by_rank(
    csv_path: Path, output_path: Path
) -> None:
    """
    Figure 2: Error vs sampling ratio for different intrinsic ranks.
    
    Goal: Show that LOWER rank → LOWER error at the same sampling ratio.
    
    Expected: rank=5 (lowest error) < rank=10 < rank=20 < rank=50 < rank=100 (highest error)
    """
    rows = load_csv(csv_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    ranks = sorted(set(int(r["rank"]) for r in rows))
    
    # Use distinct colors - darker = lower rank (better)
    rank_colors = {
        5: "#1a9850",    # Dark green (best)
        10: "#91cf60",   # Light green
        20: "#fee08b",   # Yellow
        50: "#fc8d59",   # Orange
        100: "#d73027",  # Red (worst)
    }
    rank_markers = {5: "o", 10: "s", 20: "^", 50: "D", 100: "v"}
    
    for ax_idx, algo in enumerate(["rmm_uniform", "rmm_importance"]):
        ax = axes[ax_idx]
        algo_rows = [r for r in rows if r["algo"] == algo]
        
        for rank in ranks:
            rank_rows = [r for r in algo_rows if int(r["rank"]) == rank]
            
            ratio_to_errs: Dict[float, List[float]] = {}
            for r in rank_rows:
                ratio = float(r["sampling_ratio"])
                err = float(r["rel_error"])
                if ratio not in ratio_to_errs:
                    ratio_to_errs[ratio] = []
                ratio_to_errs[ratio].append(err)
            
            ratios = sorted(ratio_to_errs.keys())
            means = [np.mean(ratio_to_errs[r]) for r in ratios]
            stds = [np.std(ratio_to_errs[r]) for r in ratios]
            
            ax.plot(
                [r * 100 for r in ratios], means,
                marker=rank_markers.get(rank, "o"), 
                color=rank_colors.get(rank, "gray"),
                label=f"rank = {rank}",
                linewidth=2.5, markersize=8
            )
        
        ax.set_xlabel("Sampling Ratio s/n (%)", fontsize=12)
        ax.set_ylabel("Relative Frobenius Error", fontsize=12)
        ax.set_title(f"{ALGO_LABELS[algo]}", fontsize=13, fontweight="bold")
        ax.set_yscale("log")
        ax.set_xlim(0, 22)
        ax.set_ylim(0.5, 50)
        # Add more y-axis tick labels
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:g}'))
        ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=[2, 3, 4, 5, 6, 7, 8, 9], numticks=10))
        ax.yaxis.set_minor_formatter(ticker.FuncFormatter(lambda x, p: f'{x:g}' if x in [2, 5, 20] else ''))
        ax.legend(loc="upper right", fontsize=10, title="Intrinsic Rank", framealpha=0.9)
        ax.grid(True, alpha=0.3, which='both')
    
    fig.suptitle(
        "Figure 2: RMM Error vs Sampling Ratio by Intrinsic Rank\n"
        "Goal: Lower rank should have LOWER error (green < yellow < red)",
        fontsize=12, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ============================================================================
# Figure 3: Error vs Sampling Ratio for Different Sparsity Levels
# ============================================================================

def plot_error_vs_sampling_ratio_by_sparsity(
    csv_path: Path, output_path: Path
) -> None:
    """
    Figure 3: Error vs sampling ratio for different sparsity levels.
    
    Goal: Show that SPARSER matrices → LOWER error at the same sampling ratio.
    
    Expected: 90% sparse (worst) > 95% > 99% > 99.9% sparse (best)
    """
    rows = load_csv(csv_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    densities = sorted(set(float(r["density"]) for r in rows), reverse=True)
    
    # Distinct colors - sparser = greener (better)
    sparsity_colors = {
        0.10: "#d73027",   # Red (10% density = 90% sparse, worst)
        0.05: "#fc8d59",   # Orange (5% density = 95% sparse)
        0.01: "#91cf60",   # Light green (1% density = 99% sparse)
        0.001: "#1a9850",  # Dark green (0.1% density = 99.9% sparse, best)
    }
    sparsity_markers = {0.10: "o", 0.05: "s", 0.01: "^", 0.001: "D"}
    
    for ax_idx, algo in enumerate(["rmm_uniform", "rmm_importance"]):
        ax = axes[ax_idx]
        algo_rows = [r for r in rows if r["algo"] == algo]
        
        for density in densities:
            density_rows = [r for r in algo_rows if abs(float(r["density"]) - density) < 1e-6]
            
            ratio_to_errs: Dict[float, List[float]] = {}
            for r in density_rows:
                ratio = float(r["sampling_ratio"])
                err = float(r["rel_error"])
                if ratio not in ratio_to_errs:
                    ratio_to_errs[ratio] = []
                ratio_to_errs[ratio].append(err)
            
            ratios = sorted(ratio_to_errs.keys())
            means = [np.mean(ratio_to_errs[r]) for r in ratios]
            stds = [np.std(ratio_to_errs[r]) for r in ratios]
            
            sparsity_pct = (1 - density) * 100
            ax.plot(
                [r * 100 for r in ratios], means,
                marker=sparsity_markers.get(density, "o"), 
                color=sparsity_colors.get(density, "gray"),
                label=f"{sparsity_pct:.1f}% sparse",
                linewidth=2.5, markersize=8
            )
        
        ax.set_xlabel("Sampling Ratio s/n (%)", fontsize=12)
        ax.set_ylabel("Relative Frobenius Error", fontsize=12)
        ax.set_title(f"{ALGO_LABELS[algo]}", fontsize=13, fontweight="bold")
        ax.set_yscale("log")
        ax.set_xlim(0, 22)
        ax.set_ylim(0.1, 100)
        # Add more y-axis tick labels
        ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=10))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:g}'))
        ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=[2, 3, 4, 5, 6, 7, 8, 9], numticks=10))
        ax.yaxis.set_minor_formatter(ticker.FuncFormatter(lambda x, p: f'{x:g}' if x in [0.2, 0.5, 2, 5, 20, 50] else ''))
        ax.legend(loc="upper right", fontsize=10, title="Sparsity Level", framealpha=0.9)
        ax.grid(True, alpha=0.3, which='both')
    
    fig.suptitle(
        "Figure 3: RMM Error vs Sampling Ratio by Sparsity Level\n"
        "Goal: Sparser matrices should have LOWER error (green < red)",
        fontsize=12, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ============================================================================
# Figure 4: Runtime vs Matrix Size
# ============================================================================

def plot_runtime_vs_size(csv_path: Path, output_path: Path) -> None:
    """
    Figure 4: Runtime vs matrix size for naive matmul and RMM.
    
    Goal: Show that RMM is FASTER than naive matmul (fair comparison).
    
    Expected: Naive matmul grows as O(n³), RMM grows as O(s·n²) where s = 5% of n.
    """
    rows = load_csv(csv_path)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    algos = ["naive_matmul", "rmm_uniform", "rmm_importance"]
    
    # Distinct colors and styles
    colors = {
        "naive_matmul": "#7f7f7f",   # Gray (baseline)
        "rmm_uniform": "#2166ac",    # Blue
        "rmm_importance": "#b2182b", # Red
    }
    markers = {"naive_matmul": "^", "rmm_uniform": "o", "rmm_importance": "s"}
    linestyles = {"naive_matmul": "--", "rmm_uniform": "-", "rmm_importance": "-"}
    
    for algo in algos:
        algo_rows = [r for r in rows if r["algo"] == algo]
        
        # For RMM, pick a representative sampling ratio (e.g., 5%)
        if algo != "naive_matmul":
            algo_rows = [r for r in algo_rows if abs(float(r["sampling_ratio"]) - 0.05) < 0.01]
        
        size_to_times: Dict[int, List[float]] = {}
        for r in algo_rows:
            n = int(r["n"])
            t = float(r["runtime_sec"]) * 1000  # Convert to ms
            if n not in size_to_times:
                size_to_times[n] = []
            size_to_times[n].append(t)
        
        sizes = sorted(size_to_times.keys())
        means = [np.mean(size_to_times[n]) for n in sizes]
        stds = [np.std(size_to_times[n]) for n in sizes]
        
        label = "Naive MatMul (baseline)" if algo == "naive_matmul" else f"{ALGO_LABELS[algo]} (s/n=5%)"
        
        ax.plot(
            sizes, means,
            marker=markers[algo], color=colors[algo], 
            linestyle=linestyles[algo],
            label=label, linewidth=2.5, markersize=9
        )
    
    ax.set_xlabel("Matrix Size n", fontsize=12)
    ax.set_ylabel("Runtime (ms)", fontsize=12)
    ax.set_title(
        "Figure 4: Runtime vs Matrix Size (Fair Comparison)\n"
        "Goal: RMM should be FASTER than Naive MatMul (blue/red below gray)",
        fontsize=13, fontweight="bold"
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    # Add more y-axis tick labels
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=15))
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:g}'))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=[2, 3, 4, 5, 6, 7, 8, 9], numticks=10))
    ax.yaxis.set_minor_formatter(ticker.FuncFormatter(lambda x, p: f'{x:g}' if x in [0.2, 0.5, 2, 5, 20, 50, 200, 500, 2000, 5000] else ''))
    ax.legend(loc="upper left", fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3, which='both')
    
    # Add custom x-tick labels
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x)}'))
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ============================================================================
# Figure 5: Speedup vs Sampling Ratio
# ============================================================================

def plot_speedup_vs_sampling_ratio(csv_path: Path, output_path: Path) -> None:
    """
    Figure 5: Speedup vs sampling ratio.
    
    Goal: Show that speedup INCREASES as sampling ratio DECREASES.
    
    Expected: At s/n=0.5%, speedup should be highest; at s/n=20%, speedup ~1x.
    """
    rows = load_csv(csv_path)
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = {"rmm_uniform": "#2166ac", "rmm_importance": "#b2182b"}
    markers = {"rmm_uniform": "o", "rmm_importance": "s"}
    
    for algo in ["rmm_uniform", "rmm_importance"]:
        algo_rows = [r for r in rows if r["algo"] == algo]
        
        ratio_to_speedups: Dict[float, List[float]] = {}
        for r in algo_rows:
            ratio = float(r["sampling_ratio"])
            speedup = float(r["speedup"])
            if ratio not in ratio_to_speedups:
                ratio_to_speedups[ratio] = []
            ratio_to_speedups[ratio].append(speedup)
        
        ratios = sorted(ratio_to_speedups.keys())
        means = [np.mean(ratio_to_speedups[r]) for r in ratios]
        stds = [np.std(ratio_to_speedups[r]) for r in ratios]
        
        ax.plot(
            [r * 100 for r in ratios], means,
            marker=markers[algo], color=colors[algo], 
            label=ALGO_LABELS[algo],
            linewidth=2.5, markersize=9
        )
    
    # Reference line at speedup = 1
    ax.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, alpha=0.7, label='No speedup (1x)')
    
    ax.set_xlabel("Sampling Ratio s/n (%)", fontsize=12)
    ax.set_ylabel("Speedup (vs Naive MatMul)", fontsize=12)
    ax.set_title(
        "Figure 5: RMM Speedup vs Sampling Ratio (Fair Comparison)\n"
        "Goal: Speedup should INCREASE as sampling ratio DECREASES (left side higher)",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlim(0, 22)
    ax.set_ylim(0, None)  # Start from 0 to show full scale
    ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ============================================================================
# Figure 6: Samples Required for Target Error vs Rank
# ============================================================================

def plot_samples_for_error_vs_rank(csv_path: Path, output_path: Path) -> None:
    """
    Figure 6: Minimum samples needed for target error vs intrinsic rank.
    
    Goal: Show that LOWER rank → FEWER samples needed for a given error.
    
    Expected: Upward trend - rank=5 needs fewer samples than rank=100.
    """
    rows = load_csv(csv_path)
    rank_rows = [r for r in rows if r["experiment"] == "rank_sweep"]
    
    # Get available target errors and find one with actual data
    target_errors = sorted(set(float(r["target_error"]) for r in rank_rows), reverse=True)
    target_error = 2.0  # Default
    for te in target_errors:
        has_data = any(float(r["ratio_needed"]) > 0 for r in rank_rows 
                       if abs(float(r["target_error"]) - te) < 0.01)
        if has_data:
            target_error = te
            break
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = {"rmm_uniform": "#2166ac", "rmm_importance": "#b2182b"}
    markers = {"rmm_uniform": "o", "rmm_importance": "s"}
    
    for algo in ["rmm_uniform", "rmm_importance"]:
        algo_rows = [r for r in rank_rows 
                     if r["algo"] == algo and abs(float(r["target_error"]) - target_error) < 0.01]
        
        rank_to_samples: Dict[int, List[float]] = {}
        for r in algo_rows:
            rank = int(r["rank"])
            ratio = float(r["ratio_needed"])
            if ratio > 0:
                if rank not in rank_to_samples:
                    rank_to_samples[rank] = []
                rank_to_samples[rank].append(ratio * 100)
        
        if not rank_to_samples:
            continue
            
        ranks = sorted(rank_to_samples.keys())
        means = [np.mean(rank_to_samples[r]) for r in ranks]
        stds = [np.std(rank_to_samples[r]) for r in ranks]
        
        ax.plot(
            ranks, means,
            marker=markers[algo], color=colors[algo], 
            label=ALGO_LABELS[algo],
            linewidth=2.5, markersize=9
        )
    
    ax.set_xlabel("Intrinsic Rank r", fontsize=12)
    ax.set_ylabel(f"Sampling Ratio s/n (%) for <{target_error*100:.0f}% Error", fontsize=12)
    ax.set_title(
        f"Figure 6: Samples Required for <{target_error*100:.0f}% Error vs Rank\n"
        "Goal: Lower rank should need FEWER samples (upward trend expected)",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlim(0, 110)
    ax.set_ylim(0, 50)
    ax.legend(loc="upper left", fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ============================================================================
# Figure 7: Samples Required for Target Error vs Sparsity
# ============================================================================

def plot_samples_for_error_vs_sparsity(csv_path: Path, output_path: Path) -> None:
    """
    Figure 7: Minimum samples needed for target error vs sparsity level.
    
    Goal: Show that SPARSER matrices → FEWER samples needed.
    
    Expected: Downward trend - 99.9% sparse needs fewer samples than 90% sparse.
    """
    rows = load_csv(csv_path)
    sparsity_rows = [r for r in rows if r["experiment"] == "sparsity_sweep"]
    
    # Get available target errors and find one with actual data
    target_errors = sorted(set(float(r["target_error"]) for r in sparsity_rows), reverse=True)
    target_error = 2.0
    for te in target_errors:
        has_data = any(float(r["ratio_needed"]) > 0 for r in sparsity_rows 
                       if abs(float(r["target_error"]) - te) < 0.01)
        if has_data:
            target_error = te
            break
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    colors = {"rmm_uniform": "#2166ac", "rmm_importance": "#b2182b"}
    markers = {"rmm_uniform": "o", "rmm_importance": "s"}
    
    for algo in ["rmm_uniform", "rmm_importance"]:
        algo_rows = [r for r in sparsity_rows 
                     if r["algo"] == algo and abs(float(r["target_error"]) - target_error) < 0.01]
        
        density_to_samples: Dict[float, List[float]] = {}
        for r in algo_rows:
            density = float(r["density"])
            ratio = float(r["ratio_needed"])
            if ratio > 0:
                if density not in density_to_samples:
                    density_to_samples[density] = []
                density_to_samples[density].append(ratio * 100)
        
        if not density_to_samples:
            continue
            
        densities = sorted(density_to_samples.keys(), reverse=True)
        sparsities = [(1 - d) * 100 for d in densities]
        means = [np.mean(density_to_samples[d]) for d in densities]
        stds = [np.std(density_to_samples[d]) for d in densities]
        
        ax.plot(
            sparsities, means,
            marker=markers[algo], color=colors[algo], 
            label=ALGO_LABELS[algo],
            linewidth=2.5, markersize=9
        )
    
    ax.set_xlabel("Sparsity Level (%)", fontsize=12)
    ax.set_ylabel(f"Sampling Ratio s/n (%) for <{target_error*100:.0f}% Error", fontsize=12)
    ax.set_title(
        f"Figure 7: Samples Required for <{target_error*100:.0f}% Error vs Sparsity\n"
        "Goal: Sparser matrices should need FEWER samples (downward trend expected)",
        fontsize=13, fontweight="bold"
    )
    ax.set_xlim(88, 100)
    ax.set_ylim(0, 50)
    ax.legend(loc="upper left", fontsize=11, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ============================================================================
# Main Entry Point
# ============================================================================

def generate_all_plots(results_dir: Path, output_dir: Path) -> None:
    """Generate all RMM plots from experiment results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Generating RMM plots from plan.md")
    print("=" * 60)
    
    # Figure 1: Matrix type comparison
    csv_path = results_dir / "rmm_matrix_type_comparison.csv"
    if csv_path.exists():
        plot_error_vs_sampling_ratio_by_matrix_type(
            csv_path, output_dir / "fig1_error_vs_ratio_by_matrix_type.png"
        )
    else:
        print(f"Skipping Figure 1: {csv_path} not found")
    
    # Figure 2: Rank sweep
    csv_path = results_dir / "rmm_rank_sweep.csv"
    if csv_path.exists():
        plot_error_vs_sampling_ratio_by_rank(
            csv_path, output_dir / "fig2_error_vs_ratio_by_rank.png"
        )
    else:
        print(f"Skipping Figure 2: {csv_path} not found")
    
    # Figure 3: Sparsity sweep
    csv_path = results_dir / "rmm_sparsity_sweep.csv"
    if csv_path.exists():
        plot_error_vs_sampling_ratio_by_sparsity(
            csv_path, output_dir / "fig3_error_vs_ratio_by_sparsity.png"
        )
    else:
        print(f"Skipping Figure 3: {csv_path} not found")
    
    # Figure 4: Runtime vs size
    csv_path = results_dir / "rmm_size_scaling.csv"
    if csv_path.exists():
        plot_runtime_vs_size(
            csv_path, output_dir / "fig4_runtime_vs_size.png"
        )
    else:
        print(f"Skipping Figure 4: {csv_path} not found")
    
    # Figure 5: Speedup vs sampling ratio
    csv_path = results_dir / "rmm_size_scaling.csv"
    if csv_path.exists():
        plot_speedup_vs_sampling_ratio(
            csv_path, output_dir / "fig5_speedup_vs_ratio.png"
        )
    else:
        print(f"Skipping Figure 5: {csv_path} not found")
    
    # Figure 6 & 7: Samples for target error
    csv_path = results_dir / "rmm_samples_for_target_error.csv"
    if csv_path.exists():
        plot_samples_for_error_vs_rank(
            csv_path, output_dir / "fig6_samples_for_5pct_error_vs_rank.png"
        )
        plot_samples_for_error_vs_sparsity(
            csv_path, output_dir / "fig7_samples_for_5pct_error_vs_sparsity.png"
        )
    else:
        print(f"Skipping Figures 6-7: {csv_path} not found")
    
    print("=" * 60)
    print(f"All plots saved to {output_dir}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RMM plots from experiment results")
    parser.add_argument(
        "--results-dir", type=Path,
        default=Path("randomized_matrix_algorithms/rmm/results"),
        help="Directory containing experiment CSV files"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("randomized_matrix_algorithms/rmm/results/figures"),
        help="Output directory for plots"
    )
    args = parser.parse_args()
    
    generate_all_plots(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
