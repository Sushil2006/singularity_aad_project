"""Plotting functions for Low-Rank GEMM experiments.

Generates figures following low_rank_approx_matrix_mul/plan.md:
1. Error vs rank r (by matrix type)
2. Speedup vs rank r
3. Error-speedup tradeoff curves
4. Error vs intrinsic rank
5. Error vs sparsity level
6. Runtime vs matrix size
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 10,
    'figure.figsize': (10, 6),
    'figure.dpi': 150,
})

# Color schemes
MATRIX_TYPE_COLORS = {
    "Dense Gaussian": "#d62728",    # Red - worst case
    "Low-Rank (r=20)": "#2ca02c",   # Green - best case
    "Sparse (1%)": "#1f77b4",       # Blue
    "NN-Like": "#9467bd",           # Purple - practical case
}

ALGO_COLORS = {
    "lrgemm_rsvd": "#2166ac",       # Blue
    "lrgemm_det": "#b2182b",        # Red
    "naive_matmul": "#7f7f7f",      # Gray
}

ALGO_LABELS = {
    "lrgemm_rsvd": "Low-Rank GEMM (RSVD)",
    "lrgemm_det": "Low-Rank GEMM (Exact SVD)",
    "naive_matmul": "Naive MatMul (baseline)",
}

ALGO_MARKERS = {
    "lrgemm_rsvd": "o",
    "lrgemm_det": "s",
    "naive_matmul": "^",
}

# Intrinsic rank colors (green to red gradient)
INTRINSIC_RANK_COLORS = {
    5: "#1a9850",
    10: "#91cf60",
    20: "#fee08b",
    50: "#fc8d59",
    100: "#d73027",
}

# Sparsity colors (red to green - sparser is better)
SPARSITY_COLORS = {
    0.10: "#d73027",   # 90% sparse
    0.05: "#fc8d59",   # 95% sparse
    0.01: "#91cf60",   # 99% sparse
    0.001: "#1a9850",  # 99.9% sparse
}


def _save_figure(fig: plt.Figure, path: Path) -> None:
    """Save figure with tight layout."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


# ============================================================================
# Figure 1: Error vs Rank by Matrix Type
# ============================================================================

def plot_error_vs_rank_by_matrix_type(csv_path: Path, output_path: Path) -> None:
    """Plot relative error vs target rank for different matrix types.
    
    Goal: Show that low-rank and NN-like matrices achieve low error with small r,
    while dense Gaussian needs larger r.
    """
    df = pd.read_csv(csv_path)
    
    # Filter to RSVD algorithm only for cleaner plot
    df = df[df["algo"] == "lrgemm_rsvd"]
    
    # Aggregate by matrix_type and rank
    agg = df.groupby(["matrix_type", "rank"]).agg({
        "rel_error": ["mean", "std"]
    }).reset_index()
    agg.columns = ["matrix_type", "rank", "error_mean", "error_std"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for matrix_type in MATRIX_TYPE_COLORS:
        subset = agg[agg["matrix_type"] == matrix_type]
        if subset.empty:
            continue
        
        color = MATRIX_TYPE_COLORS[matrix_type]
        ax.plot(
            subset["rank"], subset["error_mean"],
            marker='o', linewidth=2, markersize=8,
            color=color, label=matrix_type
        )
    
    ax.set_xlabel("Target Rank r")
    ax.set_ylabel("Relative Frobenius Error")
    ax.set_yscale("log")
    ax.set_xlim(0, max(df["rank"]) * 1.1)
    ax.legend(loc="upper right", title="Matrix Type")
    ax.grid(True, alpha=0.3)
    
    ax.set_title(
        "Figure 1: Low-Rank GEMM Error vs Target Rank\n"
        "Goal: Structured matrices (Low-Rank, NN-Like) achieve lower error with smaller r"
    )
    
    _save_figure(fig, output_path)


# ============================================================================
# Figure 2: Speedup vs Rank
# ============================================================================

def plot_speedup_vs_rank(csv_path: Path, output_path: Path) -> None:
    """Plot speedup vs target rank.
    
    Goal: Show that speedup decreases as rank increases (tradeoff).
    """
    df = pd.read_csv(csv_path)
    
    # Aggregate by algo and rank
    agg = df.groupby(["algo", "rank"]).agg({
        "speedup_online": ["mean", "std"]
    }).reset_index()
    agg.columns = ["algo", "rank", "speedup_mean", "speedup_std"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for algo in ["lrgemm_rsvd", "lrgemm_det"]:
        subset = agg[agg["algo"] == algo]
        if subset.empty:
            continue
        
        color = ALGO_COLORS[algo]
        marker = ALGO_MARKERS[algo]
        label = ALGO_LABELS[algo]
        
        ax.plot(
            subset["rank"], subset["speedup_mean"],
            marker=marker, linewidth=2, markersize=8,
            color=color, label=label
        )
    
    # Reference line at speedup = 1
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.7, label='No speedup (1x)')
    
    ax.set_xlabel("Target Rank r")
    ax.set_ylabel("Speedup (vs Naive MatMul)")
    ax.set_xlim(0, max(df["rank"]) * 1.1)
    ax.set_ylim(bottom=0)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    
    ax.set_title(
        "Figure 2: Low-Rank GEMM Speedup vs Target Rank\n"
        "Goal: Speedup DECREASES as rank increases (smaller r = faster)"
    )
    
    _save_figure(fig, output_path)


# ============================================================================
# Figure 3: Error-Speedup Tradeoff
# ============================================================================

def plot_error_speedup_tradeoff(csv_path: Path, output_path: Path) -> None:
    """Plot error vs speedup tradeoff curve.
    
    Goal: Visualize the Pareto frontier of error-speedup tradeoff.
    """
    df = pd.read_csv(csv_path)
    
    # Filter to RSVD and exclude r=4
    df = df[df["algo"] == "lrgemm_rsvd"]
    df = df[df["rank"] != 4]  # Remove r=4 for cleaner plot
    
    # Aggregate by rank
    agg = df.groupby("rank").agg({
        "rel_error": "mean",
        "speedup_online": "mean"
    }).reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Color by rank
    scatter = ax.scatter(
        agg["speedup_online"], agg["rel_error"],
        c=agg["rank"], cmap='viridis_r', s=150, edgecolors='black', linewidth=1
    )
    
    # Connect points with line
    agg_sorted = agg.sort_values("rank")
    ax.plot(
        agg_sorted["speedup_online"], agg_sorted["rel_error"],
        'k--', alpha=0.5, linewidth=1
    )
    
    # Annotate ranks
    for _, row in agg.iterrows():
        ax.annotate(
            f"r={int(row['rank'])}",
            (row["speedup_online"], row["rel_error"]),
            textcoords="offset points", xytext=(5, 5),
            fontsize=9
        )
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Target Rank r")
    
    ax.set_xlabel("Speedup (vs Naive MatMul)")
    ax.set_ylabel("Relative Frobenius Error")
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3)
    
    ax.set_title(
        "Figure 3: Error-Speedup Tradeoff (Low-Rank GEMM)\n"
        "Goal: Lower-left is better (low error, high speedup)"
    )
    
    _save_figure(fig, output_path)


# ============================================================================
# Figure 4: Error vs Intrinsic Rank
# ============================================================================

def plot_error_vs_intrinsic_rank(csv_path: Path, output_path: Path) -> None:
    """Plot error vs intrinsic rank for different target ranks.
    
    Goal: Show that when target rank >= intrinsic rank, error is very small.
    """
    df = pd.read_csv(csv_path)
    
    # Aggregate
    agg = df.groupby(["intrinsic_rank", "target_rank"]).agg({
        "rel_error": ["mean", "std"]
    }).reset_index()
    agg.columns = ["intrinsic_rank", "target_rank", "error_mean", "error_std"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot for each target rank (exclude rank 8)
    target_ranks = sorted([r for r in agg["target_rank"].unique() if r != 8])
    colors = plt.cm.viridis(np.linspace(0, 1, len(target_ranks)))
    
    for i, target_rank in enumerate(target_ranks):
        subset = agg[agg["target_rank"] == target_rank]
        ax.plot(
            subset["intrinsic_rank"], subset["error_mean"],
            marker='o', linewidth=2, markersize=8,
            color=colors[i], label=f"Target r={target_rank}"
        )
    
    # Add diagonal reference (where target_rank = intrinsic_rank)
    ax.axvline(x=0, color='gray', linestyle=':', alpha=0.5)
    
    ax.set_xlabel("Intrinsic Rank of Matrices")
    ax.set_ylabel("Relative Frobenius Error")
    ax.set_yscale("log")
    ax.set_xlim(0, max(df["intrinsic_rank"]) * 1.1)
    ax.legend(loc="upper left", title="Target Rank")
    ax.grid(True, alpha=0.3)
    
    ax.set_title(
        "Figure 4: Error vs Intrinsic Rank\n"
        "Goal: Error is small when target rank >= intrinsic rank"
    )
    
    _save_figure(fig, output_path)


# ============================================================================
# Figure 5: Error vs Sparsity
# ============================================================================

def plot_error_vs_sparsity(csv_path: Path, output_path: Path) -> None:
    """Plot error vs sparsity level for different target ranks.
    
    Goal: Show how sparsity affects low-rank GEMM performance.
    """
    df = pd.read_csv(csv_path)
    
    # Aggregate
    agg = df.groupby(["sparsity_pct", "rank"]).agg({
        "rel_error": ["mean", "std"]
    }).reset_index()
    agg.columns = ["sparsity_pct", "rank", "error_mean", "error_std"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot for each rank
    ranks = sorted(agg["rank"].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(ranks)))
    
    for i, rank in enumerate(ranks):
        subset = agg[agg["rank"] == rank]
        ax.plot(
            subset["sparsity_pct"], subset["error_mean"],
            marker='o', linewidth=2, markersize=8,
            color=colors[i], label=f"r={rank}"
        )
    
    ax.set_xlabel("Sparsity Level (%)")
    ax.set_ylabel("Relative Frobenius Error")
    ax.set_yscale("log")
    ax.set_xlim(85, 100)
    ax.legend(loc="upper left", title="Target Rank")  # Changed to upper left to avoid overlap
    ax.grid(True, alpha=0.3)
    
    ax.set_title(
        "Figure 5: Error vs Sparsity Level\n"
        "Goal: Show how sparsity interacts with low-rank approximation"
    )
    
    _save_figure(fig, output_path)


# ============================================================================
# Figure 6: Runtime vs Matrix Size
# ============================================================================

def plot_runtime_vs_size(csv_path: Path, output_path: Path) -> None:
    """Plot runtime vs matrix size.
    
    Goal: Show that low-rank GEMM scales better than naive matmul.
    """
    df = pd.read_csv(csv_path)
    
    # Aggregate
    agg = df.groupby(["algo", "n", "rank"]).agg({
        "online_runtime_sec": ["mean", "std"]
    }).reset_index()
    agg.columns = ["algo", "n", "rank", "runtime_mean", "runtime_std"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot naive matmul baseline
    baseline = agg[agg["algo"] == "naive_matmul"]
    if not baseline.empty:
        ax.plot(
            baseline["n"], baseline["runtime_mean"] * 1000,  # Convert to ms
            marker='^', linewidth=2, markersize=10,
            color='gray', linestyle='--', label='Naive MatMul (baseline)'
        )
    
    # Plot low-rank GEMM for different ranks
    lrgemm = agg[agg["algo"] == "lrgemm_rsvd"]
    ranks = sorted(lrgemm["rank"].unique())
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(ranks)))
    
    for i, rank in enumerate(ranks):
        subset = lrgemm[lrgemm["rank"] == rank]
        ax.plot(
            subset["n"], subset["runtime_mean"] * 1000,
            marker='o', linewidth=2, markersize=8,
            color=colors[i], label=f'LR-GEMM (r={int(rank)})'
        )
    
    ax.set_xlabel("Matrix Size n")
    ax.set_ylabel("Runtime (ms)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    
    ax.set_title(
        "Figure 6: Runtime vs Matrix Size\n"
        "Goal: Low-Rank GEMM should be FASTER than Naive MatMul (below gray line)"
    )
    
    _save_figure(fig, output_path)


# ============================================================================
# Figure 7: Speedup vs Matrix Size
# ============================================================================

def plot_speedup_vs_size(csv_path: Path, output_path: Path) -> None:
    """Plot speedup vs matrix size.
    
    Goal: Show that speedup increases with matrix size (approaches N/r).
    """
    df = pd.read_csv(csv_path)
    
    # Filter to lrgemm_rsvd
    df = df[df["algo"] == "lrgemm_rsvd"]
    
    # Aggregate
    agg = df.groupby(["n", "rank"]).agg({
        "speedup_online": ["mean", "std"]
    }).reset_index()
    agg.columns = ["n", "rank", "speedup_mean", "speedup_std"]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ranks = sorted(agg["rank"].unique())
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(ranks)))
    
    for i, rank in enumerate(ranks):
        subset = agg[agg["rank"] == rank]
        ax.plot(
            subset["n"], subset["speedup_mean"],
            marker='o', linewidth=2, markersize=8,
            color=colors[i], label=f'r={int(rank)}'
        )
    
    # Reference line at speedup = 1
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.7, label='No speedup (1x)')
    
    ax.set_xlabel("Matrix Size n")
    ax.set_ylabel("Speedup (vs Naive MatMul)")
    ax.set_xscale("log", base=2)
    ax.legend(loc="upper left", title="Target Rank")
    ax.grid(True, alpha=0.3)
    
    ax.set_title(
        "Figure 7: Speedup vs Matrix Size\n"
        "Goal: Speedup should INCREASE with matrix size (asymptotically N/r)"
    )
    
    _save_figure(fig, output_path)


# ============================================================================
# Main Entry Point
# ============================================================================

def generate_all_plots(results_dir: Path, output_dir: Path) -> None:
    """Generate all low-rank GEMM plots from experiment results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Generating Low-Rank GEMM plots from plan.md")
    print("=" * 60)
    
    # Figure 1: Error vs rank by matrix type
    csv_path = results_dir / "lrgemm_matrix_type_comparison.csv"
    if csv_path.exists():
        plot_error_vs_rank_by_matrix_type(
            csv_path, output_dir / "fig1_error_vs_rank_by_matrix_type.png"
        )
    else:
        print(f"Skipping Figure 1: {csv_path} not found")
    
    # Figure 2: Speedup vs rank
    csv_path = results_dir / "lrgemm_rank_sweep.csv"
    if csv_path.exists():
        plot_speedup_vs_rank(
            csv_path, output_dir / "fig2_speedup_vs_rank.png"
        )
    else:
        print(f"Skipping Figure 2: {csv_path} not found")
    
    # Figure 3: Error-speedup tradeoff
    csv_path = results_dir / "lrgemm_rank_sweep.csv"
    if csv_path.exists():
        plot_error_speedup_tradeoff(
            csv_path, output_dir / "fig3_error_speedup_tradeoff.png"
        )
    else:
        print(f"Skipping Figure 3: {csv_path} not found")
    
    # Figure 4: Error vs intrinsic rank
    csv_path = results_dir / "lrgemm_intrinsic_rank_sweep.csv"
    if csv_path.exists():
        plot_error_vs_intrinsic_rank(
            csv_path, output_dir / "fig4_error_vs_intrinsic_rank.png"
        )
    else:
        print(f"Skipping Figure 4: {csv_path} not found")
    
    # Figure 5: Error vs sparsity
    csv_path = results_dir / "lrgemm_sparsity_sweep.csv"
    if csv_path.exists():
        plot_error_vs_sparsity(
            csv_path, output_dir / "fig5_error_vs_sparsity.png"
        )
    else:
        print(f"Skipping Figure 5: {csv_path} not found")
    
    # Figure 6: Runtime vs size
    csv_path = results_dir / "lrgemm_size_scaling.csv"
    if csv_path.exists():
        plot_runtime_vs_size(
            csv_path, output_dir / "fig6_runtime_vs_size.png"
        )
    else:
        print(f"Skipping Figure 6: {csv_path} not found")
    
    # Figure 7: Speedup vs size
    csv_path = results_dir / "lrgemm_size_scaling.csv"
    if csv_path.exists():
        plot_speedup_vs_size(
            csv_path, output_dir / "fig7_speedup_vs_size.png"
        )
    else:
        print(f"Skipping Figure 7: {csv_path} not found")
    
    print("=" * 60)
    print(f"All plots saved to {output_dir}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Low-Rank GEMM plots")
    parser.add_argument(
        "--results-dir", type=Path,
        default=Path("randomized_matrix_algorithms/low_rank_gemm/results"),
        help="Directory containing experiment CSV files"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("randomized_matrix_algorithms/low_rank_gemm/results/figures"),
        help="Output directory for plots"
    )
    args = parser.parse_args()
    
    generate_all_plots(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
