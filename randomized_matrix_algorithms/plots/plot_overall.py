"""Joint comparison plots for all matrix multiplication methods.

Following overall_combined/plan.md, generates:
1. Error vs Runtime scatter plots per workload
2. Speedup vs Error curves
3. Best config under X% error tables (as plots)
4. Scaling with matrix size N
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

# Style configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'legend.fontsize': 9,
    'figure.figsize': (12, 8),
    'figure.dpi': 150,
})

# Color scheme for methods
METHOD_COLORS = {
    "NumPy GEMM": "#2c3e50",       # Dark blue-gray (baseline)
    "Naive GEMM": "#7f8c8d",       # Gray
    "Strassen": "#27ae60",         # Green (exact)
    "RMM-Uniform": "#3498db",      # Blue
    "RMM-Importance": "#9b59b6",   # Purple
    "LR-GEMM-RSVD": "#e74c3c",     # Red
    "LR-GEMM-Det": "#f39c12",      # Orange
}

METHOD_MARKERS = {
    "NumPy GEMM": "X",
    "Naive GEMM": "^",
    "Strassen": "D",
    "RMM-Uniform": "o",
    "RMM-Importance": "s",
    "LR-GEMM-RSVD": "v",
    "LR-GEMM-Det": "P",
}

# Workload colors
WORKLOAD_COLORS = {
    "NN-Like": "#e74c3c",
    "Dense Gaussian": "#3498db",
    "Low-Rank": "#27ae60",
    "Recsys": "#9b59b6",
    "Sparse": "#f39c12",
}


def _save_figure(fig: plt.Figure, path: Path) -> None:
    """Save figure with tight layout."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved: {path}")


# ============================================================================
# Figure 1: Error vs Runtime Scatter (per workload)
# ============================================================================

def plot_error_vs_runtime_scatter(csv_path: Path, output_dir: Path) -> None:
    """Plot error vs runtime scatter for each workload.
    
    Each point = one configuration of one method.
    Color = method, size = matrix size.
    """
    df = pd.read_csv(csv_path)
    
    # Get unique workloads
    workloads = df["workload"].unique()
    
    # Create one plot per workload category
    workload_categories = {
        "NN-Like": [w for w in workloads if "NN" in w],
        "Dense Gaussian": [w for w in workloads if "Gaussian" in w],
        "Low-Rank": [w for w in workloads if "Low-Rank" in w],
        "Recsys": [w for w in workloads if "Recsys" in w],
        "Sparse": [w for w in workloads if "Sparse" in w],
    }
    
    for category, category_workloads in workload_categories.items():
        if not category_workloads:
            continue
        
        subset = df[df["workload"].isin(category_workloads)]
        if subset.empty:
            continue
        
        # Aggregate by method, config, n
        agg = subset.groupby(["method", "config", "n"]).agg({
            "runtime_sec": "mean",
            "rel_error": "mean",
        }).reset_index()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        for method in METHOD_COLORS:
            method_data = agg[agg["method"] == method]
            if method_data.empty:
                continue
            
            # Size by matrix dimension
            sizes = method_data["n"].values
            size_scale = (sizes / sizes.max()) * 200 + 50
            
            ax.scatter(
                method_data["runtime_sec"] * 1000,  # Convert to ms
                method_data["rel_error"],
                c=METHOD_COLORS.get(method, "gray"),
                marker=METHOD_MARKERS.get(method, "o"),
                s=size_scale,
                alpha=0.7,
                label=method,
                edgecolors='black',
                linewidth=0.5,
            )
        
        ax.set_xlabel("Runtime (ms)")
        ax.set_ylabel("Relative Frobenius Error")
        ax.set_xscale("log")
        ax.set_yscale("log")
        
        # Format axes
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:g}'))
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:g}'))
        
        ax.legend(loc="upper left", framealpha=0.9)
        ax.grid(True, alpha=0.3, which='both')
        
        ax.set_title(
            f"Error vs Runtime: {category} Workload\n"
            "Goal: Lower-left is better (low error, fast runtime)"
        )
        
        _save_figure(fig, output_dir / f"scatter_error_runtime_{category.lower().replace(' ', '_')}.png")


# ============================================================================
# Figure 2: Speedup vs Error Curves
# ============================================================================

def plot_speedup_vs_error(csv_path: Path, output_dir: Path) -> None:
    """Plot speedup vs error curves for each method.
    
    Shows the tradeoff frontier for each method.
    """
    df = pd.read_csv(csv_path)
    
    # Filter to a single representative size
    n_target = df["n"].max()
    df = df[df["n"] == n_target]
    
    # Aggregate by method and config
    agg = df.groupby(["method", "config"]).agg({
        "speedup_vs_naive": "mean",
        "rel_error": "mean",
    }).reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    for method in METHOD_COLORS:
        method_data = agg[agg["method"] == method].sort_values("rel_error")
        if method_data.empty:
            continue
        
        # Skip exact methods for this plot (they have ~0 error)
        if method in ["NumPy GEMM", "Naive GEMM", "Strassen"]:
            # Plot as single point
            ax.scatter(
                method_data["rel_error"].values,
                method_data["speedup_vs_naive"].values,
                c=METHOD_COLORS[method],
                marker=METHOD_MARKERS[method],
                s=200,
                label=method,
                edgecolors='black',
                linewidth=1,
                zorder=10,
            )
        else:
            # Plot as curve
            ax.plot(
                method_data["rel_error"].values,
                method_data["speedup_vs_naive"].values,
                marker=METHOD_MARKERS[method],
                color=METHOD_COLORS[method],
                linewidth=2,
                markersize=8,
                label=method,
            )
    
    ax.set_xlabel("Relative Frobenius Error")
    ax.set_ylabel("Speedup (vs Naive MatMul)")
    ax.set_xscale("log")
    
    # Reference line at speedup = 1
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.7, label='No speedup (1x)')
    
    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    ax.set_title(
        f"Speedup vs Error Tradeoff (N={n_target})\n"
        "Goal: Upper-left is better (high speedup, low error)"
    )
    
    _save_figure(fig, output_dir / "speedup_vs_error_curves.png")


# ============================================================================
# Figure 3: Method Comparison by Workload
# ============================================================================

def plot_method_comparison_by_workload(csv_path: Path, output_dir: Path) -> None:
    """Bar chart comparing methods across workloads at fixed error budget."""
    df = pd.read_csv(csv_path)
    
    # Filter to largest size
    n_target = df["n"].max()
    df = df[df["n"] == n_target]
    
    # Get approximate methods only
    approx_methods = ["RMM-Uniform", "RMM-Importance", "LR-GEMM-RSVD", "LR-GEMM-Det"]
    df = df[df["method"].isin(approx_methods)]
    
    # For each workload, find best config per method (lowest error)
    workload_categories = ["NN-Like", "Dense Gaussian", "Low-Rank", "Sparse"]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for ax_idx, category in enumerate(workload_categories):
        ax = axes[ax_idx]
        
        # Filter to this category
        subset = df[df["workload"].str.contains(category.split()[0])]
        if subset.empty:
            ax.set_visible(False)
            continue
        
        # Get best config per method (lowest error that's still reasonable)
        best_per_method = subset.groupby("method").apply(
            lambda x: x.loc[x["rel_error"].idxmin()]
        ).reset_index(drop=True)
        
        methods = best_per_method["method"].values
        speedups = best_per_method["speedup_vs_naive"].values
        errors = best_per_method["rel_error"].values
        
        x = np.arange(len(methods))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, speedups, width, label='Speedup', 
                       color=[METHOD_COLORS.get(m, 'gray') for m in methods])
        
        ax2 = ax.twinx()
        bars2 = ax2.bar(x + width/2, errors * 100, width, label='Error (%)', 
                        color=[METHOD_COLORS.get(m, 'gray') for m in methods], alpha=0.5)
        
        ax.set_xlabel("Method")
        ax.set_ylabel("Speedup (vs Naive)")
        ax2.set_ylabel("Error (%)")
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("-", "\n") for m in methods], fontsize=9)
        ax.set_title(f"{category} Workload")
        ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
        
    fig.suptitle(
        f"Method Comparison by Workload (N={n_target})\n"
        "Bars show speedup (solid) and error (transparent)",
        fontsize=14, fontweight='bold'
    )
    plt.tight_layout()
    
    _save_figure(fig, output_dir / "method_comparison_by_workload.png")


# ============================================================================
# Figure 4: Scaling with Matrix Size
# ============================================================================

def plot_scaling_with_size(csv_path: Path, output_dir: Path) -> None:
    """Plot runtime and speedup scaling with matrix size."""
    df = pd.read_csv(csv_path)
    
    # Aggregate by method and n
    agg = df.groupby(["method", "n"]).agg({
        "runtime_sec": "mean",
        "speedup_vs_naive": "mean",
        "rel_error": "mean",
    }).reset_index()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Runtime vs Size
    ax1 = axes[0]
    for method in METHOD_COLORS:
        method_data = agg[agg["method"] == method].sort_values("n")
        if method_data.empty:
            continue
        
        ax1.plot(
            method_data["n"],
            method_data["runtime_sec"] * 1000,  # ms
            marker=METHOD_MARKERS.get(method, "o"),
            color=METHOD_COLORS[method],
            linewidth=2,
            markersize=8,
            label=method,
        )
    
    ax1.set_xlabel("Matrix Size N")
    ax1.set_ylabel("Runtime (ms)")
    ax1.set_xscale("log", base=2)
    ax1.set_yscale("log")
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x)}'))
    ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{x:g}'))
    ax1.legend(loc="upper left", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Runtime vs Matrix Size")
    
    # Plot 2: Speedup vs Size
    ax2 = axes[1]
    for method in METHOD_COLORS:
        if method in ["NumPy GEMM", "Naive GEMM"]:
            continue  # Skip baselines
        
        method_data = agg[agg["method"] == method].sort_values("n")
        if method_data.empty:
            continue
        
        ax2.plot(
            method_data["n"],
            method_data["speedup_vs_naive"],
            marker=METHOD_MARKERS.get(method, "o"),
            color=METHOD_COLORS[method],
            linewidth=2,
            markersize=8,
            label=method,
        )
    
    ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.7, label='No speedup (1x)')
    ax2.set_xlabel("Matrix Size N")
    ax2.set_ylabel("Speedup (vs Naive MatMul)")
    ax2.set_xscale("log", base=2)
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, p: f'{int(x)}'))
    ax2.legend(loc="upper left", fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Speedup vs Matrix Size")
    
    fig.suptitle(
        "Scaling with Matrix Size (NN-Like Workload)\n"
        "Goal: Speedup should increase with size for approximate methods",
        fontsize=14, fontweight='bold'
    )
    
    _save_figure(fig, output_dir / "scaling_with_size.png")


# ============================================================================
# Figure 5: Best Config Under Error Budget
# ============================================================================

def plot_error_budget_table(csv_path: Path, output_dir: Path) -> None:
    """Visualize best configurations under different error budgets."""
    df = pd.read_csv(csv_path)
    
    # Filter out "None" results
    df = df[df["best_method"] != "None"]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    workloads = df["workload"].unique()
    error_budgets = sorted(df["error_budget_pct"].unique())
    
    # Create a grouped bar chart
    x = np.arange(len(workloads))
    width = 0.15
    
    for i, budget in enumerate(error_budgets):
        budget_data = df[df["error_budget_pct"] == budget]
        speedups = []
        colors = []
        
        for workload in workloads:
            row = budget_data[budget_data["workload"] == workload]
            if not row.empty:
                speedups.append(row["speedup_vs_naive"].values[0])
                colors.append(METHOD_COLORS.get(row["best_method"].values[0], "gray"))
            else:
                speedups.append(0)
                colors.append("gray")
        
        offset = (i - len(error_budgets)/2 + 0.5) * width
        bars = ax.bar(x + offset, speedups, width, label=f'≤{budget:.0f}% error', alpha=0.8)
        
        # Color bars by best method
        for bar, color in zip(bars, colors):
            bar.set_facecolor(color)
    
    ax.set_xlabel("Workload")
    ax.set_ylabel("Speedup (vs Naive MatMul)")
    ax.set_xticks(x)
    ax.set_xticklabels(workloads, rotation=15, ha='right')
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
    ax.legend(title="Error Budget", loc="upper right")
    ax.grid(True, alpha=0.3, axis='y')
    
    ax.set_title(
        "Best Speedup Under Error Budget by Workload\n"
        "Bar color indicates best method for that budget"
    )
    
    _save_figure(fig, output_dir / "error_budget_comparison.png")


# ============================================================================
# Figure 6: Comprehensive Heatmap
# ============================================================================

def plot_method_workload_heatmap(csv_path: Path, output_dir: Path) -> None:
    """Heatmap showing speedup for each method-workload combination."""
    df = pd.read_csv(csv_path)
    
    # Filter to largest size and approximate methods
    n_target = df["n"].max()
    df = df[df["n"] == n_target]
    
    # Get best config per method (by speedup)
    approx_methods = ["RMM-Uniform", "RMM-Importance", "LR-GEMM-RSVD", "LR-GEMM-Det", "Strassen"]
    df = df[df["method"].isin(approx_methods)]
    
    # Create pivot table
    pivot = df.groupby(["workload", "method"]).agg({
        "speedup_vs_naive": "max",
        "rel_error": "min",
    }).reset_index()
    
    # Get unique workloads and methods
    workloads = sorted(pivot["workload"].unique())
    methods = approx_methods
    
    # Create speedup matrix
    speedup_matrix = np.zeros((len(workloads), len(methods)))
    for i, workload in enumerate(workloads):
        for j, method in enumerate(methods):
            row = pivot[(pivot["workload"] == workload) & (pivot["method"] == method)]
            if not row.empty:
                speedup_matrix[i, j] = row["speedup_vs_naive"].values[0]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    im = ax.imshow(speedup_matrix, cmap='RdYlGn', aspect='auto')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Speedup (vs Naive MatMul)")
    
    # Set ticks
    ax.set_xticks(np.arange(len(methods)))
    ax.set_yticks(np.arange(len(workloads)))
    ax.set_xticklabels([m.replace("-", "\n") for m in methods], fontsize=10)
    ax.set_yticklabels(workloads, fontsize=9)
    
    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    for i in range(len(workloads)):
        for j in range(len(methods)):
            val = speedup_matrix[i, j]
            if val > 0:
                text = ax.text(j, i, f"{val:.1f}x",
                              ha="center", va="center", color="black", fontsize=9)
    
    ax.set_title(
        f"Speedup Heatmap: Method vs Workload (N={n_target})\n"
        "Green = high speedup, Red = low speedup"
    )
    
    _save_figure(fig, output_dir / "method_workload_heatmap.png")


# ============================================================================
# Main Entry Point
# ============================================================================

def generate_all_plots(results_dir: Path, output_dir: Path) -> None:
    """Generate all overall comparison plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Generating Overall Comparison Plots")
    print("=" * 60)
    
    # Main comparison plots
    comparison_csv = results_dir / "overall_comparison.csv"
    if comparison_csv.exists():
        plot_error_vs_runtime_scatter(comparison_csv, output_dir)
        plot_speedup_vs_error(comparison_csv, output_dir)
        plot_method_comparison_by_workload(comparison_csv, output_dir)
        plot_method_workload_heatmap(comparison_csv, output_dir)
    else:
        print(f"Skipping comparison plots: {comparison_csv} not found")
    
    # Scaling plots
    scaling_csv = results_dir / "scaling_comparison.csv"
    if scaling_csv.exists():
        plot_scaling_with_size(scaling_csv, output_dir)
    else:
        print(f"Skipping scaling plots: {scaling_csv} not found")
    
    # Error budget plots
    budget_csv = results_dir / "error_budget_comparison.csv"
    if budget_csv.exists():
        plot_error_budget_table(budget_csv, output_dir)
    else:
        print(f"Skipping error budget plots: {budget_csv} not found")
    
    print("=" * 60)
    print(f"All plots saved to {output_dir}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Overall Comparison Plots")
    parser.add_argument(
        "--results-dir", type=Path,
        default=Path("randomized_matrix_algorithms/overall/results"),
        help="Directory containing experiment CSV files"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("randomized_matrix_algorithms/overall/results/figures"),
        help="Output directory for plots"
    )
    args = parser.parse_args()
    
    generate_all_plots(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
