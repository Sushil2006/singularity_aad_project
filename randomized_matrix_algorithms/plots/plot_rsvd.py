"""Plotting utilities for RSVD experiments.

Generates publication-ready figures from:
- rsvd_rank_sweep.csv
- rsvd_hyperparam_sweep.csv
- rsvd_size_scaling.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import LogLocator, LogFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# Matplotlib defaults for readability
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titlesize": 14,
    "lines.linewidth": 2,
    "lines.markersize": 7,
    "errorbar.capsize": 3,
})

MATRIX_TYPE_COLORS = {
    "Dense Gaussian": "#d73027",   # red (worst structure)
    "Low-Rank (r=20)": "#1a9850",  # green (best structure)
    "Low-Rank (r=rank)": "#1a9850",
    "Sparse (1%)": "#3288bd",      # blue
    "NN-Like": "#9467bd",          # purple
}

POWER_COLORS = {
    0: "#92c5de",
    1: "#4393c3",
    2: "#2166ac",
}

OVERSAMPLING_MARKERS = {
    0: "o",
    5: "s",
    10: "^",
    20: "D",
}

ALGO_COLORS = {
    "rsvd": "#7b3294",
    "full_svd": "#4d4d4d",
}

ALGO_MARKERS = {
    "rsvd": "o",
    "full_svd": "x",
}

def _set_log_ticks(ax, which_axes: str = "y") -> None:
    """Add dense log ticks and labels for readability."""

    locator = LogLocator(base=10, subs=np.arange(1, 10))
    formatter = LogFormatter(base=10, labelOnlyBase=False)
    if "x" in which_axes:
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_minor_locator(locator)
        ax.xaxis.set_major_formatter(formatter)
        ax.xaxis.set_minor_formatter(formatter)
    if "y" in which_axes:
        ax.yaxis.set_major_locator(locator)
        ax.yaxis.set_minor_locator(locator)
        ax.yaxis.set_major_formatter(formatter)
        ax.yaxis.set_minor_formatter(formatter)


def load_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _group_mean_std(
    rows: List[Dict],
    group_key: str,
    x_key: str,
    y_key: str,
) -> Dict[str, Tuple[List[float], List[float], List[float]]]:
    grouped: Dict[Tuple[str, float], List[float]] = {}
    for r in rows:
        g = r[group_key]
        x = float(r[x_key])
        y = float(r[y_key])
        grouped.setdefault((g, x), []).append(y)

    result: Dict[str, Tuple[List[float], List[float], List[float]]] = {}
    groups = sorted({k[0] for k in grouped.keys()})
    for g in groups:
        xs = sorted({k[1] for k in grouped.keys() if k[0] == g})
        means = [float(np.mean(grouped[(g, x)])) for x in xs]
        stds = [float(np.std(grouped[(g, x)])) for x in xs]
        result[g] = (xs, means, stds)
    return result


# ============================================================================
# Figure 1: Error vs rank by matrix type
# ============================================================================

def plot_error_vs_rank(csv_path: Path, output_path: Path) -> None:
    rows = load_csv(csv_path)
    agg = _group_mean_std(rows, group_key="matrix_type", x_key="rank", y_key="rel_error")

    fig, ax = plt.subplots(figsize=(10, 6))
    for mtype, (ranks, means, stds) in agg.items():
        ax.errorbar(
            ranks,
            means,
            yerr=stds,
            marker="o",
            color=MATRIX_TYPE_COLORS.get(mtype, "#4d4d4d"),
            label=mtype,
        )

    ax.set_xlabel("Target Rank k")
    ax.set_ylabel("Relative Frobenius Error")
    ax.set_yscale("log")
    _set_log_ticks(ax, "y")
    ax.set_title("RSVD Error vs Rank by Matrix Type\n(Lower rank → lower error for structured matrices)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper right")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ============================================================================
# Figure 2: Effect of oversampling and power iterations
# ============================================================================

def plot_hyperparams(csv_path: Path, output_path: Path) -> None:
    rows = load_csv(csv_path)
    matrix_types = sorted({r["matrix_type"] for r in rows})
    n_types = len(matrix_types)
    fig, axes = plt.subplots(1, n_types, figsize=(7 * n_types, 6), sharey=True)
    if n_types == 1:
        axes = [axes]  # type: ignore[assignment]

    for ax_idx, matrix_type in enumerate(matrix_types):
        ax = axes[ax_idx]
        sub = [r for r in rows if r["matrix_type"] == matrix_type]

        # Group by power_iter, x=oversampling
        grouped = _group_mean_std(sub, group_key="power_iter", x_key="oversampling", y_key="rel_error")
        for power_str, (overs, means, stds) in grouped.items():
            p_int = int(float(power_str))
            ax.errorbar(
                overs,
                means,
                yerr=stds,
                marker="o",
                color=POWER_COLORS.get(p_int, "#4d4d4d"),
                label=f"power iters = {p_int}",
            )

        ax.set_xlabel("Oversampling p")
        ax.set_xscale("linear")
        ax.set_title(f"{matrix_type}")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="upper right")

    axes[0].set_ylabel("Relative Frobenius Error (log scale)")
    axes[0].set_yscale("log")
    _set_log_ticks(axes[0], "y")
    fig.suptitle("RSVD Error vs Oversampling and Power Iterations (k fixed)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ============================================================================
# Figure 3: Runtime vs size (RSVD vs full SVD)
# ============================================================================

def plot_runtime_vs_size(csv_path: Path, output_path: Path) -> None:
    rows = load_csv(csv_path)

    sizes = sorted({int(float(r["n"])) for r in rows})
    size_to_rsvd: Dict[int, List[float]] = {s: [] for s in sizes}
    size_to_baseline: Dict[int, List[float]] = {s: [] for s in sizes}

    for r in rows:
        n = int(float(r["n"]))
        size_to_rsvd[n].append(float(r["runtime_sec"]))
        size_to_baseline[n].append(float(r["baseline_runtime_sec"]))

    fig, ax = plt.subplots(figsize=(10, 6))
    for label, data, color, marker in [
        ("RSVD", size_to_rsvd, ALGO_COLORS["rsvd"], ALGO_MARKERS["rsvd"]),
        ("Full SVD", size_to_baseline, ALGO_COLORS["full_svd"], ALGO_MARKERS["full_svd"]),
    ]:
        xs = sizes
        means = [float(np.mean(data[s])) for s in xs]
        stds = [float(np.std(data[s])) for s in xs]
        ax.errorbar(xs, means, yerr=stds, color=color, marker=marker, label=label)

    ax.set_xlabel("Matrix Size n")
    ax.set_ylabel("Runtime (seconds)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    _set_log_ticks(ax, "y")
    ax.set_title("Runtime vs Matrix Size: RSVD vs Full SVD")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper left")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ============================================================================
# Figure 4: Error vs runtime scatter (Pareto frontier)
# ============================================================================

def plot_error_runtime_scatter(csv_path: Path, output_path: Path) -> None:
    rows = load_csv(csv_path)
    matrix_types = sorted({r["matrix_type"] for r in rows})

    fig, ax = plt.subplots(figsize=(12, 6))

    # Compute speedup vs error with rank encoded in size
    errors = np.array([float(r["rel_error"]) for r in rows])
    runtimes = np.array([float(r["runtime_sec"]) for r in rows])
    baseline_rt = np.array([float(r["baseline_runtime_sec"]) for r in rows])
    ranks = np.array([int(r["rank"]) for r in rows])
    speedups = baseline_rt / runtimes

    # Avoid zero errors on log axis
    eps = 1e-6
    errors = np.maximum(errors, eps)

    for mtype in matrix_types:
        mask = np.array([r["matrix_type"] == mtype for r in rows], dtype=bool)
        ax.scatter(
            errors[mask],
            speedups[mask],
            s=20 + 0.2 * ranks[mask],
            alpha=0.85,
            label=mtype,
            color=MATRIX_TYPE_COLORS.get(mtype, "#4d4d4d"),
            edgecolors="none",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Relative Frobenius Error (log scale)")
    ax.set_ylabel("Speedup vs Full SVD")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="upper right")

    # Nice ticks
    _set_log_ticks(ax, "x")
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=8))

    fig.suptitle("Speedup vs Error by Matrix Type (RSVD rank sweep)\nMarker size ∝ rank k")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ============================================================================
# Entrypoint
# ============================================================================

def generate_all_plots(results_dir: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    rank_csv = results_dir / "rsvd_rank_sweep.csv"
    hyper_csv = results_dir / "rsvd_hyperparam_sweep.csv"
    size_csv = results_dir / "rsvd_size_scaling.csv"

    if rank_csv.exists():
        plot_error_vs_rank(rank_csv, output_dir / "rsvd_fig1_error_vs_rank.png")
        plot_error_runtime_scatter(rank_csv, output_dir / "rsvd_fig4_error_vs_runtime.png")
    else:
        print(f"Skipping: {rank_csv} not found")

    if hyper_csv.exists():
        plot_hyperparams(hyper_csv, output_dir / "rsvd_fig2_hyperparams.png")
    else:
        print(f"Skipping: {hyper_csv} not found")

    if size_csv.exists():
        plot_runtime_vs_size(size_csv, output_dir / "rsvd_fig3_runtime_vs_size.png")
    else:
        print(f"Skipping: {size_csv} not found")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RSVD plots from experiment CSVs")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("randomized_matrix_algorithms/rsvd/results"),
        help="Directory containing RSVD CSVs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("randomized_matrix_algorithms/rsvd/results/figures"),
        help="Directory to save plots",
    )
    args = parser.parse_args()
    generate_all_plots(args.results_dir, args.output_dir)


if __name__ == "__main__":  # pragma: no cover - plotting entrypoint
    main()
