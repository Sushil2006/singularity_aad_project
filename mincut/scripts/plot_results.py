"""
Plot runtime and accuracy metrics for min-cut benchmarks.

For each distribution, this script renders a figure with two subplots:
1) Mean time per graph (ns) vs n.
2) Mean squared error (MSE) vs n.

Outputs are saved to `mincut/results/plots/<dist>_time_mse.png`.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt

STYLES = {
    "Karger": ("#1f77b4", "o"),
    "StoerWagner": ("#d62728", "s"),
}


def mean(values: Iterable[float]) -> float:
    """
    Compute arithmetic mean of provided values.

    Args:
        values (Iterable[float]): Numeric values.

    Returns:
        float: Average value.
    """
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def stddev(values: Iterable[float]) -> float:
    """
    Compute population standard deviation of provided values.

    Args:
        values (Iterable[float]): Numeric values.

    Returns:
        float: Standard deviation (0 if fewer than two values).
    """
    vals = list(values)
    if len(vals) < 2:
        return 0.0
    mu = mean(vals)
    variance = sum((v - mu) ** 2 for v in vals) / len(vals)
    return math.sqrt(variance)


def load_results(csv_path: Path) -> Dict[str, Dict[str, Dict[int, List[Tuple[float, float]]]]]:
    """
    Load benchmark rows grouped by distribution, algorithm, and n.

    Args:
        csv_path (Path): Path to raw_results.csv.

    Returns:
        Dict[str, Dict[str, Dict[int, List[Tuple[float, float]]]]]:
            dist -> algo -> n -> list of (time_per_graph_ns, mse_value).
    """
    data: Dict[str, Dict[str, Dict[int, List[Tuple[float, float]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            dist = row["dist"]
            algo = row["algo"]
            n = int(row["n"])
            graphs_per_rep = int(row["graphs_per_rep"])
            total_time_ns = int(row["total_time_ns"])
            sum_sq_error = int(row["sum_sq_error"])
            time_per_graph = total_time_ns / graphs_per_rep
            mse_value = sum_sq_error / graphs_per_rep
            data[dist][algo][n].append((time_per_graph, mse_value))
    return data


def aggregate(data: Dict[str, Dict[str, Dict[int, List[Tuple[float, float]]]]]):
    """
    Aggregate mean and std metrics per configuration.

    Args:
        data: Nested mapping from load_results.

    Returns:
        Dict[str, Dict[str, Dict[int, Tuple[float, float, float]]]]:
            dist -> algo -> n -> (mean_time_per_graph_ns, std_time_per_graph_ns, mean_mse).
    """
    agg: Dict[str, Dict[str, Dict[int, Tuple[float, float, float]]]] = defaultdict(dict)
    for dist, algo_dict in data.items():
        agg[dist] = defaultdict(dict)
        for algo, n_dict in algo_dict.items():
            for n, entries in n_dict.items():
                times = [t for t, _ in entries]
                mses = [m for _, m in entries]
                agg[dist][algo][n] = (mean(times), stddev(times), mean(mses))
    return agg


def plot_distribution(
    dist: str,
    algo_data: Dict[str, Dict[int, Tuple[float, float, float]]],
    output_dir: Path,
) -> None:
    """
    Render time and MSE plots for a single distribution.

    Args:
        dist (str): Distribution name.
        algo_data (Dict[str, Dict[int, Tuple[float, float, float]]]): Aggregated metrics.
        output_dir (Path): Directory to write plot PNG.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True)
    ax_time, ax_mse = axes

    for algo, (color, marker) in STYLES.items():
        if algo not in algo_data:
            continue
        points = sorted(algo_data[algo].items(), key=lambda x: x[0])
        ns = [n for n, _ in points]
        mean_times = [vals[0] for _, vals in points]
        mean_mses = [vals[2] for _, vals in points]
        ax_time.plot(ns, mean_times, label=algo, color=color, marker=marker, linewidth=1.6, markersize=5)
        ax_mse.plot(ns, mean_mses, label=algo, color=color, marker=marker, linewidth=1.6, markersize=5)

    ax_time.set_title(f"{dist}: mean time per graph (ns)")
    ax_time.set_xlabel("n")
    ax_time.set_ylabel("time per graph (ns)")

    ax_mse.set_title(f"{dist}: mean squared error")
    ax_mse.set_xlabel("n")
    ax_mse.set_ylabel("MSE")

    for ax in axes:
        ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.7)
        ax.legend()

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    outfile = output_dir / f"{dist}_time_mse.png"
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """
    Entry point: load data, aggregate metrics, and write plots for each distribution.
    """
    repo_root = Path(__file__).resolve().parent.parent
    csv_path = repo_root / "results" / "raw_results.csv"
    output_dir = repo_root / "results" / "plots"

    data = load_results(csv_path)
    agg = aggregate(data)

    for dist, algo_data in agg.items():
        plot_distribution(dist, algo_data, output_dir)
    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    plt.style.use("seaborn-v0_8-darkgrid")
    main()
