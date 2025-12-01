"""
Generate comparative plots for sorting benchmarks.

For each distribution, this script creates a figure with two side-by-side subplots:
1) Mean time (ns) vs n.
2) Mean swaps vs n.

Scales are chosen automatically (log if the data spans multiple orders of magnitude).
Outputs are saved to `results/plots/<dist>_time_swaps.png`.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt

# Colors and markers for consistent styling across plots.
STYLES = {
    "QS": ("#1f77b4", "o"),
    "QS_small": ("#ff7f0e", "s"),
    "QS_cut16": ("#2ca02c", "^"),
    "MS": ("#d62728", "v"),
    "STD": ("#9467bd", "D"),
}


def choose_scale(values: Iterable[float]) -> str:
    """
    Pick 'log' if the value range spans multiple orders of magnitude; otherwise 'linear'.

    Args:
        values (Iterable[float]): Data values to inspect.

    Returns:
        str: 'log' or 'linear'.
    """
    vals = [v for v in values if v > 0]
    if not vals:
        return "linear"
    min_v = min(vals)
    max_v = max(vals)
    return "log" if max_v / min_v > 50 else "linear"


def load_results(csv_path: Path) -> Dict[str, Dict[str, Dict[int, List[Tuple[int, int]]]]]:
    """
    Load benchmark rows grouped by distribution, algorithm, and n.

    Args:
        csv_path (Path): Path to raw_results.csv.

    Returns:
        Dict[str, Dict[str, Dict[int, List[Tuple[int, int]]]]]: Nested mapping dist -> algo -> n -> list of (time_ns, swaps).
    """
    data: Dict[str, Dict[str, Dict[int, List[Tuple[int, int]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            dist = row["dist"]
            algo = row["algo"]
            n = int(row["n"])
            time_ns = int(row["time_ns"])
            swaps = int(row["swaps"])
            data[dist][algo][n].append((time_ns, swaps))
    return data


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


def aggregate_means(raw: Dict[str, Dict[str, Dict[int, List[Tuple[int, int]]]]]):
    """
    Aggregate mean time and mean swaps per (dist, algo, n).

    Args:
        raw: Nested mapping from load_results.

    Returns:
        Dict[str, Dict[str, Dict[int, Tuple[float, float | None]]]]:
            dist -> algo -> n -> (mean_time_ns, mean_swaps or None if unavailable).
    """
    agg: Dict[str, Dict[str, Dict[int, Tuple[float, float | None]]]] = defaultdict(dict)
    for dist, algo_dict in raw.items():
        agg[dist] = defaultdict(dict)
        for algo, n_dict in algo_dict.items():
            for n, entries in n_dict.items():
                times = [t for t, _ in entries]
                swaps_vals = [s for _, s in entries if s >= 0]
                mean_time = mean(times)
                mean_swaps = mean(swaps_vals) if swaps_vals else None
                agg[dist][algo][n] = (mean_time, mean_swaps)
    return agg


def plot_distribution(
    dist: str,
    algo_data: Dict[str, Dict[int, Tuple[float, float | None]]],
    output_dir: Path,
) -> None:
    """
    Render the time/swaps plots for a single distribution.

    Args:
        dist (str): Distribution name.
        algo_data (Dict[str, Dict[int, Tuple[float, float | None]]]): Aggregated means for this distribution.
        output_dir (Path): Directory to write plot PNG.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    ax_time, ax_swaps = axes

    # Collect ranges for scale decisions.
    time_values = []
    swap_values = []

    for algo, (color, marker) in STYLES.items():
        if algo not in algo_data:
            continue
        points = sorted(algo_data[algo].items(), key=lambda x: x[0])
        ns = [n for n, _ in points]
        times = [vals[0] for _, vals in points]
        swaps = [vals[1] for _, vals in points if vals[1] is not None]

        ax_time.plot(ns, times, label=algo, color=color, marker=marker, linewidth=1.6, markersize=5)
        time_values.extend(times)

        # Only plot swaps when available (std::sort reports -1).
        swap_points = [(n, vals[1]) for n, vals in points if vals[1] is not None]
        if swap_points:
            ns_swaps = [n for n, _ in swap_points]
            swap_vals = [s for _, s in swap_points]
            ax_swaps.plot(
                ns_swaps, swap_vals, label=algo, color=color, marker=marker, linewidth=1.6, markersize=5
            )
            swap_values.extend(swap_vals)

    ax_time.set_title(f"{dist}: mean time (ns)")
    ax_time.set_xlabel("n")
    ax_time.set_ylabel("time (ns)")

    ax_swaps.set_title(f"{dist}: mean swaps")
    ax_swaps.set_xlabel("n")
    ax_swaps.set_ylabel("swaps")

    # Axis scales.
    x_values_all = sorted({n for algo in algo_data.values() for n in algo.keys()})
    x_scale = choose_scale(x_values_all)
    ax_time.set_xscale(x_scale)
    ax_swaps.set_xscale(x_scale)
    ax_time.set_yscale(choose_scale(time_values))
    if swap_values:
        ax_swaps.set_yscale(choose_scale(swap_values))

    for ax in axes:
        ax.grid(True, which="both", linestyle="--", linewidth=0.6, alpha=0.7)
        ax.legend()

    fig.suptitle(f"Distribution: {dist}", fontsize=14, y=1.02)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    outfile = output_dir / f"{dist}_time_swaps.png"
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """
    Entry point: load data, aggregate metrics, and write plots for each distribution.
    """
    repo_root = Path(__file__).resolve().parent.parent
    csv_path = repo_root / "results" / "raw_results.csv"
    output_dir = repo_root / "results" / "plots"

    raw = load_results(csv_path)
    agg = aggregate_means(raw)

    for dist, algo_data in agg.items():
        plot_distribution(dist, algo_data, output_dir)
    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    plt.style.use("seaborn-v0_8-darkgrid")
    main()
