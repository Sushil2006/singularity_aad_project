"""
Plot benchmark results for Miller–Rabin experiments.

Generates per-distribution figures (time/error vs bits) and a highlight bar chart
for Carmichael numbers. Outputs are saved under `miller_rabin/results/plots/`.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt

STYLES = {
    "TD": ("#1f77b4", "o"),
    "Fermat": ("#ff7f0e", "s"),
    "MR": ("#2ca02c", "^"),
}


def mean(values: Iterable[float]) -> float:
    """
    Compute arithmetic mean.

    Args:
        values (Iterable[float]): Numeric values.

    Returns:
        float: Average value.
    """
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def choose_scale(values: Iterable[float]) -> str:
    """
    Decide between linear and log scale based on spread.

    Args:
        values (Iterable[float]): Values to inspect.

    Returns:
        str: 'log' or 'linear'.
    """
    vals = [v for v in values if v > 0]
    if not vals:
        return "linear"
    mn, mx = min(vals), max(vals)
    return "log" if mx / mn > 50 else "linear"


def load_results(csv_path: Path):
    """
    Load raw results into nested dictionaries.

    Args:
        csv_path (Path): Path to raw_results.csv.

    Returns:
        Dict[tuple, List[Tuple[float, float, float]]]: Mapping (dist, algo, bits, rounds) -> list of (time_ns_total, error_count, sample_count).
    """
    data: Dict[tuple, List[Tuple[float, float, float]]] = defaultdict(list)
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["dist"], row["algo"], int(row["bits"]), int(row["rounds"]))
            data[key].append((float(row["time_ns_total"]), float(row["error_count"]), float(row["sample_count"])))
    return data


def aggregate(data):
    """
    Aggregate means per configuration.

    Args:
        data: Output from load_results.

    Returns:
        Dict[tuple, Tuple[float, float, float, float]]: Mapping (dist, algo, bits, rounds) -> (mean_time_per_test_ns, mean_error_rate, total_errors, total_tests).
    """
    agg = {}
    for key, rows in data.items():
        times = [t / s for t, _, s in rows]
        errors = [e / s for _, e, s in rows]
        total_errors = sum(e for _, e, _ in rows)
        total_tests = sum(s for _, _, s in rows)
        agg[key] = (mean(times), mean(errors), total_errors, total_tests)
    return agg


def plot_dist(dist: str, rounds_values: List[int], agg):
    """
    Plot time and error vs bits for a distribution across round settings.

    Args:
        dist (str): Distribution name.
        rounds_values (List[int]): Round counts present.
        agg: Aggregated data mapping.
    """
    output_dir = Path(__file__).resolve().parent.parent / "results" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    for rounds in rounds_values:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
        ax_time, ax_err = axes
        time_vals = []
        err_vals = []

        for algo, (color, marker) in STYLES.items():
            xs = []
            ys_time = []
            ys_err = []
            for bits in sorted({k[2] for k in agg if k[0] == dist and k[1] == algo and k[3] == rounds}):
                key = (dist, algo, bits, rounds)
                if key not in agg:
                    continue
                mean_time, mean_err, _, _ = agg[key]
                xs.append(bits)
                ys_time.append(mean_time)
                ys_err.append(mean_err)
            if not xs:
                continue
            ax_time.plot(xs, ys_time, label=algo, color=color, marker=marker, linewidth=1.5, markersize=5)
            ax_err.plot(xs, ys_err, label=algo, color=color, marker=marker, linewidth=1.5, markersize=5)
            time_vals.extend(ys_time)
            err_vals.extend(ys_err)

        ax_time.set_title(f"{dist} (rounds={rounds}) time per test (ns)")
        ax_time.set_xlabel("bits")
        ax_time.set_ylabel("time (ns)")
        ax_err.set_title(f"{dist} (rounds={rounds}) error rate")
        ax_err.set_xlabel("bits")
        ax_err.set_ylabel("error rate")

        x_values = sorted({k[2] for k in agg if k[0] == dist and k[3] == rounds})
        if x_values:
            x_scale = choose_scale(x_values)
            ax_time.set_xscale(x_scale)
            ax_err.set_xscale(x_scale)
        ax_time.set_yscale(choose_scale(time_vals))
        ax_err.set_yscale(choose_scale(err_vals))
        for ax in axes:
            ax.grid(True, which="both", linestyle="--", alpha=0.6)
            ax.legend()
        fig.suptitle(f"Distribution: {dist} (rounds={rounds})", fontsize=14, y=1.02)
        fig.tight_layout()
        outfile = output_dir / f"{dist}_r{rounds}_time_error.png"
        fig.savefig(outfile, dpi=200, bbox_inches="tight")
        plt.close(fig)


def plot_carmichael_rounds(agg):
    """
    Plot error rates on Carmichael numbers at 48 bits for all available round counts.

    Args:
        agg: Aggregated data mapping.
    """
    dist = "carmichael"
    bits = 48
    output_dir = Path(__file__).resolve().parent.parent / "results" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    rounds_values = sorted({key[3] for key in agg if key[0] == dist and key[2] == bits})
    if not rounds_values:
        return

    fig, axes = plt.subplots(1, len(rounds_values), figsize=(4 * len(rounds_values), 4.5), sharey=True)
    if len(rounds_values) == 1:
        axes = [axes]

    # Determine a common y-axis upper bound across subplots.
    overall_max = 0.0
    for rounds in rounds_values:
        for algo in STYLES.keys():
            key = (dist, algo, bits, rounds)
            if key in agg:
                overall_max = max(overall_max, agg[key][1])
    if overall_max == 0.0:
        overall_max = 1.0
    y_upper = overall_max * 1.1

    for ax, rounds in zip(axes, rounds_values):
        labels = []
        values = []
        colors = []
        totals = []
        for algo, (color, _) in STYLES.items():
            key = (dist, algo, bits, rounds)
            if key not in agg:
                continue
            _, mean_err, total_err, total_tests = agg[key]
            labels.append(algo)
            values.append(mean_err)
            colors.append(color)
            totals.append((total_err, total_tests))

        bars = ax.bar(labels, values, color=colors)
        ax.set_title(f"Rounds={rounds}")
        ax.set_ylabel("error rate")
        ax.set_ylim(0, y_upper)
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        if bars:
            ax.bar_label(bars, labels=[f"{int(err)}/{int(total)}" for err, total in totals], padding=3, fontsize=9)

    fig.suptitle(f"Carmichael error rate at {bits} bits", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    outfile = output_dir / f"{dist}_bits{bits}_rounds_bar.png"
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """
    Entry point: load results and generate plots.
    """
    repo_root = Path(__file__).resolve().parent.parent
    csv_path = repo_root / "results" / "raw_results.csv"
    data = load_results(csv_path)
    agg = aggregate(data)

    dists = sorted({key[0] for key in agg})
    rounds_values = sorted({key[3] for key in agg})
    for dist in dists:
        plot_dist(dist, rounds_values, agg)
    plot_carmichael_rounds(agg)
    print("Plots saved to results/plots")


if __name__ == "__main__":
    plt.style.use("seaborn-v0_8-darkgrid")
    main()
