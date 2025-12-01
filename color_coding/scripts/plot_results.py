"""
Plot runtime for color-coding k-cycle benchmarks.

Generates one plot per k: mean time-per-graph vs n for both algorithms, with timeouts drawn at the timeout threshold.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt

# Keep timeout consistent with the benchmark runner.
TIMEOUT_SECS = 20


def mean(values: Iterable[float]) -> float:
    vals = list(values)
    return sum(vals) / len(vals) if vals else 0.0


def stddev(values: Iterable[float]) -> float:
    vals = list(values)
    if len(vals) < 2:
        return 0.0
    mu = mean(vals)
    variance = sum((v - mu) ** 2 for v in vals) / len(vals)
    return math.sqrt(variance)


def load_records(csv_path: Path) -> List[dict]:
    records: List[dict] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["algo"] = row["algo"]
            row["n"] = int(row["n"])
            row["k"] = int(row["k"])
            row["graphs_per_rep"] = int(row["graphs_per_rep"])
            row["timeout"] = int(row.get("timeout", 0))
            row["total_time_ns"] = int(row["total_time_ns"])
            records.append(row)
    return records


def aggregate_metrics(records: List[dict]) -> Dict[Tuple[str, int, int], Tuple[float, float, float, float]]:
    """
    Aggregate mean time-per-graph and timeout fraction per (algo, k, n).

    Returns:
        Mapping (algo, k, n) -> (mean_time_ns, std_time_ns, timeout_fraction, timeout_time_ns_per_graph).
    """
    grouped: Dict[Tuple[str, int, int], Dict[str, float]] = {}
    for row in records:
        key = (row["algo"], row["k"], row["n"])
        timeout_per_graph = (TIMEOUT_SECS * 1_000_000_000) / row["graphs_per_rep"]
        if key not in grouped:
            grouped[key] = {
                "times": [],
                "timeouts": 0,
                "total": 0,
                "timeout_time": timeout_per_graph,
            }
        grouped[key]["total"] += 1
        if row["timeout"]:
            grouped[key]["timeouts"] += 1
        else:
            time_per_graph = row["total_time_ns"] / row["graphs_per_rep"]
            grouped[key]["times"].append(time_per_graph)

    metrics: Dict[Tuple[str, int, int], Tuple[float, float, float, float]] = {}
    for key, data in grouped.items():
        times = data["times"]
        timeout_frac = data["timeouts"] / data["total"]
        mean_time = mean(times) if times else float("nan")
        std_time = stddev(times) if times else 0.0
        metrics[key] = (mean_time, std_time, timeout_frac, data["timeout_time"])
    return metrics


def plot_time_vs_n(metrics: Dict[Tuple[str, int, int], Tuple[float, float, float, float]], output_dir: Path) -> None:
    ks = sorted({key[1] for key in metrics})
    for k in ks:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        timeout_label_used = False
        y_values: List[float] = []
        timeout_records: List[Tuple[int, float, float, str]] = []

        for algo in ["cc_k_cycle", "dfs_k_cycle"]:
            points: List[Tuple[int, float]] = []
            for (alg, kk, n), vals in metrics.items():
                if alg != algo or kk != k:
                    continue
                mean_time, _, timeout_frac, timeout_time = vals
                if timeout_frac > 0:
                    timeout_records.append((n, timeout_time, timeout_frac, algo))
                if math.isnan(mean_time):
                    continue
                points.append((n, mean_time))
                y_values.append(mean_time)
            points.sort(key=lambda x: x[0])
            if points:
                ns = [p[0] for p in points]
                times = [p[1] for p in points]
                ax.plot(ns, times, marker="o", linewidth=1.6, label=algo)

        # Set y-limits based on real data; enforce bottom = 0 with minimal headroom.
        if y_values:
            y_max = max(y_values)
            y_max *= 1.02
            if y_max <= 0:
                y_max = 1.0
            ax.set_ylim(bottom=0, top=y_max)
        else:
            ax.set_ylim(bottom=0, top=1.0)

        # Draw timeout markers near the top of the plot without stretching axes.
        current_top = ax.get_ylim()[1]
        y_marker = current_top * 0.95
        for n_val, timeout_time, frac, algo in timeout_records:
            label = "timeout (dfs_k_cycle)" if algo == "dfs_k_cycle" and not timeout_label_used else None
            ax.scatter([n_val], [y_marker], marker="x", color="red", label=label, clip_on=False)
            ax.text(n_val, y_marker, f"{frac:.0%}", ha="center", va="bottom", fontsize=8)
            if label:
                timeout_label_used = True
        ax.set_title(f"Mean time per graph vs n (k={k})")
        ax.set_xlabel("n")
        ax.set_ylabel("mean time per graph (ns)")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend(loc="upper left")
        output_dir.mkdir(parents=True, exist_ok=True)
        outfile = output_dir / f"time_vs_n_k{k}.png"
        fig.tight_layout()
        fig.savefig(outfile, dpi=200)
        plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    csv_path = repo_root / "results" / "raw_results.csv"
    output_dir = repo_root / "results" / "plots"

    records = load_records(csv_path)
    metrics = aggregate_metrics(records)
    plot_time_vs_n(metrics, output_dir)
    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    plt.style.use("seaborn-v0_8-darkgrid")
    main()
