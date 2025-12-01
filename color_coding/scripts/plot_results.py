"""
Plot runtime and error metrics for color-coding k-cycle benchmarks.

Generates:
- time_vs_n_k{K}_p{P}.png: mean time-per-graph vs n for fixed (k, p_noise), both algorithms.
- time_vs_k_n{N}_p{P}.png: mean time-per-graph vs k for fixed (n, p_noise), both algorithms.
- error_vs_k_n{N}_p{P}.png: mean error rate vs k for color-coding.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt


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
    Compute population standard deviation.

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


def load_records(csv_path: Path) -> List[dict]:
    """
    Load benchmark rows from CSV.

    Args:
        csv_path (Path): Path to raw_results.csv.

    Returns:
        List[dict]: Parsed records.
    """
    records: List[dict] = []
    with csv_path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["n"] = int(row["n"])
            row["k"] = int(row["k"])
            row["p_noise"] = float(row["p_noise"])
            row["graphs_per_rep"] = int(row["graphs_per_rep"])
            row["rep"] = int(row["rep"])
            row["total_time_ns"] = int(row["total_time_ns"])
            row["error_count"] = int(row["error_count"])
            records.append(row)
    return records


def aggregate_metrics(records: List[dict]) -> Dict[Tuple[str, int, int, float], Tuple[float, float, float]]:
    """
    Aggregate mean time-per-graph, stddev, and mean error rate per configuration.

    Args:
        records (List[dict]): Raw rows.

    Returns:
        Dict[Tuple[str, int, int, float], Tuple[float, float, float]]:
            Mapping (algo, n, k, p_noise) -> (mean_time_ns, std_time_ns, mean_error_rate).
    """
    grouped: DefaultDict[Tuple[str, int, int, float], List[Tuple[float, float]]] = defaultdict(list)
    for row in records:
        key = (row["algo"], row["n"], row["k"], row["p_noise"])
        time_per_graph = row["total_time_ns"] / row["graphs_per_rep"]
        error_rate = row["error_count"] / row["graphs_per_rep"]
        grouped[key].append((time_per_graph, error_rate))

    metrics: Dict[Tuple[str, int, int, float], Tuple[float, float, float]] = {}
    for key, entries in grouped.items():
        times = [t for t, _ in entries]
        errors = [e for _, e in entries]
        metrics[key] = (mean(times), stddev(times), mean(errors))
    return metrics


def unique_values(metrics: Dict[Tuple[str, int, int, float], Tuple[float, float, float]]):
    """
    Extract sorted unique n, k, and p_noise values present in metrics.

    Args:
        metrics: Aggregated metrics mapping.

    Returns:
        Tuple[List[int], List[int], List[float]]: Unique ns, ks, and p values.
    """
    ns = sorted({key[1] for key in metrics})
    ks = sorted({key[2] for key in metrics})
    ps = sorted({key[3] for key in metrics})
    return ns, ks, ps


def format_p(p_noise: float) -> str:
    """
    Create a file-friendly string for p_noise.

    Args:
        p_noise (float): Noise probability.

    Returns:
        str: Sanitized string such as "0p05" for 0.05.
    """
    return str(p_noise).replace(".", "p")


def plot_time_vs_n(
    metrics: Dict[Tuple[str, int, int, float], Tuple[float, float, float]],
    ks: List[int],
    ps: List[float],
    output_dir: Path,
) -> None:
    """
    Plot mean time-per-graph vs n for each (k, p_noise) pair.

    Args:
        metrics: Aggregated metrics mapping.
        ks: List of k values present.
        ps: List of p_noise values present.
        output_dir: Directory for output PNGs.
    """
    for k in ks:
        for p in ps:
            fig, ax = plt.subplots(figsize=(6.5, 4))
            for algo in ["cc_k_cycle", "dfs_k_cycle"]:
                points = []
                for key, vals in metrics.items():
                    if key[0] == algo and key[2] == k and math.isclose(key[3], p):
                        points.append((key[1], vals[0], vals[1]))
                if not points:
                    continue
                points.sort(key=lambda x: x[0])
                ns = [pt[0] for pt in points]
                mean_times = [pt[1] for pt in points]
                std_times = [pt[2] for pt in points]
                ax.errorbar(ns, mean_times, yerr=std_times, label=algo, marker="o", linewidth=1.6, capsize=3)
            ax.set_title(f"time vs n (k={k}, p_noise={p})")
            ax.set_xlabel("n")
            ax.set_ylabel("mean time per graph (ns)")
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend()
            output_dir.mkdir(parents=True, exist_ok=True)
            outfile = output_dir / f"time_vs_n_k{k}_p{format_p(p)}.png"
            fig.tight_layout()
            fig.savefig(outfile, dpi=200)
            plt.close(fig)


def plot_time_vs_k(
    metrics: Dict[Tuple[str, int, int, float], Tuple[float, float, float]],
    ns: List[int],
    ps: List[float],
    output_dir: Path,
) -> None:
    """
    Plot mean time-per-graph vs k for each (n, p_noise) pair.

    Args:
        metrics: Aggregated metrics mapping.
        ns: List of n values present.
        ps: List of p_noise values present.
        output_dir: Directory for output PNGs.
    """
    for n in ns:
        for p in ps:
            fig, ax = plt.subplots(figsize=(6.5, 4))
            for algo in ["cc_k_cycle", "dfs_k_cycle"]:
                points = []
                for key, vals in metrics.items():
                    if key[0] == algo and key[1] == n and math.isclose(key[3], p):
                        points.append((key[2], vals[0], vals[1]))
                if not points:
                    continue
                points.sort(key=lambda x: x[0])
                ks = [pt[0] for pt in points]
                mean_times = [pt[1] for pt in points]
                std_times = [pt[2] for pt in points]
                ax.errorbar(ks, mean_times, yerr=std_times, label=algo, marker="o", linewidth=1.6, capsize=3)
            ax.set_title(f"time vs k (n={n}, p_noise={p})")
            ax.set_xlabel("k")
            ax.set_ylabel("mean time per graph (ns)")
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend()
            output_dir.mkdir(parents=True, exist_ok=True)
            outfile = output_dir / f"time_vs_k_n{n}_p{format_p(p)}.png"
            fig.tight_layout()
            fig.savefig(outfile, dpi=200)
            plt.close(fig)


def plot_error_vs_k(
    metrics: Dict[Tuple[str, int, int, float], Tuple[float, float, float]],
    ns: List[int],
    ps: List[float],
    output_dir: Path,
) -> None:
    """
    Plot mean error rate vs k for color-coding across (n, p_noise) pairs.

    Args:
        metrics: Aggregated metrics mapping.
        ns: List of n values present.
        ps: List of p_noise values present.
        output_dir: Directory for output PNGs.
    """
    for n in ns:
        for p in ps:
            points = []
            for key, vals in metrics.items():
                if key[0] == "cc_k_cycle" and key[1] == n and math.isclose(key[3], p):
                    points.append((key[2], vals[2]))
            if not points:
                continue
            points.sort(key=lambda x: x[0])
            ks = [pt[0] for pt in points]
            errors = [pt[1] for pt in points]
            fig, ax = plt.subplots(figsize=(6.5, 4))
            ax.plot(ks, errors, marker="o", linewidth=1.6, label="cc_k_cycle")
            ax.set_title(f"error rate vs k (n={n}, p_noise={p})")
            ax.set_xlabel("k")
            ax.set_ylabel("mean error rate")
            ax.grid(True, linestyle="--", alpha=0.6)
            ax.legend()
            output_dir.mkdir(parents=True, exist_ok=True)
            outfile = output_dir / f"error_vs_k_n{n}_p{format_p(p)}.png"
            fig.tight_layout()
            fig.savefig(outfile, dpi=200)
            plt.close(fig)


def main() -> None:
    """
    Entry point: load CSV, aggregate metrics, and render plots.
    """
    repo_root = Path(__file__).resolve().parent.parent
    csv_path = repo_root / "results" / "raw_results.csv"
    output_dir = repo_root / "results" / "plots"

    records = load_records(csv_path)
    metrics = aggregate_metrics(records)
    ns, ks, ps = unique_values(metrics)

    plot_time_vs_n(metrics, ks, ps, output_dir)
    plot_time_vs_k(metrics, ns, ps, output_dir)
    plot_error_vs_k(metrics, ns, ps, output_dir)
    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    plt.style.use("seaborn-v0_8-darkgrid")
    main()
