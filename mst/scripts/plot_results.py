"""
Plotting helper for MST benchmarks.

Loads `results/raw_results.csv`, aggregates timing statistics, and produces
plots comparing KKT vs Kruskal across densities and graph sizes.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from math import sqrt
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt

ALGOS: List[str] = ["KKT", "Kruskal"]
DENSITIES: List[int] = [2, 4, 8, 16]


def load_rows(csv_path: Path) -> List[dict]:
    """
    Load benchmark rows from the CSV file.
    """
    rows: List[dict] = []
    with csv_path.open("r", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            rows.append(
                {
                    "algo": row["algo"],
                    "n": int(row["n"]),
                    "m": int(row["m"]),
                    "graphs_per_rep": int(row["graphs_per_rep"]),
                    "rep": int(row["rep"]),
                    "total_time_ns": int(row["total_time_ns"]),
                }
            )
    return rows


def aggregate_stats(rows: Iterable[dict]) -> Dict[Tuple[str, int, int], Dict[str, float]]:
    """
    Aggregate per-configuration timing statistics.
    """
    grouped: Dict[Tuple[str, int, int], List[float]] = defaultdict(list)
    for row in rows:
        time_per_graph = row["total_time_ns"] / row["graphs_per_rep"]
        key = (row["algo"], row["n"], row["m"])
        grouped[key].append(time_per_graph)

    stats: Dict[Tuple[str, int, int], Dict[str, float]] = {}
    for key, values in grouped.items():
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        stats[key] = {"mean": mean, "std": sqrt(variance)}
    return stats


def plot_time_vs_n(
    stats: Dict[Tuple[str, int, int], Dict[str, float]],
    densities: List[int],
    n_values: List[int],
    output_dir: Path,
) -> None:
    """
    Plot mean time per graph vs n for each density factor.
    """
    for density in densities:
        fig, ax = plt.subplots()
        for algo in ALGOS:
            xs: List[int] = []
            ys: List[float] = []
            errs: List[float] = []
            for n in sorted(n_values):
                key = (algo, n, density * n)
                if key not in stats:
                    continue
                xs.append(n)
                ys.append(stats[key]["mean"])
                errs.append(stats[key]["std"])
            if xs:
                ax.errorbar(xs, ys, yerr=errs, marker="o", label=algo, capsize=3)
        if not ax.lines:
            plt.close(fig)
            continue
        ax.set_xlabel("n (vertices)")
        ax.set_ylabel("mean time per graph (ns)")
        ax.set_title(f"Time vs n (m = {density}n)")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle="--", linewidth=0.6)
        ax.legend()
        fig.tight_layout()
        output_path = output_dir / f"time_vs_n_m{density}n.png"
        fig.savefig(output_path, dpi=160)
        plt.close(fig)


def plot_time_vs_m(
    stats: Dict[Tuple[str, int, int], Dict[str, float]],
    densities: List[int],
    n_values: List[int],
    output_dir: Path,
) -> None:
    """
    Plot mean time per graph vs m for each fixed n.
    """
    for n in sorted(n_values):
        fig, ax = plt.subplots()
        for algo in ALGOS:
            xs: List[int] = []
            ys: List[float] = []
            errs: List[float] = []
            for density in densities:
                m = density * n
                key = (algo, n, m)
                if key not in stats:
                    continue
                xs.append(m)
                ys.append(stats[key]["mean"])
                errs.append(stats[key]["std"])
            if xs:
                ax.errorbar(xs, ys, yerr=errs, marker="s", label=algo, capsize=3)
        if not ax.lines:
            plt.close(fig)
            continue
        ax.set_xlabel("m (edges)")
        ax.set_ylabel("mean time per graph (ns)")
        ax.set_title(f"Time vs m (n = {n})")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.grid(True, which="both", linestyle="--", linewidth=0.6)
        ax.legend()
        fig.tight_layout()
        output_path = output_dir / f"time_vs_m_n{n}.png"
        fig.savefig(output_path, dpi=160)
        plt.close(fig)


def main() -> None:
    """
    Aggregate results and emit all plots.
    """
    repo_root = Path(__file__).resolve().parent.parent
    csv_path = repo_root / "results" / "raw_results.csv"
    output_dir = repo_root / "results" / "plots"
    if not csv_path.exists():
        raise FileNotFoundError(f"Missing results CSV: {csv_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(csv_path)
    stats = aggregate_stats(rows)
    n_values = sorted({row["n"] for row in rows})

    plot_time_vs_n(stats, DENSITIES, n_values, output_dir)
    plot_time_vs_m(stats, DENSITIES, n_values, output_dir)
    print(f"Wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
