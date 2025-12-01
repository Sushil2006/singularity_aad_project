"""
Benchmark runner for Karger and Stoer-Wagner min-cut algorithms.

This script drives the compiled `mincut_bench` binary across configured algorithms,
graph distributions, and sizes, collecting per-repetition timing and error metrics
into `mincut/results/raw_results.csv`.
"""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# Algorithm and distribution mappings to match the C++ program.
ALGOS: Dict[str, int] = {
    "Karger": 0,
    "StoerWagner": 1,
}

DISTS: Dict[str, int] = {
    "two_cluster_gnp": 0,
    "gnp_sparse": 1,
    "gnp_dense": 2,
    "adversarial_barbell": 3,
}

N_LIST: List[int] = [30, 60, 90]
GRAPHS_PER_REP: int = 20
REPS: int = 5
SEED_BASE_GLOBAL: int = 12345


def compute_seed_base(algo_id: int, dist_id: int, n: int) -> int:
    """
    Derive a deterministic seed base from the configuration tuple.

    Args:
        algo_id (int): Algorithm identifier.
        dist_id (int): Distribution identifier.
        n (int): Graph size.

    Returns:
        int: Seed base for this configuration.
    """
    return SEED_BASE_GLOBAL + algo_id * 1_000_003 + dist_id * 10_007 + n * 101


def run_single_config(
    binary: Path,
    algo_id: int,
    dist_id: int,
    n: int,
    graphs_per_rep: int,
    reps: int,
) -> List[Tuple[int, int]]:
    """
    Execute the benchmark binary once for a given configuration.

    Args:
        binary (Path): Path to the compiled `mincut_bench` binary.
        algo_id (int): Algorithm identifier.
        dist_id (int): Distribution identifier.
        n (int): Graph size.
        graphs_per_rep (int): Number of graphs per repetition.
        reps (int): Number of repetitions to request from the binary.

    Returns:
        List[Tuple[int, int]]: List of (total_time_ns, sum_sq_error) pairs, one per repetition.
    """
    seed_base = compute_seed_base(algo_id, dist_id, n)
    cmd = [
        str(binary),
        str(algo_id),
        str(dist_id),
        str(n),
        str(graphs_per_rep),
        str(seed_base),
        str(reps),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != reps:
        raise RuntimeError(f"Expected {reps} lines from benchmark, got {len(lines)}")

    results: List[Tuple[int, int]] = []
    for line in lines:
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"Malformed output line: '{line}'")
        total_time_ns = int(parts[0])
        sum_sq_error = int(parts[1])
        results.append((total_time_ns, sum_sq_error))
    return results


def ensure_results_dir(results_path: Path) -> None:
    """
    Ensure the results directory exists before writing CSV output.

    Args:
        results_path (Path): Path to the output CSV file.
    """
    results_path.parent.mkdir(parents=True, exist_ok=True)


def write_csv(rows: Iterable[dict], output_path: Path) -> None:
    """
    Write benchmark rows to CSV.

    Args:
        rows (Iterable[dict]): Iterable of rows to write.
        output_path (Path): Destination file path.
    """
    ensure_results_dir(output_path)
    fieldnames = ["algo", "dist", "n", "graphs_per_rep", "rep", "total_time_ns", "sum_sq_error"]
    with output_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    """
    Drive all configured benchmarks and persist results.
    """
    repo_root = Path(__file__).resolve().parent.parent
    binary = repo_root / "mincut_bench"
    output_path = repo_root / "results" / "raw_results.csv"

    all_rows: List[dict] = []
    for algo_name, algo_id in ALGOS.items():
        for dist_name, dist_id in DISTS.items():
            for n in N_LIST:
                results = run_single_config(binary, algo_id, dist_id, n, GRAPHS_PER_REP, REPS)
                for rep_idx, (total_time_ns, sum_sq_error) in enumerate(results):
                    all_rows.append(
                        {
                            "algo": algo_name,
                            "dist": dist_name,
                            "n": n,
                            "graphs_per_rep": GRAPHS_PER_REP,
                            "rep": rep_idx,
                            "total_time_ns": total_time_ns,
                            "sum_sq_error": sum_sq_error,
                        }
                    )
                print(f"Finished algo={algo_name} dist={dist_name} n={n}")

    write_csv(all_rows, output_path)
    print(f"Wrote results to {output_path}")


if __name__ == "__main__":
    main()
