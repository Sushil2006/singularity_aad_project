"""
Benchmark runner for KKT and Kruskal MST algorithms.

Uses the compiled `mst_bench` binary to cover all configured (algo, n, m)
combinations, streaming per-repetition timings into `results/raw_results.csv`.
"""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List

ALGOS: Dict[str, int] = {
    "KKT": 0,
    "Kruskal": 1,
}

N_LIST: List[int] = [1 << k for k in range(10, 18)]
DENSITIES: List[int] = [2, 4, 8, 16]
GRAPHS_PER_REP: int = 3
REPS: int = 3
SEED_BASE_GLOBAL: int = 424242


def compute_seed_base(algo_id: int, n: int, m: int) -> int:
    """
    Deterministically derive a seed base from the configuration tuple.
    """
    return SEED_BASE_GLOBAL + algo_id * 1_000_003 + n * 101 + m * 10_007


def run_single_config(
    binary: Path,
    algo_id: int,
    n: int,
    m: int,
    graphs_per_rep: int,
    reps: int,
) -> List[int]:
    """
    Execute one benchmark configuration and return per-repetition timings.
    """
    seed_base = compute_seed_base(algo_id, n, m)
    cmd = [
        str(binary),
        str(algo_id),
        str(n),
        str(m),
        str(graphs_per_rep),
        str(seed_base),
        str(reps),
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != reps:
        raise RuntimeError(f"Expected {reps} lines from benchmark, got {len(lines)}")
    times: List[int] = []
    for line in lines:
        parts = line.split()
        if len(parts) != 1:
            raise ValueError(f"Malformed output line: '{line}'")
        times.append(int(parts[0]))
    return times


def write_csv(rows: Iterable[dict], output_path: Path) -> None:
    """
    Stream rows to the CSV output file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["algo", "n", "m", "graphs_per_rep", "rep", "total_time_ns"]
    with output_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    """
    Drive all configured benchmark combinations and persist results.
    """
    repo_root = Path(__file__).resolve().parent.parent
    binary = repo_root / "mst_bench"
    output_path = repo_root / "results" / "raw_results.csv"

    all_rows: List[dict] = []
    for algo_name, algo_id in ALGOS.items():
        for n in N_LIST:
            for density in DENSITIES:
                m = density * n
                times = run_single_config(binary, algo_id, n, m, GRAPHS_PER_REP, REPS)
                for rep_idx, total_time_ns in enumerate(times):
                    all_rows.append(
                        {
                            "algo": algo_name,
                            "n": n,
                            "m": m,
                            "graphs_per_rep": GRAPHS_PER_REP,
                            "rep": rep_idx,
                            "total_time_ns": total_time_ns,
                        }
                    )
                print(f"Finished algo={algo_name} n={n} m={m}")

    write_csv(all_rows, output_path)
    print(f"Wrote results to {output_path}")


if __name__ == "__main__":
    main()
