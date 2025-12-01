"""
Benchmark runner for randomized color-coding vs DFS k-cycle detection.

This script drives the compiled `kcycle_bench` binary across configured
algorithms, graph sizes, and cycle lengths. A negative `p_noise` selects the
layered diamond ladder distribution; this runner is currently configured to
exercise only that adversarial case. Each binary invocation is capped at
`TIMEOUT_SECS` and marked as a timeout if exceeded.
"""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

# Algorithm identifiers matching the C++ binary.
ALGOS: Dict[str, int] = {
    "cc_k_cycle": 0,
    "dfs_k_cycle": 1,
}

N_LIST: List[int] = [20,40,80,200,300,500]
K_LIST: List[int] = [5,6,7]
# Sentinel p_noise = -1 selects the layered diamond ladder distribution.
P_NOISE_LIST: List[float] = [-1.0]
GRAPHS_PER_REP: int = 20
REPS: int = 5
SEED_BASE_GLOBAL: int = 12345
TIMEOUT_SECS: int = 20


def compute_seed_base(algo_id: int, n: int, k: int, p_noise: float) -> int:
    """
    Derive a deterministic seed base from configuration parameters.

    Args:
        algo_id (int): Algorithm identifier.
        n (int): Number of vertices.
        k (int): Cycle length.
        p_noise (float): Noise edge probability.

    Returns:
        int: Seed base for this configuration.
    """
    scaled_p = int(round(p_noise * 10_000))
    return SEED_BASE_GLOBAL + algo_id * 1_000_003 + n * 101 + k * 10_007 + scaled_p


def run_single_config(
    binary: Path,
    algo_id: int,
    n: int,
    k: int,
    p_noise: float,
    graphs_per_rep: int,
    reps: int,
) -> List[Tuple[int, int, int]]:
    """
    Execute the benchmark binary once for a configuration tuple.

    Args:
        binary (Path): Path to the compiled `kcycle_bench` binary.
        algo_id (int): Algorithm identifier.
        n (int): Number of vertices.
        k (int): Cycle length.
        p_noise (float): Noise probability.
        graphs_per_rep (int): Graphs per repetition.
        reps (int): Number of repetitions requested from the binary.

    Returns:
        List[Tuple[int, int, int]]: (total_time_ns, error_count, timeout_flag) per repetition.
    """
    seed_base = compute_seed_base(algo_id, n, k, p_noise)
    cmd = [
        str(binary),
        str(algo_id),
        str(n),
        str(k),
        str(p_noise),
        str(graphs_per_rep),
        str(seed_base),
        str(reps),
    ]
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
            timeout=TIMEOUT_SECS,
        )
        lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        if len(lines) != reps:
            raise RuntimeError(f"Expected {reps} lines from benchmark, got {len(lines)}")

        results: List[Tuple[int, int, int]] = []
        for line in lines:
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Malformed output line: '{line}'")
            total_time_ns = int(parts[0])
            error_count = int(parts[1])
            results.append((total_time_ns, error_count, 0))
        return results
    except subprocess.TimeoutExpired:
        # Mark all reps as timed out; use sentinel -1 for timing/error fields.
        return [(-1, -1, 1) for _ in range(reps)]


def ensure_results_dir(results_path: Path) -> None:
    """
    Ensure the results directory exists before writing CSV output.

    Args:
        results_path (Path): Path to the output CSV file.
    """
    results_path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """
    Drive all configured benchmarks and persist rows to CSV as they are produced.
    """
    repo_root = Path(__file__).resolve().parent.parent
    binary = repo_root / "kcycle_bench"
    output_path = repo_root / "results" / "raw_results.csv"

    ensure_results_dir(output_path)
    fieldnames = [
        "algo",
        "n",
        "k",
        "p_noise",
        "graphs_per_rep",
        "rep",
        "timeout",
        "total_time_ns",
        "error_count",
    ]

    with output_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for algo_name, algo_id in ALGOS.items():
            for n in N_LIST:
                for k in K_LIST:
                    for p_noise in P_NOISE_LIST:
                        results = run_single_config(
                            binary,
                            algo_id,
                            n,
                            k,
                            p_noise,
                            GRAPHS_PER_REP,
                            REPS,
                        )
                        any_timeout = any(t == 1 for _, _, t in results)
                        for rep_idx, (total_time_ns, error_count, timeout_flag) in enumerate(results):
                            writer.writerow(
                                {
                                    "algo": algo_name,
                                    "n": n,
                                    "k": k,
                                    "p_noise": p_noise,
                                    "graphs_per_rep": GRAPHS_PER_REP,
                                    "rep": rep_idx,
                                    "timeout": timeout_flag,
                                    "total_time_ns": total_time_ns,
                                    "error_count": error_count,
                                }
                            )
                        status_note = " (timeout)" if any_timeout else ""
                        print(f"Finished algo={algo_name} n={n} k={k} p_noise={p_noise}{status_note}")

    print(f"Wrote results to {output_path}")


if __name__ == "__main__":
    main()
