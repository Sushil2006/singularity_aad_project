"""
Benchmark runner for QuickSort variants, Merge Sort, and std::sort.

This script drives the compiled `quicksort_bench` binary across configured algorithms,
distributions, input sizes, and repetitions, collecting per-run timing and swap counts
into `quicksort/results/raw_results.csv`.
"""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

# Algorithm and distribution mappings to match the C++ program.
ALGOS: Dict[str, int] = {
    "QS": 0,
    "QS_small": 1,
    "QS_cut16": 2,
    "MS": 3,
    "STD": 4,
}

DISTS: Dict[str, int] = {
    "sorted": 0,
    "almost_sorted": 1,
    "uniform": 2,
    "normal": 3,
    "stock": 4,
}

SIZES: List[int] = [2**i for i in range(10, 21)]
REPS_DEFAULT: int = 5
SEED_BASE_GLOBAL: int = 42


def compute_seed_base(algo_id: int, dist_id: int, n: int) -> int:
    """
    Derive a deterministic seed base from the configuration tuple.

    Args:
        algo_id (int): Algorithm identifier.
        dist_id (int): Distribution identifier.
        n (int): Problem size.

    Returns:
        int: Seed base for this configuration.
    """
    return SEED_BASE_GLOBAL + algo_id * 1_000_003 + dist_id * 10_007 + n


def run_single_config(
    binary: Path,
    algo_id: int,
    dist_id: int,
    n: int,
    reps: int,
    stock_path: Path,
) -> List[Tuple[int, int]]:
    """
    Execute the benchmark binary once for a given configuration.

    Args:
        binary (Path): Path to the compiled `quicksort_bench` binary.
        algo_id (int): Algorithm identifier.
        dist_id (int): Distribution identifier.
        n (int): Problem size.
        reps (int): Number of repetitions to request from the binary.
        stock_path (Path): Path to the stock data file (used only for dist_id 4).

    Returns:
        List[Tuple[int, int]]: List of (time_ns, swaps) pairs, one per repetition.
    """
    seed_base = compute_seed_base(algo_id, dist_id, n)
    cmd = [
        str(binary),
        str(algo_id),
        str(dist_id),
        str(n),
        str(seed_base),
        str(reps),
        str(stock_path),
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
        time_ns = int(parts[0])
        swaps = int(parts[1])
        results.append((time_ns, swaps))
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
    fieldnames = ["algo", "dist", "n", "rep", "time_ns", "swaps"]
    with output_path.open("w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def count_stock_values(stock_path: Path) -> int:
    """
    Count how many integers are available in the preprocessed stock file.

    Args:
        stock_path (Path): Path to the stock data file.

    Returns:
        int: Number of integer tokens in the file.
    """
    count = 0
    with stock_path.open() as f:
        for token in f.read().split():
            if token:
                count += 1
    return count


def main() -> None:
    """
    Drive all configured benchmarks and persist results.
    """
    repo_root = Path(__file__).resolve().parent.parent
    binary = repo_root / "quicksort_bench"
    stock_path = repo_root / "data" / "nifty_1m_int_1M.txt"
    output_path = repo_root / "results" / "raw_results.csv"
    stock_capacity = count_stock_values(stock_path)

    all_rows: List[dict] = []
    for algo_name, algo_id in ALGOS.items():
        for dist_name, dist_id in DISTS.items():
            sizes_for_dist = SIZES
            if dist_name == "stock":
                # Ensure we include one run on the full available dataset.
                sizes_for_dist = sorted(set(SIZES + [stock_capacity]))
            for n in sizes_for_dist:
                if dist_name == "stock" and n > stock_capacity:
                    print(f"Skipping stock distribution for n={n} (only {stock_capacity} values available)")
                    continue
                results = run_single_config(binary, algo_id, dist_id, n, REPS_DEFAULT, stock_path)
                for rep_idx, (time_ns, swaps) in enumerate(results):
                    all_rows.append(
                        {
                            "algo": algo_name,
                            "dist": dist_name,
                            "n": n,
                            "rep": rep_idx,
                            "time_ns": time_ns,
                            "swaps": swaps,
                        }
                    )
                print(f"Finished algo={algo_name} dist={dist_name} n={n}")

    write_csv(all_rows, output_path)
    print(f"Wrote results to {output_path}")


if __name__ == "__main__":
    main()
