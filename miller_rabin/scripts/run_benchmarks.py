"""
Benchmark runner for Miller–Rabin project.

This script drives `mr_bench` across algorithms, distributions, bit-lengths, and
round counts, writing aggregated rows to `miller_rabin/results/raw_results.csv`.
Execution parameters below are tuned for heavy repetition counts per configuration.
"""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List

# Configurable parameters.
SAMPLE_COUNT = 1
REPS_DEFAULT = 100_000  # High repetition count per configuration.
TD_REPS_48 = 100  # Reduced reps for TD at 48 bits to avoid excessive runtime.
ROUNDS_LIST = [1, 5, 10, 10000]

ALGOS: Dict[str, int] = {"TD": 0, "Fermat": 1, "MR": 2}
DISTS: Dict[str, int] = {"rand_odd": 0, "carmichael": 1, "comp_small_factor": 2, "primes": 3}

BITS_PER_ALGO = {
    "TD": [16, 32, 48],
    "Fermat": [16, 32, 48, 64],
    "MR": [16, 32, 48, 64],
}

SEED_BASE_GLOBAL = 12345


def seed_base(algo_id: int, dist_id: int, bits: int, rounds: int) -> int:
    """
    Derive a deterministic seed base for reproducibility.

    Args:
        algo_id (int): Algorithm identifier.
        dist_id (int): Distribution identifier.
        bits (int): Bit-length.
        rounds (int): Rounds parameter for Fermat/MR (ignored by TD but still set).

    Returns:
        int: Seed base value.
    """
    return SEED_BASE_GLOBAL + algo_id * 1_000_003 + dist_id * 10_007 + bits * 101 + rounds * 17


def ensure_results_dir(path: Path) -> None:
    """
    Ensure the target directory exists.

    Args:
        path (Path): Output CSV path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)


def write_header(csvfile) -> csv.DictWriter:
    """
    Write CSV header and return the writer.

    Args:
        csvfile: Open file object.

    Returns:
        csv.DictWriter: Writer positioned after header.
    """
    fieldnames = ["algo", "dist", "bits", "rounds", "sample_count", "rep", "time_ns_total", "error_count"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    return writer


def run_single_and_stream(
    binary: Path,
    algo_id: int,
    dist_id: int,
    bits: int,
    rounds: int,
    sample_count: int,
    reps: int,
    writer: csv.DictWriter,
) -> None:
    """
    Execute one benchmark configuration and stream rows directly to CSV.

    Args:
        binary (Path): Path to mr_bench.
        algo_id (int): Algorithm identifier.
        dist_id (int): Distribution identifier.
        bits (int): Bit-length.
        rounds (int): Rounds for Fermat/MR.
        sample_count (int): Numbers per repetition.
        reps (int): Number of repetitions.
        writer (csv.DictWriter): CSV writer with header already written.
    """
    sb = seed_base(algo_id, dist_id, bits, rounds)
    cmd = [
        str(binary),
        str(algo_id),
        str(dist_id),
        str(bits),
        str(sample_count),
        str(rounds),
        str(sb),
        str(reps),
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, bufsize=1)
    if proc.stdout is None:
        raise RuntimeError("Failed to capture stdout")
    rep_idx = 0
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 2:
            raise ValueError(f"Bad line: '{line}'")
        time_ns, errors = int(parts[0]), int(parts[1])
        writer.writerow(
            {
                "algo": [name for name, idx in ALGOS.items() if idx == algo_id][0],
                "dist": [name for name, idx in DISTS.items() if idx == dist_id][0],
                "bits": bits,
                "rounds": rounds,
                "sample_count": sample_count,
                "rep": rep_idx,
                "time_ns_total": time_ns,
                "error_count": errors,
            }
        )
        rep_idx += 1
    proc.wait()
    if proc.returncode != 0:
        raise RuntimeError(f"Benchmark subprocess failed with code {proc.returncode}")
    if rep_idx != reps:
        raise RuntimeError(f"Expected {reps} lines, got {rep_idx}")


def main() -> None:
    """
    Drive all benchmark combinations subject to a wall-clock budget.
    """
    repo_root = Path(__file__).resolve().parent.parent
    binary = repo_root / "mr_bench"
    output_path = repo_root / "results" / "raw_results.csv"

    ensure_results_dir(output_path)
    with output_path.open("w", newline="") as csvfile:
        writer = write_header(csvfile)
        for algo_name, algo_id in ALGOS.items():
            for dist_name, dist_id in DISTS.items():
                bits_list = BITS_PER_ALGO[algo_name]
                for bits in bits_list:
                    for rounds in ROUNDS_LIST:
                        reps = REPS_DEFAULT
                        if algo_name == "TD" and bits == 48:
                            reps = TD_REPS_48
                        run_single_and_stream(binary, algo_id, dist_id, bits, rounds, SAMPLE_COUNT, reps, writer)
                        print(f"Finished algo={algo_name} dist={dist_name} bits={bits} rounds={rounds} reps={reps}")

    print(f"Wrote results to {output_path}")


if __name__ == "__main__":
    main()
