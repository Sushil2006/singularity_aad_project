"""
Prepare a small list of Carmichael numbers for benchmarking.

The script writes one integer per line to `miller_rabin/data/carmichael_odd.txt`.
Numbers are sourced from well-known Carmichael sequences (OEIS A002997) and cover
bit-lengths from ~10 bits up to ~50 bits to stress-test Fermat vs Miller–Rabin.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List


def get_carmichael_numbers() -> List[int]:
    """
    Return a curated list of Carmichael numbers.

    Returns:
        List[int]: Carmichael numbers (sorted ascending).
    """
    small = [
        561,
        1105,
        1729,
        2465,
        2821,
        6601,
        8911,
        10585,
        15841,
        29341,
        41041,
        46657,
        52633,
        62745,
        63973,
        75361,
        101101,
        115921,
        126217,
        162401,
        172081,
        188461,
        252601,
        278545,
        294409,
        314821,
        334153,
        340561,
        399001,
        410041,
        449065,
        488881,
        512461,
        530881,
        552721,
        656601,
        658801,
        670033,
        748657,
        825265,
        838201,
        852841,
        997633,
        1024651,
        1033669,
        1050985,
        1058197,
        1082809,
        1152271,
        1193221,
        1461241,
        1543537,
        1622473,
        1723681,
        1887001,
        1909001,
        2100901,
        2113921,
    ]

    larger = [
        3215031751,
        5394826801,
        232250619601,
        9746347772161,
        1436697831295441,
    ]

    return sorted(small + larger)


def write_numbers(nums: Iterable[int], path: Path) -> None:
    """
    Write integers to the target file, one per line.

    Args:
        nums (Iterable[int]): Sequence of Carmichael numbers.
        path (Path): Destination file path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for n in nums:
            f.write(f"{n}\n")


def main() -> None:
    """
    Entry point for writing the Carmichael list.
    """
    repo_root = Path(__file__).resolve().parent.parent
    out_path = repo_root / "data" / "carmichael_odd.txt"
    write_numbers(get_carmichael_numbers(), out_path)
    print(f"Wrote Carmichael numbers to {out_path}")


if __name__ == "__main__":
    main()
