"""
Utility script to preprocess intraday stock CSV data into a single-column text file.

This script reads a CSV file with columns that include `Open` prices and writes the
first `MAX_VALUES` open prices (scaled by 100 and rounded to integers) into a space-
separated text file for use by the C++ benchmarks.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable, List

# Maximum number of values to retain (2^20 as requested).
MAX_VALUES = 1 << 20


def parse_open_price(row: List[str]) -> int | None:
    """
    Extract and scale the Open price from a CSV row.

    Args:
        row (List[str]): Parsed CSV row where the 4th column is expected to be the Open price.

    Returns:
        int | None: Open price scaled by 100 and rounded, or None if parsing fails.
    """
    if len(row) < 4:
        return None
    try:
        # Column order: Instrument,Date,Time,Open,High,Low,Close
        price = float(row[3])
        return int(math.floor(price * 100 + 0.5))
    except ValueError:
        return None


def stream_open_prices(csv_path: Path, limit: int = MAX_VALUES) -> Iterable[int]:
    """
    Stream scaled Open prices from the input CSV until the limit is reached.

    Args:
        csv_path (Path): Path to the input CSV file.
        limit (int): Maximum number of price values to yield.

    Yields:
        int: Scaled Open prices, one per valid row.
    """
    yielded = 0
    with csv_path.open(newline="") as f:
        reader = csv.reader(f)
        # Skip header
        next(reader, None)
        for row in reader:
            if yielded >= limit:
                break
            price = parse_open_price(row)
            if price is None:
                continue
            yielded += 1
            yield price


def write_prices(prices: Iterable[int], output_path: Path) -> int:
    """
    Write prices to the output file as a single space-separated line.

    Args:
        prices (Iterable[int]): Iterable of integer prices to write.
        output_path (Path): Destination file path.

    Returns:
        int: Number of prices written.
    """
    count = 0
    with output_path.open("w") as out:
        for price in prices:
            if count > 0:
                out.write(" ")
            out.write(str(price))
            count += 1
    return count


def main() -> None:
    """
    Entry point: convert `BNF_2010_2020.csv` into `nifty_1m_int_1M.txt` with Open prices.

    The script expects to be run from the repository root so that input/output paths
    resolve relative to `quicksort/data/`.
    """
    data_dir = Path(__file__).resolve().parent.parent / "data"
    input_path = data_dir / "BNF_2010_2020.csv"
    output_path = data_dir / "nifty_1m_int_1M.txt"

    prices = stream_open_prices(input_path, MAX_VALUES)
    written = write_prices(prices, output_path)
    print(f"Wrote {written} prices to {output_path}")


if __name__ == "__main__":
    main()
