# Quicksort benchmarks

This module benchmarks three QuickSort variants, Merge Sort, and `std::sort` on multiple integer distributions.

## Preparing data

1) Preprocess the provided intraday CSV to extract Open prices (scaled by 100 and rounded) into a single text file (max 2^20 values, file currently contains 1,007,339 values):

```bash
python3 scripts/prepare_stock_data.py
```

The output is `data/nifty_1m_int_1M.txt`.

## Build

```bash
g++ -O2 -std=c++17 src/main.cpp -o quicksort_bench
```

## Running the benchmark binary directly

```
./quicksort_bench <algo_id> <dist_id> <n> <seed_base> <reps> <stock_path>
```

Mappings:
- `algo_id`: `0=QS`, `1=QS_small` (tail recursion), `2=QS_cut16` (tail recursion + cutoff), `3=MS`, `4=STD`.
- `dist_id`: `0=sorted`, `1=almost_sorted`, `2=uniform`, `3=normal`, `4=stock`.

The program prints one line per repetition: `<time_ns> <swaps>`. For `std::sort`, swaps is set to `-1`.

## Running all benchmarks

`scripts/run_benchmarks.py` drives every combination of algorithm, distribution, and size, writing results to `results/raw_results.csv`.

Defaults:
- `sizes = [2^10, 2^11, ..., 2^20]`
- `reps = 5`
- Seed bases derived deterministically per configuration.
- Stock runs include an extra size equal to the full available dataset (currently 1,007,339 values) and skip only if a requested `n` exceeds availability.

Example:
```bash
python3 scripts/run_benchmarks.py
```

## Plotting results

Generate time/swaps plots (two subplots per distribution) into `results/plots/`:
```bash
python3 scripts/plot_results.py
```

## Notes

- QuickSort implementations are in-place and instrumented with a global swap counter (`g_swap_count`).
- Timing measures only the algorithm body per repetition (data generation and setup are excluded).
- Set `kVerifySorted` in `src/main.cpp` to `true` if you want per-run output validation during debugging.
