# MST Benchmarks: KKT vs Kruskal

This module benchmarks the Karger–Klein–Tarjan (KKT) randomized MST algorithm against classic Kruskal with union–find on random connected graphs.

## Build

From `mst/`:

```
g++ -std=c++17 -O3 -march=native -DNDEBUG -o mst_bench src/main.cpp
```

## Binary usage

```
./mst_bench <algo_id> <n> <m> <graphs_per_rep> <seed_base> <reps>
```

- `algo_id`: `0 = KKT`, `1 = Kruskal`.
- `n`: vertices; `m`: edges (must satisfy `n-1 <= m <= n(n-1)/2`).
- `graphs_per_rep`: number of independent random graphs per repetition.
- `seed_base`: 64-bit RNG base seed (per repetition we add `rep`).
- `reps`: number of repetitions; the binary prints one timing line per repetition (`<total_time_ns>`).

The generator first builds a random spanning tree, then adds random extra edges until the edge count reaches `m`; weights are uniform integers in `[1, 1e6]`.

## Scripts

- `scripts/run_benchmarks.py`  
  Drives all requested configurations and writes `results/raw_results.csv`. Defaults:
  - Algorithms: `KKT`, `Kruskal`.
  - Sizes: `n = 2^k` for `k = 10..17`.
  - Densities: `m ∈ {2n, 4n, 8n, 16n}`.
  - `graphs_per_rep = 3`, `reps = 3`, deterministic seeds from config.

- `scripts/plot_results.py`  
  Reads the CSV and generates plots in `results/plots/`:
  - Time per graph vs `n` for each density factor.
  - Time per graph vs `m` for each fixed `n`.

## Results layout

- `results/raw_results.csv`: appended by the benchmark runner.
- `results/plots/*.png`: figures produced by the plotting script.
