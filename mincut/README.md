# Minimum cut benchmarks

This module benchmarks repeated Karger contractions against the deterministic Stoer-Wagner min-cut algorithm on a few small random graph families.

## Build

From the `mincut/` directory:
```bash
g++ -O2 -std=c++17 src/main.cpp -o mincut_bench
```

## Algorithm overview

- `algo_id = 0`: Repeated Karger with target failure probability `delta_target = 0.1`. Repetitions use  
  `R(n) = max(1, ceil(0.5 * n * (n - 1) * log(1 / delta_target)))`.
- `algo_id = 1`: Stoer-Wagner (exact min-cut).

Graph generators (all regenerate until connected):
- `dist_id = 0` (`two_cluster_gnp`): planted bisection, `p_in = 0.3`, `p_out = 0.05` (even `n` only).
- `dist_id = 1` (`gnp_sparse`): Erdos-Renyi with `p = min(1, c / n)`, `c = 6.0`.
- `dist_id = 2` (`gnp_dense`): Erdos-Renyi with `p = 0.2`.
- `dist_id = 3` (`adversarial_barbell`): two cliques of sizes `⌊n/2⌋` and `⌈n/2⌉` joined by a single bridge edge (unique min-cut = 1).

## Benchmark binary usage

```
./mincut_bench <algo_id> <dist_id> <n> <graphs_per_rep> <seed_base> <reps>
```

Per repetition, the program prints one line:
```
<total_time_ns> <sum_sq_error>
```
`total_time_ns` sums only the chosen algorithm's runtime across `graphs_per_rep` graphs. `sum_sq_error` accumulates `(lambda_hat - lambda_truth)^2` using Stoer-Wagner as the oracle.

## Run all configured benchmarks

Defaults in `scripts/run_benchmarks.py`:
- `algos = ["Karger", "StoerWagner"]`
- `dists = ["two_cluster_gnp", "gnp_sparse", "gnp_dense", "adversarial_barbell"]`
- `n_list = [30, 60, 90]`
- `graphs_per_rep = 20`
- `reps = 5`
- `seed_base_global = 12345`

Run:
```bash
python3 scripts/run_benchmarks.py
```
This writes `results/raw_results.csv` with columns `algo, dist, n, graphs_per_rep, rep, total_time_ns, sum_sq_error`.

## Plotting

Generate time-per-graph and MSE curves (one figure per distribution) into `results/plots/`:
```bash
python3 scripts/plot_results.py
```
