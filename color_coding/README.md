# Randomized color-coding for k-cycles

This module benchmarks a randomized color-coding algorithm against a naive DFS/backtracking baseline for detecting simple k-cycles in undirected graphs sampled from a single planted distribution.

## Build

From `color_coding/`:
```bash
g++ -O2 -std=c++17 src/main.cpp -o kcycle_bench
```

## Graph model
- Distribution: `planted_cycle_noise(n, k, p_noise)`.
- Guarantees: always includes at least one simple k-cycle (the planted one); optional connectivity fix adds random bridges if the sampled graph is disconnected.

## Binary usage
```
./kcycle_bench <algo_id> <n> <k> <p_noise> <graphs_per_rep> <seed_base> <reps>
```
- `algo_id`: `0 = cc_k_cycle` (color-coding + DP), `1 = dfs_k_cycle` (deterministic DFS/backtracking).
- `n`: number of vertices; `k`: target cycle length (3 ≤ k ≤ n); `p_noise`: noise edge probability in [0,1].
- `graphs_per_rep`: graphs per repetition; `seed_base`: 64-bit base seed; `reps`: number of repetitions to print. Per rep, the binary prints one line: `<total_time_ns> <error_count>` where `error_count` counts missed/invalid k-cycles across the generated graphs.

Color-coding uses repetitions `R(k) = ceil(log(1/delta_target) / (k!/k^k))` with `delta_target = 0.1` (capped at 1000), run for exactly `R(k)` colorings per graph.

## Benchmark scripts

### Run all configured benchmarks
Defaults live in `scripts/run_benchmarks.py`:
- `algos = ["cc_k_cycle", "dfs_k_cycle"]`
- `n_list = [30, 60, 90, 120]`
- `k_list = [5, 6, 7, 8]`
- `p_noise_list = [0.05, 0.1, 0.2]`
- `graphs_per_rep = 20`, `reps = 5`, `seed_base_global = 12345`

Run:
```bash
python3 scripts/run_benchmarks.py
```
This writes `results/raw_results.csv` with columns `algo,n,k,p_noise,graphs_per_rep,rep,total_time_ns,error_count`.

### Plotting
```bash
python3 scripts/plot_results.py
```
This reads `results/raw_results.csv` and produces PNGs under `results/plots/`:
- `time_vs_n_k{K}_p{P}.png`: mean time-per-graph vs n for each (k, p_noise).
- `time_vs_k_n{N}_p{P}.png`: mean time-per-graph vs k for each (n, p_noise).
- `error_vs_k_n{N}_p{P}.png`: error rate vs k for color-coding (DFS should be exact in-range).
