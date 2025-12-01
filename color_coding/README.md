# Randomized color-coding for k-cycles

This module benchmarks a randomized color-coding algorithm against a naive DFS/backtracking baseline for detecting simple k-cycles. Two graph families are available: a planted noisy cycle, and an adversarial layered ladder meant to explode DFS branching.

## Build

From `color_coding/`:
```bash
g++ -O2 -std=c++17 src/main.cpp -o kcycle_bench
```

## Graph models
- `planted_cycle_noise(n, k, p_noise)`: planted k-cycle with Bernoulli noise edges; optional connectivity patch adds random bridges if disconnected.
- `layered_diamond_ladder(n, k)`: complete bipartite layers of width `b = floor((n-1)/(k-1))` between consecutive layers, a single closing edge back to the start; exactly one k-cycle but `b^(k-1)` length-`k-1` paths make DFS explore exponentially many branches. Activate by passing a negative `p_noise` (e.g., `-1`).

## Binary usage
```
./kcycle_bench <algo_id> <n> <k> <p_noise> <graphs_per_rep> <seed_base> <reps>
```
- `algo_id`: `0 = cc_k_cycle` (color-coding + DP), `1 = dfs_k_cycle` (deterministic DFS/backtracking).
- `n`: number of vertices; `k`: target cycle length (3 ≤ k ≤ n).
- `p_noise`: noise edge probability in [0,1] for `planted_cycle_noise`; **set to a negative value to sample the layered diamond ladder**.
- `graphs_per_rep`: graphs per repetition; `seed_base`: 64-bit base seed; `reps`: number of repetitions to print. Per rep, the binary prints one line: `<total_time_ns> <error_count>` where `error_count` counts missed/invalid k-cycles across the generated graphs.

Color-coding uses repetitions `R(k) = ceil(log(1/delta_target) / (k!/k^k))` with `delta_target = 0.1` (capped at 1000), run for exactly `R(k)` colorings per graph. Benchmark invocations are killed after 20 seconds and marked as timeouts in CSV output.

## Benchmark scripts

### Run all configured benchmarks
Defaults live in `scripts/run_benchmarks.py` (currently targeting only the adversarial ladder):
- `algos = ["cc_k_cycle", "dfs_k_cycle"]`
- `n_list = [500]`
- `k_list = [7]`
- `p_noise_list = [-1.0]` (negative = layered ladder)
- `graphs_per_rep = 20`, `reps = 5`, `seed_base_global = 12345`

Run:
```bash
python3 scripts/run_benchmarks.py
```
This writes `results/raw_results.csv` with columns `algo,n,k,p_noise,graphs_per_rep,rep,timeout,total_time_ns,error_count`. When a binary call exceeds the 20s wall-clock limit, all reps for that configuration are recorded with `timeout=1` and sentinel timings.

### Plotting
```bash
python3 scripts/plot_results.py
```
This reads `results/raw_results.csv` and produces one figure per k under `results/plots/`:
- `time_vs_n_k{K}.png`: mean time-per-graph vs n for both algorithms. If a configuration times out, the curve rises to the timeout threshold and the point is marked with a red “x”.
