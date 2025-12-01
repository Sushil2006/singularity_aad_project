#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_set>
#include <vector>

/**
 * Simple undirected graph represented by adjacency lists.
 */
struct Graph {
    int n = 0;
    std::vector<std::vector<int>> adj;
    std::vector<std::unordered_set<int>> adj_set;
};

/**
 * Timer utility to measure elapsed wall-clock time in nanoseconds.
 */
class Timer {
  public:
    /**
     * Start or restart the timer.
     */
    void start() { start_time_ = std::chrono::steady_clock::now(); }

    /**
     * Return elapsed nanoseconds since the last start.
     */
    long long elapsed_ns() const {
        auto end_time = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time_).count();
    }

  private:
    std::chrono::steady_clock::time_point start_time_{};
};

/**
 * Check for adjacency between two vertices.
 *
 * @param g Graph instance.
 * @param u First vertex.
 * @param v Second vertex.
 * @return true if edge (u, v) exists.
 */
bool has_edge(const Graph &g, int u, int v) {
    if (!g.adj_set.empty()) {
        return g.adj_set[u].find(v) != g.adj_set[u].end();
    }
    // Fallback to linear scan if sets are absent.
    return std::find(g.adj[u].begin(), g.adj[u].end(), v) != g.adj[u].end();
}

/**
 * Compute connected components via BFS.
 *
 * @param g Input graph.
 * @return Components as vectors of vertex indices.
 */
std::vector<std::vector<int>> connected_components(const Graph &g) {
    std::vector<char> visited(g.n, 0);
    std::vector<std::vector<int>> comps;
    std::vector<int> queue;
    for (int start = 0; start < g.n; ++start) {
        if (visited[start]) {
            continue;
        }
        queue.clear();
        queue.push_back(start);
        visited[start] = 1;
        for (size_t idx = 0; idx < queue.size(); ++idx) {
            int v = queue[idx];
            for (int nei : g.adj[v]) {
                if (!visited[nei]) {
                    visited[nei] = 1;
                    queue.push_back(nei);
                }
            }
        }
        comps.push_back(queue);
    }
    return comps;
}

/**
 * Sample a graph from the planted_cycle_noise distribution.
 *
 * Steps:
 *  - Pick k distinct vertices uniformly and add a planted k-cycle.
 *  - Add noise edges between all other pairs with probability p_noise.
 *  - Optionally connect remaining components with random bridges.
 *
 * @param n Total number of vertices.
 * @param k Length of the planted simple cycle (3 <= k <= n).
 * @param p_noise Probability of inserting a non-cycle edge.
 * @param rng Random generator.
 * @return Generated graph instance.
 */
Graph planted_cycle_noise(int n, int k, double p_noise, std::mt19937_64 &rng) {
    constexpr int kEdgeCap = 10000;
    if (k < 3 || k > n) {
        throw std::invalid_argument("k must satisfy 3 <= k <= n");
    }
    if (k > kEdgeCap) {
        throw std::invalid_argument("edge cap exceeded while planting cycle");
    }
    Graph g;
    g.n = n;
    g.adj.assign(n, {});
    g.adj_set.assign(n, {});

    int edge_count = 0;
    auto add_edge = [&](int u, int v) -> bool {
        if (u == v || edge_count >= kEdgeCap) {
            return false;
        }
        if (g.adj_set[u].find(v) != g.adj_set[u].end()) {
            return false;
        }
        g.adj_set[u].insert(v);
        g.adj_set[v].insert(u);
        g.adj[u].push_back(v);
        g.adj[v].push_back(u);
        ++edge_count;
        return true;
    };

    // Choose cycle vertices uniformly at random.
    std::vector<int> vertices(n);
    std::iota(vertices.begin(), vertices.end(), 0);
    std::shuffle(vertices.begin(), vertices.end(), rng);
    std::vector<int> cycle_vertices(vertices.begin(), vertices.begin() + k);

    for (int i = 0; i < k; ++i) {
        int u = cycle_vertices[i];
        int v = cycle_vertices[(i + 1) % k];
        if (!add_edge(u, v)) {
            throw std::runtime_error("Failed to insert planted cycle edge before hitting cap");
        }
    }

    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (int u = 0; u < n; ++u) {
        if (edge_count >= kEdgeCap) {
            break;
        }
        for (int v = u + 1; v < n; ++v) {
            if (edge_count >= kEdgeCap) {
                break;
            }
            if (g.adj_set[u].find(v) != g.adj_set[u].end()) {
                continue;
            }
            if (dist(rng) < p_noise) {
                add_edge(u, v);
            }
        }
    }

    // Connectivity patch: if disconnected, add random bridges between components.
    auto comps = connected_components(g);
    if (comps.size() > 1 && edge_count < kEdgeCap) {
        std::shuffle(comps.begin(), comps.end(), rng);
        for (size_t i = 0; i + 1 < comps.size(); ++i) {
            if (edge_count >= kEdgeCap) {
                break;
            }
            const auto &a = comps[i];
            const auto &b = comps[i + 1];
            std::uniform_int_distribution<int> dist_a(0, static_cast<int>(a.size()) - 1);
            std::uniform_int_distribution<int> dist_b(0, static_cast<int>(b.size()) - 1);
            // Limit retries to keep runtime predictable.
            for (int attempt = 0; attempt < 10 && edge_count < kEdgeCap; ++attempt) {
                int u = a[dist_a(rng)];
                int v = b[dist_b(rng)];
                if (add_edge(u, v)) {
                    break;
                }
            }
        }
    }

    return g;
}

/**
 * Verify that a candidate cycle is a simple cycle of length k in the graph.
 *
 * @param g Graph instance.
 * @param k Target cycle length.
 * @param cycle Sequence of vertices (length k) intended to form the cycle.
 * @return true if the cycle is valid.
 */
bool validate_simple_cycle(const Graph &g, int k, const std::vector<int> &cycle) {
    if (static_cast<int>(cycle.size()) != k) {
        return false;
    }
    std::vector<char> seen(g.n, 0);
    for (int v : cycle) {
        if (v < 0 || v >= g.n || seen[v]) {
            return false;
        }
        seen[v] = 1;
    }
    for (int i = 0; i < k; ++i) {
        int u = cycle[i];
        int v = cycle[(i + 1) % k];
        if (!has_edge(g, u, v)) {
            return false;
        }
    }
    return true;
}

/**
 * Compute repetition count R(k) for color-coding given target failure probability.
 *
 * @param k Cycle length.
 * @param delta_target Allowed failure probability.
 * @param r_max Upper cap on repetitions.
 * @return Integer repetition count.
 */
int compute_repetitions(int k, double delta_target, int r_max) {
    long double log_fail = std::log(1.0L / delta_target);
    long double log_fact = std::lgammal(static_cast<long double>(k) + 1.0L);
    long double log_p = log_fact - static_cast<long double>(k) * std::log(static_cast<long double>(k));
    long double p = std::exp(log_p);
    int r = static_cast<int>(std::ceil(log_fail / std::max(p, std::numeric_limits<long double>::min())));
    if (r < 1) {
        r = 1;
    }
    if (r > r_max) {
        r = r_max;
    }
    return r;
}

/**
 * Reconstruct a path encoded by parent transitions.
 *
 * @param mask Current color mask.
 * @param v Terminal vertex.
 * @param parent_mask Flat parent mask array.
 * @param parent_vertex Flat parent vertex array.
 * @param n Number of vertices in the graph.
 * @return Path from start to v inclusive.
 */
std::vector<int> reconstruct_path(int mask, int v, const std::vector<int> &parent_mask,
                                  const std::vector<int> &parent_vertex, int n) {
    std::vector<int> path;
    while (mask >= 0 && v >= 0) {
        path.push_back(v);
        int idx = mask * n + v;
        int pm = parent_mask[idx];
        int pv = parent_vertex[idx];
        if (pm == -1) {
            break;
        }
        mask = pm;
        v = pv;
    }
    std::reverse(path.begin(), path.end());
    return path;
}

/**
 * Color-coding algorithm for finding a simple k-cycle.
 *
 * Runs exactly R random colorings, each with subset DP over color masks.
 * Returns true if any colorful k-cycle is found.
 *
 * @param g Input graph.
 * @param k Target cycle length.
 * @param R Number of colorings to run.
 * @param rng Random generator (advanced internally).
 * @param cycle_out On success, populated with k vertices forming a cycle.
 * @return true if a cycle is found across the R trials.
 */
bool cc_k_cycle(const Graph &g, int k, int R, std::mt19937_64 &rng, std::vector<int> &cycle_out) {
    if (k < 3 || k > 20) {
        return false;
    }
    int n = g.n;
    if (n == 0 || k > n) {
        return false;
    }
    int full_mask = (1 << k) - 1;
    int state_count = (1 << k) * n;

    std::vector<int> colors(n, 0);
    std::uniform_int_distribution<int> color_dist(0, k - 1);

    std::vector<char> dp(state_count, 0);
    std::vector<int> parent_mask(state_count, -1);
    std::vector<int> parent_vertex(state_count, -1);
    std::vector<std::vector<int>> masks_by_popcount(k + 1);
    for (int mask = 1; mask <= full_mask; ++mask) {
        int pc = __builtin_popcount(static_cast<unsigned>(mask));
        masks_by_popcount[pc].push_back(mask);
    }

    bool found_any = false;
    for (int iter = 0; iter < R; ++iter) {
        for (int v = 0; v < n; ++v) {
            colors[v] = color_dist(rng);
        }
        std::fill(dp.begin(), dp.end(), 0);
        std::fill(parent_mask.begin(), parent_mask.end(), -1);
        std::fill(parent_vertex.begin(), parent_vertex.end(), -1);

        for (int v = 0; v < n; ++v) {
            int mask = 1 << colors[v];
            int idx = mask * n + v;
            dp[idx] = 1;
            parent_mask[idx] = -1;
            parent_vertex[idx] = -1;
        }

        for (int pc = 1; pc < k; ++pc) {
            for (int mask : masks_by_popcount[pc]) {
                for (int v = 0; v < n; ++v) {
                    int idx = mask * n + v;
                    if (!dp[idx]) {
                        continue;
                    }
                    for (int nei : g.adj[v]) {
                        int c = colors[nei];
                        if (mask & (1 << c)) {
                            continue;
                        }
                        int next_mask = mask | (1 << c);
                        int next_idx = next_mask * n + nei;
                        if (!dp[next_idx]) {
                            dp[next_idx] = 1;
                            parent_mask[next_idx] = mask;
                            parent_vertex[next_idx] = v;
                        }
                    }
                }
            }
        }

        // Detect colorful k-cycle by closing any full-mask path into its start.
        for (int v = 0; v < n; ++v) {
            int idx = full_mask * n + v;
            if (!dp[idx]) {
                continue;
            }
            std::vector<int> path = reconstruct_path(full_mask, v, parent_mask, parent_vertex, n);
            if (static_cast<int>(path.size()) != k) {
                continue;
            }
            int start = path.front();
            int end = path.back();
            if (has_edge(g, start, end)) {
                if (!found_any) {
                    cycle_out = path;
                }
                found_any = true;
                break;
            }
        }
    }
    return found_any;
}

/**
 * Depth-limited DFS/backtracking for detecting a simple k-cycle.
 *
 * @param g Input graph.
 * @param k Target cycle length.
 * @param cycle_out On success, filled with k vertices forming the cycle.
 * @return true if a k-cycle is found.
 */
bool dfs_k_cycle(const Graph &g, int k, std::vector<int> &cycle_out) {
    if (k < 3 || k > g.n) {
        return false;
    }
    std::vector<char> visited(g.n, 0);
    std::vector<int> path;
    path.reserve(k);

    std::function<bool(int, int, int)> dfs = [&](int start, int v, int depth) -> bool {
        if (depth == k) {
            if (has_edge(g, v, start)) {
                cycle_out = path;
                return true;
            }
            return false;
        }
        for (int nei : g.adj[v]) {
            if (visited[nei]) {
                continue;
            }
            visited[nei] = 1;
            path.push_back(nei);
            if (dfs(start, nei, depth + 1)) {
                return true;
            }
            path.pop_back();
            visited[nei] = 0;
        }
        return false;
    };

    for (int start = 0; start < g.n; ++start) {
        std::fill(visited.begin(), visited.end(), 0);
        path.clear();
        visited[start] = 1;
        path.push_back(start);
        if (dfs(start, start, 1)) {
            return true;
        }
    }
    return false;
}

/**
 * Parse CLI arguments and run requested repetitions.
 *
 * Usage: ./kcycle_bench <algo_id> <n> <k> <p_noise> <graphs_per_rep> <seed_base> <reps>
 * algo_id: 0 = cc_k_cycle, 1 = dfs_k_cycle
 */
int main(int argc, char **argv) {
    if (argc != 8) {
        std::cerr << "Usage: ./kcycle_bench <algo_id> <n> <k> <p_noise> <graphs_per_rep> <seed_base> <reps>" << std::endl;
        return 1;
    }
    int arg_idx = 1;
    int algo_id = std::stoi(argv[arg_idx++]);
    int n = std::stoi(argv[arg_idx++]);
    int k = std::stoi(argv[arg_idx++]);
    double p_noise = std::stod(argv[arg_idx++]);
    int graphs_per_rep = std::stoi(argv[arg_idx++]);
    uint64_t seed_base = static_cast<uint64_t>(std::stoll(argv[arg_idx++]));
    int reps = std::stoi(argv[arg_idx++]);

    if (n <= 0 || k <= 0 || graphs_per_rep <= 0 || reps <= 0) {
        std::cerr << "All integer arguments must be positive." << std::endl;
        return 1;
    }
    if (p_noise < 0.0 || p_noise > 1.0) {
        std::cerr << "p_noise must lie in [0,1]." << std::endl;
        return 1;
    }

    const double delta_target = 0.1;
    const int R_max = 20;
    int repetitions_cc = compute_repetitions(k, delta_target, R_max);

    for (int rep = 0; rep < reps; ++rep) {
        std::mt19937_64 rng(seed_base + static_cast<uint64_t>(rep));
        long long total_time_ns = 0;
        int error_count = 0;

        for (int g_idx = 0; g_idx < graphs_per_rep; ++g_idx) {
            Graph g = planted_cycle_noise(n, k, p_noise, rng);
            Timer timer;
            timer.start();
            std::vector<int> cycle;
            bool found = false;
            if (algo_id == 0) {
                found = cc_k_cycle(g, k, repetitions_cc, rng, cycle);
            } else if (algo_id == 1) {
                found = dfs_k_cycle(g, k, cycle);
            } else {
                std::cerr << "Unknown algo_id: " << algo_id << std::endl;
                return 1;
            }
            total_time_ns += timer.elapsed_ns();
            bool valid = found && validate_simple_cycle(g, k, cycle);
            if (!valid) {
                error_count += 1;
            }
        }
        std::cout << total_time_ns << " " << error_count << std::endl;
    }
    return 0;
}
