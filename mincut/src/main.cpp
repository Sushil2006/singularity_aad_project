#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

/**
 * Graph container holding edge list and adjacency matrix for an undirected multigraph.
 */
struct Graph {
    int n = 0;
    std::vector<std::pair<int, int>> edges;
    std::vector<std::vector<int>> adj_matrix;
};

/**
 * Simple timer utility for measuring elapsed time in nanoseconds.
 */
class Timer {
  public:
    /**
     * Start or restart the timer.
     */
    void start() { start_time_ = std::chrono::steady_clock::now(); }

    /**
     * Return elapsed time in nanoseconds since the last start.
     *
     * @return long long elapsed nanoseconds.
     */
    long long elapsed_ns() const {
        auto end_time = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time_).count();
    }

  private:
    std::chrono::steady_clock::time_point start_time_{};
};

/**
 * Disjoint-set union (Union-Find) with path compression and union by size.
 */
class DSU {
  public:
    /**
     * Construct DSU with n singleton sets.
     *
     * @param n Number of elements.
     */
    explicit DSU(int n) : parent_(n), size_(n, 1) {
        for (int i = 0; i < n; ++i) {
            parent_[i] = i;
        }
    }

    /**
     * Find the representative of an element with path compression.
     *
     * @param x Element index.
     * @return int Representative element.
     */
    int find(int x) {
        if (parent_[x] != x) {
            parent_[x] = find(parent_[x]);
        }
        return parent_[x];
    }

    /**
     * Union two sets by size.
     *
     * @param a First element.
     * @param b Second element.
     * @return bool True if union performed, false if already in same set.
     */
    bool unite(int a, int b) {
        int root_a = find(a);
        int root_b = find(b);
        if (root_a == root_b) {
            return false;
        }
        if (size_[root_a] < size_[root_b]) {
            std::swap(root_a, root_b);
        }
        parent_[root_b] = root_a;
        size_[root_a] += size_[root_b];
        return true;
    }

  private:
    std::vector<int> parent_;
    std::vector<int> size_;
};

/**
 * Check whether a graph described by edges is connected using BFS.
 *
 * @param n Number of vertices.
 * @param edges Edge list.
 * @return bool True if connected, else false.
 */
bool is_connected(int n, const std::vector<std::pair<int, int>>& edges) {
    if (n == 0) {
        return true;
    }
    std::vector<std::vector<int>> adj(n);
    adj.reserve(n);
    for (const auto& e : edges) {
        adj[e.first].push_back(e.second);
        adj[e.second].push_back(e.first);
    }
    std::vector<bool> visited(n, false);
    std::vector<int> stack;
    stack.push_back(0);
    visited[0] = true;
    while (!stack.empty()) {
        int v = stack.back();
        stack.pop_back();
        for (int nei : adj[v]) {
            if (!visited[nei]) {
                visited[nei] = true;
                stack.push_back(nei);
            }
        }
    }
    for (bool v : visited) {
        if (!v) {
            return false;
        }
    }
    return true;
}

/**
 * Build adjacency matrix from the current edge list.
 *
 * @param n Number of vertices.
 * @param edges Edge list (parallel edges allowed).
 * @return std::vector<std::vector<int>> Symmetric adjacency matrix with multiplicities.
 */
std::vector<std::vector<int>> build_adj_matrix(int n, const std::vector<std::pair<int, int>>& edges) {
    std::vector<std::vector<int>> w(n, std::vector<int>(n, 0));
    for (const auto& e : edges) {
        int u = e.first;
        int v = e.second;
        w[u][v] += 1;
        w[v][u] += 1;
    }
    return w;
}

/**
 * Generate a single sample of the two-cluster G(n, p) model (not necessarily connected).
 *
 * @param n Number of vertices (must be even).
 * @param p_in Edge probability within a cluster.
 * @param p_out Edge probability across clusters.
 * @param rng Random number generator.
 * @return Graph Sampled graph instance.
 */
Graph generate_two_cluster_sample(int n, double p_in, double p_out, std::mt19937_64& rng) {
    if (n % 2 != 0) {
        throw std::invalid_argument("two_cluster_gnp requires even n");
    }
    int half = n / 2;
    Graph g;
    g.n = n;
    std::vector<std::pair<int, int>> edges;
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    // Edges within first cluster.
    for (int i = 0; i < half; ++i) {
        for (int j = i + 1; j < half; ++j) {
            if (dist(rng) < p_in) {
                edges.emplace_back(i, j);
            }
        }
    }
    // Edges within second cluster.
    for (int i = half; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (dist(rng) < p_in) {
                edges.emplace_back(i, j);
            }
        }
    }
    // Cross-cluster edges.
    for (int i = 0; i < half; ++i) {
        for (int j = half; j < n; ++j) {
            if (dist(rng) < p_out) {
                edges.emplace_back(i, j);
            }
        }
    }
    g.edges = std::move(edges);
    g.adj_matrix = build_adj_matrix(n, g.edges);
    return g;
}

/**
 * Generate a connected two-cluster G(n, p) graph, retrying until connectivity holds.
 *
 * @param n Number of vertices (must be even).
 * @param p_in Edge probability within a cluster.
 * @param p_out Edge probability across clusters.
 * @param rng Random number generator.
 * @return Graph Connected graph instance.
 */
Graph generate_two_cluster_connected(int n, double p_in, double p_out, std::mt19937_64& rng) {
    constexpr int kMaxAttempts = 200;
    for (int attempt = 0; attempt < kMaxAttempts; ++attempt) {
        Graph g = generate_two_cluster_sample(n, p_in, p_out, rng);
        if (is_connected(n, g.edges)) {
            return g;
        }
    }
    throw std::runtime_error("Failed to generate connected two_cluster_gnp graph after 200 attempts");
}

/**
 * Generate one Erdos-Renyi G(n, p) sample (not necessarily connected).
 *
 * @param n Number of vertices.
 * @param p Edge probability.
 * @param rng Random number generator.
 * @return Graph Sampled graph instance.
 */
Graph generate_gnp_sample(int n, double p, std::mt19937_64& rng) {
    Graph g;
    g.n = n;
    std::vector<std::pair<int, int>> edges;
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            if (dist(rng) < p) {
                edges.emplace_back(i, j);
            }
        }
    }
    g.edges = std::move(edges);
    g.adj_matrix = build_adj_matrix(n, g.edges);
    return g;
}

/**
 * Generate a connected G(n, p) graph with retry.
 *
 * @param n Number of vertices.
 * @param p Edge probability.
 * @param rng Random number generator.
 * @return Graph Connected graph instance.
 */
Graph generate_gnp_connected(int n, double p, std::mt19937_64& rng) {
    constexpr int kMaxAttempts = 200;
    for (int attempt = 0; attempt < kMaxAttempts; ++attempt) {
        Graph g = generate_gnp_sample(n, p, rng);
        if (is_connected(n, g.edges)) {
            return g;
        }
    }
    throw std::runtime_error("Failed to generate connected G(n,p) graph after 200 attempts");
}

/**
 * Generate a deterministic adversarial barbell: two cliques joined by a single bridge edge.
 *
 * @param n Number of vertices (split roughly in half).
 * @return Graph Barbell graph with min-cut = 1 on the bridge.
 */
Graph generate_barbell(int n) {
    if (n < 2) {
        throw std::invalid_argument("barbell requires at least 2 vertices");
    }
    int left_size = n / 2;
    int right_size = n - left_size;
    Graph g;
    g.n = n;
    std::vector<std::pair<int, int>> edges;

    // Left clique on vertices [0, left_size).
    for (int i = 0; i < left_size; ++i) {
        for (int j = i + 1; j < left_size; ++j) {
            edges.emplace_back(i, j);
        }
    }
    // Right clique on vertices [left_size, n).
    for (int i = left_size; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            edges.emplace_back(i, j);
        }
    }
    // Single bridge edge connecting the cliques (unique min-cut).
    edges.emplace_back(left_size - 1, left_size);

    g.edges = std::move(edges);
    g.adj_matrix = build_adj_matrix(n, g.edges);
    return g;
}

/**
 * Generate a connected simple d-regular graph using the configuration model with rejection.
 *
 * @param n Number of vertices.
 * @param degree Desired regular degree (must satisfy n * degree even).
 * @param rng Random number generator.
 * @return Graph Connected d-regular graph instance.
 */
Graph generate_regular_connected(int n, int degree, std::mt19937_64& rng) {
    if (degree <= 0 || degree >= n) {
        throw std::invalid_argument("degree must be in [1, n - 1] for regular graph");
    }
    if ((n * degree) % 2 != 0) {
        throw std::invalid_argument("n * degree must be even for regular graph");
    }
    constexpr int kMaxAttempts = 500;
    for (int attempt = 0; attempt < kMaxAttempts; ++attempt) {
        std::vector<int> stubs;
        stubs.reserve(n * degree);
        for (int v = 0; v < n; ++v) {
            for (int j = 0; j < degree; ++j) {
                stubs.push_back(v);
            }
        }
        std::shuffle(stubs.begin(), stubs.end(), rng);

        std::vector<std::pair<int, int>> edges;
        edges.reserve(stubs.size() / 2);
        std::vector<std::vector<int>> seen(n, std::vector<int>(n, 0));
        bool simple = true;
        for (size_t i = 0; i + 1 < stubs.size(); i += 2) {
            int u = stubs[i];
            int v = stubs[i + 1];
            if (u == v || seen[u][v] > 0) {
                simple = false;
                break;
            }
            seen[u][v] = 1;
            seen[v][u] = 1;
            edges.emplace_back(u, v);
        }
        if (!simple) {
            continue;
        }
        if (!is_connected(n, edges)) {
            continue;
        }
        Graph g;
        g.n = n;
        g.edges = std::move(edges);
        g.adj_matrix = build_adj_matrix(n, g.edges);
        return g;
    }
    throw std::runtime_error("Failed to generate connected regular graph after 500 attempts");
}

/**
 * Factory for generating graphs according to dist_id.
 *
 * @param dist_id Distribution identifier (0 two_cluster_gnp, 1 gnp_sparse, 2 gnp_dense, 3 adversarial_barbell, 4 regular_degree_3).
 * @param n Number of vertices.
 * @param rng Random number generator.
 * @return Graph Connected sampled graph.
 */
Graph generate_graph(int dist_id, int n, std::mt19937_64& rng) {
    constexpr double kPIn = 0.3;
    constexpr double kPOut = 0.05;
    constexpr double kSparseC = 6.0;
    constexpr double kPDense = 0.2;
    constexpr int kRegularDegree = 3;

    switch (dist_id) {
        case 0:
            return generate_two_cluster_connected(n, kPIn, kPOut, rng);
        case 1: {
            double p_sparse = kSparseC / static_cast<double>(n);
            if (p_sparse > 1.0) {
                p_sparse = 1.0;
            }
            return generate_gnp_connected(n, p_sparse, rng);
        }
        case 2:
            return generate_gnp_connected(n, kPDense, rng);
        case 3:
            return generate_barbell(n);
        case 4:
            return generate_regular_connected(n, kRegularDegree, rng);
        default:
            throw std::invalid_argument("Invalid dist_id");
    }
}

/**
 * Single run of Karger's randomized contraction algorithm on a fixed edge list.
 *
 * @param g Graph with n vertices and edge list.
 * @param rng Random number generator.
 * @return int Cut value found in this run.
 */
int karger_single_run(const Graph& g, std::mt19937_64& rng) {
    if (g.n < 2) {
        throw std::invalid_argument("Graph must have at least 2 vertices for Karger");
    }
    DSU dsu(g.n);
    int components = g.n;
    std::uniform_int_distribution<int> edge_dist(0, static_cast<int>(g.edges.size()) - 1);
    while (components > 2) {
        const auto& e = g.edges[edge_dist(rng)];
        int u_root = dsu.find(e.first);
        int v_root = dsu.find(e.second);
        if (u_root == v_root) {
            continue;
        }
        dsu.unite(u_root, v_root);
        --components;
    }
    int cut = 0;
    for (const auto& e : g.edges) {
        if (dsu.find(e.first) != dsu.find(e.second)) {
            ++cut;
        }
    }
    return cut;
}

/**
 * Compute number of repetitions R(n) to hit target failure probability delta_target.
 *
 * @param n Number of vertices.
 * @param delta_target Desired failure probability upper bound.
 * @return int Number of independent runs to perform (at least 1).
 */
int karger_repetitions(int n, double delta_target) {
    double expected_runs = 0.5 * static_cast<double>(n) * static_cast<double>(n - 1) *
                           std::log(1.0 / delta_target);
    int runs = static_cast<int>(std::ceil(expected_runs));
    runs = std::min(runs, 100);
    return std::max(1, runs);
}

/**
 * Run repeated Karger trials and return the best cut observed.
 *
 * @param g Graph instance.
 * @param rng Random number generator.
 * @param delta_target Global failure probability target.
 * @return int Minimum cut value observed across runs.
 */
int repeated_karger(const Graph& g, std::mt19937_64& rng, double delta_target) {
    int runs = karger_repetitions(g.n, delta_target);
    int best_cut = std::numeric_limits<int>::max();
    for (int i = 0; i < runs; ++i) {
        int cut = karger_single_run(g, rng);
        if (cut < best_cut) {
            best_cut = cut;
        }
    }
    return best_cut;
}

/**
 * Stoer-Wagner global min-cut algorithm for undirected weighted graphs.
 *
 * @param w Symmetric adjacency matrix with integer weights.
 * @return int Exact global min-cut value.
 */
int stoer_wagner(std::vector<std::vector<int>> w) {
    int n = static_cast<int>(w.size());
    if (n == 0) {
        return 0;
    }
    std::vector<int> vertices(n);
    for (int i = 0; i < n; ++i) {
        vertices[i] = i;
    }
    int best_cut = std::numeric_limits<int>::max();
    for (int phase = 0; phase < n - 1; ++phase) {
        int m = n - phase;
        std::vector<int> weights(n, 0);
        std::vector<bool> added(n, false);
        int prev = -1;

        for (int i = 0; i < m; ++i) {
            int sel = -1;
            for (int j = 0; j < m; ++j) {
                int v = vertices[j];
                if (!added[v] && (sel == -1 || weights[v] > weights[sel])) {
                    sel = v;
                }
            }
            if (i == m - 1) {
                // 'sel' is t, 'prev' is s.
                if (weights[sel] < best_cut) {
                    best_cut = weights[sel];
                }
                // Merge sel into prev.
                int idx_prev = -1;
                int idx_sel = -1;
                for (int j = 0; j < m; ++j) {
                    if (vertices[j] == prev) idx_prev = j;
                    if (vertices[j] == sel) idx_sel = j;
                }
                if (idx_prev == -1 || idx_sel == -1) {
                    throw std::runtime_error("Stoer-Wagner internal indexing failure");
                }
                for (int j = 0; j < m; ++j) {
                    int v = vertices[j];
                    if (v == prev) continue;
                    w[prev][v] += w[sel][v];
                    w[v][prev] = w[prev][v];
                }
                // Remove sel from active vertices.
                vertices[idx_sel] = vertices[m - 1];
            } else {
                added[sel] = true;
                for (int j = 0; j < m; ++j) {
                    int v = vertices[j];
                    if (!added[v]) {
                        weights[v] += w[sel][v];
                    }
                }
                prev = sel;
            }
        }
    }
    return best_cut;
}

/**
 * Parse CLI arguments and execute benchmarks per specification.
 */
int main(int argc, char* argv[]) {
    if (argc != 7) {
        std::cerr << "Usage: ./mincut_bench <algo_id> <dist_id> <n> <graphs_per_rep> <seed_base> <reps>\n";
        return 1;
    }

    int algo_id = 0;
    int dist_id = 0;
    int n = 0;
    int graphs_per_rep = 0;
    uint64_t seed_base = 0;
    int reps = 0;
    try {
        algo_id = std::stoi(argv[1]);
        dist_id = std::stoi(argv[2]);
        n = std::stoi(argv[3]);
        graphs_per_rep = std::stoi(argv[4]);
        seed_base = std::stoull(argv[5]);
        reps = std::stoi(argv[6]);
    } catch (const std::exception& e) {
        std::cerr << "Error parsing arguments: " << e.what() << "\n";
        return 1;
    }

    if (algo_id < 0 || algo_id > 1) {
        std::cerr << "algo_id must be 0 (Karger) or 1 (StoerWagner)\n";
        return 1;
    }
    if (dist_id < 0 || dist_id > 4) {
        std::cerr << "dist_id must be 0, 1, 2, 3, or 4\n";
        return 1;
    }
    if (n < 2) {
        std::cerr << "n must be at least 2\n";
        return 1;
    }
    if (graphs_per_rep <= 0 || reps <= 0) {
        std::cerr << "graphs_per_rep and reps must be positive\n";
        return 1;
    }

    constexpr double kDeltaTarget = 0.95;
    for (int rep = 0; rep < reps; ++rep) {
        std::mt19937_64 rng(seed_base + static_cast<uint64_t>(rep));
        long long total_time_ns = 0;
        long long sum_sq_error = 0;
        for (int g_idx = 0; g_idx < graphs_per_rep; ++g_idx) {
            Graph g = generate_graph(dist_id, n, rng);
            int lambda_truth = stoer_wagner(g.adj_matrix);

            Timer timer;
            timer.start();
            int lambda_hat = 0;
            if (algo_id == 0) {
                lambda_hat = repeated_karger(g, rng, kDeltaTarget);
            } else {
                lambda_hat = stoer_wagner(g.adj_matrix);
            }
            total_time_ns += timer.elapsed_ns();
            long long err = static_cast<long long>(lambda_hat) - static_cast<long long>(lambda_truth);
            sum_sq_error += err * err;
        }
        std::cout << total_time_ns << " " << sum_sq_error << "\n";
    }
    return 0;
}
