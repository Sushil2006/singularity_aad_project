#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <queue>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

struct Edge {
    int u;
    int v;
    int w;
};

struct MSTResult {
    long long weight = 0;
    std::vector<Edge> edges;
};

constexpr int BASE_N = 32;
constexpr int BASE_M = 64;

// Simple utility to check connectivity via DFS on an undirected graph.
bool is_connected(int n, const std::vector<Edge>& edges) {
    if (n == 0) return false;
    std::vector<std::vector<int>> adj(n);
    adj.reserve(n);
    for (const auto& e : edges) {
        adj[e.u].push_back(e.v);
        adj[e.v].push_back(e.u);
    }
    std::vector<int> stack;
    std::vector<char> vis(n, 0);
    stack.push_back(0);
    vis[0] = 1;
    while (!stack.empty()) {
        int v = stack.back();
        stack.pop_back();
        for (int to : adj[v]) {
            if (vis[to]) continue;
            vis[to] = 1;
            stack.push_back(to);
        }
    }
    for (char f : vis) {
        if (!f) return false;
    }
    return true;
}

// Pick the minimum outgoing edge for each component.
std::vector<int> choose_min_outgoing(
    int comp_count,
    const std::vector<int>& comp_id,
    const std::vector<Edge>& edges
) {
    std::vector<int> best(comp_count, -1);
    for (int idx = 0; idx < static_cast<int>(edges.size()); ++idx) {
        const auto& e = edges[idx];
        int a = comp_id[e.u];
        int b = comp_id[e.v];
        if (a == b) continue;
        if (best[a] == -1 || e.w < edges[best[a]].w) best[a] = idx;
        if (best[b] == -1 || e.w < edges[best[b]].w) best[b] = idx;
    }
    std::vector<char> seen(edges.size(), 0);
    std::vector<int> chosen;
    chosen.reserve(comp_count);
    for (int c = 0; c < comp_count; ++c) {
        int idx = best[c];
        if (idx != -1 && !seen[idx]) {
            chosen.push_back(idx);
            seen[idx] = 1;
        }
    }
    return chosen;
}

struct ComponentUpdate {
    std::vector<int> new_comp_id;
    int new_comp_count = 0;
    std::vector<std::vector<int>> groups;
};

// Merge components according to chosen edges (between components).
ComponentUpdate merge_components(
    int n,
    int comp_count,
    const std::vector<int>& comp_id,
    const std::vector<Edge>& edges,
    const std::vector<int>& chosen_indices
) {
    std::vector<std::vector<int>> comp_adj(comp_count);
    comp_adj.reserve(comp_count);
    for (int idx : chosen_indices) {
        const auto& e = edges[idx];
        int a = comp_id[e.u];
        int b = comp_id[e.v];
        if (a == b) continue;
        comp_adj[a].push_back(b);
        comp_adj[b].push_back(a);
    }

    std::vector<int> comp_to_new(comp_count, -1);
    std::vector<std::vector<int>> groups;
    groups.reserve(comp_count);
    int new_cnt = 0;
    std::vector<int> stack;
    std::vector<char> vis(comp_count, 0);
    for (int c = 0; c < comp_count; ++c) {
        if (vis[c]) continue;
        vis[c] = 1;
        stack.push_back(c);
        std::vector<int> current;
        while (!stack.empty()) {
            int x = stack.back();
            stack.pop_back();
            comp_to_new[x] = new_cnt;
            current.push_back(x);
            for (int to : comp_adj[x]) {
                if (!vis[to]) {
                    vis[to] = 1;
                    stack.push_back(to);
                }
            }
        }
        groups.push_back(std::move(current));
        ++new_cnt;
    }

    std::vector<int> new_comp_id(n, -1);
    for (int v = 0; v < n; ++v) {
        new_comp_id[v] = comp_to_new[comp_id[v]];
    }

    ComponentUpdate res;
    res.new_comp_id = std::move(new_comp_id);
    res.new_comp_count = new_cnt;
    res.groups = std::move(groups);
    return res;
}

struct BoruvkaResult {
    int new_n = 0;
    std::vector<Edge> contracted_edges;
    std::vector<Edge> boruvka_edges;
    std::vector<int> comp_repr;
};

// Contract parallel edges using bucketed adjacency without hashing.
std::vector<Edge> build_contracted_edges(
    int n,
    int comp_count,
    const std::vector<int>& comp_id,
    const std::vector<Edge>& edges
) {
    std::vector<std::vector<std::pair<int, int>>> buckets(comp_count);
    for (const auto& e : edges) {
        int a = comp_id[e.u];
        int b = comp_id[e.v];
        if (a == b) continue;
        if (a > b) std::swap(a, b);
        buckets[a].push_back({b, e.w});
    }

    std::vector<int> mark(comp_count, -1);
    std::vector<int> best(comp_count, 0);
    int stamp = 0;
    std::vector<int> neighbor_list;
    neighbor_list.reserve(comp_count);

    std::vector<Edge> contracted;
    contracted.reserve(edges.size());

    for (int a = 0; a < comp_count; ++a) {
        ++stamp;
        neighbor_list.clear();
        for (const auto& item : buckets[a]) {
            int b = item.first;
            int w = item.second;
            if (mark[b] != stamp) {
                mark[b] = stamp;
                best[b] = w;
                neighbor_list.push_back(b);
            } else if (w < best[b]) {
                best[b] = w;
            }
        }
        for (int b : neighbor_list) {
            contracted.push_back({a, b, best[b]});
        }
    }
    return contracted;
}

// Two Boruvka steps using component labeling via DFS (no union-find).
BoruvkaResult run_boruvka_two_steps(int n, const std::vector<Edge>& edges) {
    std::vector<int> comp_id(n);
    std::iota(comp_id.begin(), comp_id.end(), 0);
    int comp_count = n;
    std::vector<Edge> chosen_edges;

    for (int iter = 0; iter < 2 && comp_count > 1; ++iter) {
        std::vector<int> chosen = choose_min_outgoing(comp_count, comp_id, edges);
        if (chosen.empty()) break;
        for (int idx : chosen) chosen_edges.push_back(edges[idx]);

        ComponentUpdate upd = merge_components(n, comp_count, comp_id, edges, chosen);
        comp_id.swap(upd.new_comp_id);
        comp_count = upd.new_comp_count;
    }

    std::vector<int> repr(comp_count, -1);
    for (int v = 0; v < n; ++v) {
        int cid = comp_id[v];
        if (repr[cid] == -1) repr[cid] = v;
    }

    std::vector<Edge> contracted = build_contracted_edges(n, comp_count, comp_id, edges);
    BoruvkaResult res;
    res.new_n = comp_count;
    res.contracted_edges = std::move(contracted);
    res.boruvka_edges = std::move(chosen_edges);
    res.comp_repr = std::move(repr);
    return res;
}

// Boruvka-based MST (used for base cases and sampling).
MSTResult boruvka_mst(int n, const std::vector<Edge>& edges, bool require_connected = true) {
    if (n == 0) throw std::invalid_argument("n must be positive");
    std::vector<int> comp_id(n);
    std::iota(comp_id.begin(), comp_id.end(), 0);
    int comp_count = n;
    MSTResult res;

    std::vector<char> seen(edges.size(), 0);
    int stagnant = 0;
    while (comp_count > 1) {
        std::vector<int> chosen = choose_min_outgoing(comp_count, comp_id, edges);
        if (chosen.empty()) {
            stagnant++;
            if (stagnant > 1) break;
            continue;
        }
        for (int idx : chosen) {
            if (!seen[idx]) {
                res.weight += edges[idx].w;
                res.edges.push_back(edges[idx]);
                seen[idx] = 1;
            }
        }
        ComponentUpdate upd = merge_components(n, comp_count, comp_id, edges, chosen);
        comp_id.swap(upd.new_comp_id);
        comp_count = upd.new_comp_count;
    }
    if (require_connected && comp_count != 1) {
        throw std::runtime_error("Graph is disconnected.");
    }
    return res;
}

// Build Boruvka tree over a given MST to answer max-edge queries on tree paths.
struct BoruvkaTree {
    std::vector<std::vector<std::pair<int, int>>> adj;  // node -> [(to, weight)]
    int root = -1;
    int node_count = 0;
    int leaf_count = 0;
};

struct ComponentGroups {
    std::vector<int> comp_to_new;
    std::vector<std::vector<int>> groups;
    int new_count = 0;
};

ComponentGroups merge_component_graph(
    int comp_count,
    const std::vector<int>& best_edge_idx,
    const std::vector<Edge>& edges,
    const std::vector<int>& comp_id
) {
    std::vector<std::vector<int>> comp_adj(comp_count);
    for (int c = 0; c < comp_count; ++c) {
        int idx = best_edge_idx[c];
        if (idx == -1) continue;
        int a = comp_id[edges[idx].u];
        int b = comp_id[edges[idx].v];
        if (a == b) continue;
        comp_adj[a].push_back(b);
        comp_adj[b].push_back(a);
    }
    std::vector<int> comp_to_new(comp_count, -1);
    std::vector<std::vector<int>> groups;
    groups.reserve(comp_count);
    std::vector<int> stack;
    std::vector<char> vis(comp_count, 0);
    int new_cnt = 0;
    for (int c = 0; c < comp_count; ++c) {
        if (vis[c]) continue;
        vis[c] = 1;
        stack.push_back(c);
        std::vector<int> current;
        while (!stack.empty()) {
            int x = stack.back();
            stack.pop_back();
            comp_to_new[x] = new_cnt;
            current.push_back(x);
            for (int to : comp_adj[x]) {
                if (!vis[to]) {
                    vis[to] = 1;
                    stack.push_back(to);
                }
            }
        }
        groups.push_back(std::move(current));
        ++new_cnt;
    }
    ComponentGroups res;
    res.comp_to_new = std::move(comp_to_new);
    res.groups = std::move(groups);
    res.new_count = new_cnt;
    return res;
}

BoruvkaTree build_boruvka_tree(int n, const std::vector<Edge>& mst_edges) {
    BoruvkaTree tree;
    int max_nodes = 2 * n;
    tree.adj.assign(max_nodes, {});
    tree.leaf_count = n;

    std::vector<int> comp_id(n);
    std::iota(comp_id.begin(), comp_id.end(), 0);
    int comp_count = n;
    std::vector<int> node_id(comp_count);
    std::iota(node_id.begin(), node_id.end(), 0);
    int next_node = n;

    while (comp_count > 1) {
        std::vector<int> best(comp_count, -1);
        for (int idx = 0; idx < static_cast<int>(mst_edges.size()); ++idx) {
            const auto& e = mst_edges[idx];
            int a = comp_id[e.u];
            int b = comp_id[e.v];
            if (a == b) continue;
            if (best[a] == -1 || e.w < mst_edges[best[a]].w) best[a] = idx;
            if (best[b] == -1 || e.w < mst_edges[best[b]].w) best[b] = idx;
        }

        ComponentGroups merged = merge_component_graph(comp_count, best, mst_edges, comp_id);
        std::vector<int> new_node_id(merged.new_count, -1);

        for (int new_c = 0; new_c < merged.new_count; ++new_c) {
            int parent_node = next_node++;
            new_node_id[new_c] = parent_node;
            for (int old_c : merged.groups[new_c]) {
                int edge_idx = best[old_c];
                int weight = edge_idx == -1 ? 0 : mst_edges[edge_idx].w;
                int child_node = node_id[old_c];
                tree.adj[parent_node].push_back({child_node, weight});
                tree.adj[child_node].push_back({parent_node, weight});
            }
        }

        std::vector<int> new_comp_id(n, -1);
        for (int v = 0; v < n; ++v) {
            int old_c = comp_id[v];
            int new_c = merged.comp_to_new[old_c];
            new_comp_id[v] = new_c;
        }
        comp_id.swap(new_comp_id);
        comp_count = merged.new_count;
        node_id.swap(new_node_id);
    }

    tree.node_count = next_node;
    tree.root = node_id[comp_id[0]];
    tree.adj.resize(tree.node_count);
    return tree;
}

// Euler tour + RMQ (sparse table) for LCA on Boruvka tree; O(1) queries.
struct LCAData {
    std::vector<int> euler;
    std::vector<int> depth;
    std::vector<int> first;
    std::vector<std::vector<int>> st;
    std::vector<int> log2;
    std::vector<int> edge_to_parent;
    std::vector<int> parent;
};

LCAData build_lca(const BoruvkaTree& tree) {
    int N = tree.node_count;
    LCAData data;
    data.first.assign(N, -1);
    data.edge_to_parent.assign(N, 0);
    data.parent.assign(N, -1);

    struct Frame { int v; int p; int w; size_t idx; int depth; };
    std::vector<Frame> stack;
    stack.push_back({tree.root, -1, 0, 0, 0});

    while (!stack.empty()) {
        Frame& f = stack.back();
        if (f.idx == 0) {
            data.parent[f.v] = f.p;
            data.edge_to_parent[f.v] = f.w;
            data.first[f.v] = static_cast<int>(data.euler.size());
            data.euler.push_back(f.v);
            data.depth.push_back(f.depth);
        }
        if (f.idx == tree.adj[f.v].size()) {
            stack.pop_back();
            if (f.p != -1) {
                data.euler.push_back(f.p);
                data.depth.push_back(f.depth - 1);
            }
            continue;
        }
        auto [to, w] = tree.adj[f.v][f.idx++];
        if (to == f.p) continue;
        stack.push_back({to, f.v, w, 0, f.depth + 1});
    }

    int M = static_cast<int>(data.euler.size());
    data.log2.assign(M + 1, 0);
    for (int i = 2; i <= M; ++i) data.log2[i] = data.log2[i / 2] + 1;
    int K = data.log2[M] + 1;
    data.st.assign(K, std::vector<int>(M));
    for (int i = 0; i < M; ++i) data.st[0][i] = i;
    for (int k = 1; k < K; ++k) {
        int span = 1 << k;
        int half = span >> 1;
        for (int i = 0; i + span <= M; ++i) {
            int left = data.st[k - 1][i];
            int right = data.st[k - 1][i + half];
            data.st[k][i] = (data.depth[left] < data.depth[right]) ? left : right;
        }
    }
    return data;
}

int lca_query(const LCAData& data, int u, int v) {
    int l = data.first[u];
    int r = data.first[v];
    if (l > r) std::swap(l, r);
    int len = r - l + 1;
    int k = data.log2[len];
    int left = data.st[k][l];
    int right = data.st[k][r - (1 << k) + 1];
    return (data.depth[left] < data.depth[right]) ? data.euler[left] : data.euler[right];
}

// Compute maximum edge weight on the path between two leaves in the Boruvka tree.
int max_on_path(
    const BoruvkaTree& tree,
    const LCAData& lca,
    int u,
    int v
) {
    int ancestor = lca_query(lca, u, v);
    int ans = 0;
    int cur = u;
    while (cur != ancestor) {
        ans = std::max(ans, lca.edge_to_parent[cur]);
        cur = lca.parent[cur];
    }
    cur = v;
    while (cur != ancestor) {
        ans = std::max(ans, lca.edge_to_parent[cur]);
        cur = lca.parent[cur];
    }
    return ans;
}

// Filter F-heavy edges using Boruvka tree queries.
std::vector<Edge> filter_f_light_edges(
    int n,
    const std::vector<Edge>& edges,
    const std::vector<Edge>& forest_edges
) {
    BoruvkaTree tree = build_boruvka_tree(n, forest_edges);
    LCAData lca = build_lca(tree);
    std::vector<Edge> filtered;
    filtered.reserve(edges.size());
    for (const auto& e : edges) {
        int max_w = max_on_path(tree, lca, e.u, e.v);
        if (e.w <= max_w) filtered.push_back(e);
    }
    return filtered;
}

MSTResult kkt_mst(int n, const std::vector<Edge>& edges, std::mt19937_64& rng) {
    if (n <= BASE_N || static_cast<int>(edges.size()) <= BASE_M) {
        return boruvka_mst(n, edges);
    }

    BoruvkaResult b = run_boruvka_two_steps(n, edges);
    long long boruvka_weight = 0;
    for (const auto& e : b.boruvka_edges) boruvka_weight += e.w;
    if (b.new_n == 1) {
        MSTResult res;
        res.weight = boruvka_weight;
        res.edges = b.boruvka_edges;
        return res;
    }

    std::bernoulli_distribution coin(0.5);
    std::vector<Edge> sample_edges;
    sample_edges.reserve(b.contracted_edges.size() / 2 + 1);
    for (const auto& e : b.contracted_edges) {
        if (coin(rng)) sample_edges.push_back(e);
    }

    MSTResult sample_mst = boruvka_mst(b.new_n, sample_edges);
    std::vector<Edge> f_light = filter_f_light_edges(b.new_n, b.contracted_edges, sample_mst.edges);
    if (!is_connected(b.new_n, f_light)) {
        f_light = b.contracted_edges;
    }
    MSTResult filtered_mst = kkt_mst(b.new_n, f_light, rng);

    MSTResult res;
    res.weight = boruvka_weight + filtered_mst.weight;
    res.edges = b.boruvka_edges;
    res.edges.reserve(res.edges.size() + filtered_mst.edges.size());
    for (const auto& e : filtered_mst.edges) {
        int u_orig = b.comp_repr[e.u];
        int v_orig = b.comp_repr[e.v];
        res.edges.push_back({u_orig, v_orig, e.w});
    }
    return res;
}

// Random connected graph generator (allows parallel edges, avoids hashes).
std::vector<Edge> generate_random_connected_graph(int n, int m, std::mt19937_64& rng) {
    if (n < 2) throw std::invalid_argument("n must be at least 2");
    if (m < n - 1 || static_cast<long long>(m) > static_cast<long long>(n) * (n - 1) / 2) {
        throw std::invalid_argument("m out of valid range for simple undirected graph");
    }

    std::uniform_int_distribution<int> weight_dist(1, 1'000'000);
    std::uniform_int_distribution<int> vertex_dist(0, n - 1);

    std::vector<Edge> edges;
    edges.reserve(m);

    for (int v = 1; v < n; ++v) {
        std::uniform_int_distribution<int> parent_dist(0, v - 1);
        int u = parent_dist(rng);
        int w = weight_dist(rng);
        edges.push_back({u, v, w});
    }

    while (static_cast<int>(edges.size()) < m) {
        int u = vertex_dist(rng);
        int v = vertex_dist(rng);
        if (u == v) continue;
        int w = weight_dist(rng);
        edges.push_back({u, v, w});
    }
    return edges;
}

long long run_single_algo(int algo_id, int n, int m, std::mt19937_64& rng) {
    std::vector<Edge> edges = generate_random_connected_graph(n, m, rng);
    if (!is_connected(n, edges)) throw std::runtime_error("Generated graph disconnected");

    if (algo_id == 0) {
        MSTResult ref = boruvka_mst(n, edges);
        auto start = std::chrono::steady_clock::now();
        MSTResult res = kkt_mst(n, edges, rng);
        auto end = std::chrono::steady_clock::now();
        if (res.weight != ref.weight) {
            throw std::runtime_error("KKT mismatch");
        }
        return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    } else if (algo_id == 1) {
        auto start = std::chrono::steady_clock::now();
        MSTResult res = boruvka_mst(n, edges);
        auto end = std::chrono::steady_clock::now();
        (void)res;
        return std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
    throw std::invalid_argument("Unknown algo_id");
}

int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr << "Usage: ./mst_bench <algo_id> <n> <m> <graphs_per_rep> <seed_base> <reps>\n";
        return 1;
    }
    int algo_id = std::stoi(argv[1]);
    int n = std::stoi(argv[2]);
    int m = std::stoi(argv[3]);
    int graphs_per_rep = std::stoi(argv[4]);
    uint64_t seed_base = static_cast<uint64_t>(std::stoull(argv[5]));
    int reps = std::stoi(argv[6]);

    for (int rep = 0; rep < reps; ++rep) {
        std::mt19937_64 rng(seed_base + static_cast<uint64_t>(rep));
        long long total_time_ns = 0;
        for (int g = 0; g < graphs_per_rep; ++g) {
            total_time_ns += run_single_algo(algo_id, n, m, rng);
        }
        std::cout << total_time_ns << "\n";
    }
    return 0;
}
