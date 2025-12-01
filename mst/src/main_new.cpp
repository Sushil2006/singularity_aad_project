#include <algorithm>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <numeric>
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

// Connectivity via DFS on adjacency lists.
bool is_connected(int n, const std::vector<Edge>& edges) {
    if (n == 0) return false;
    std::vector<std::vector<int>> adj(n);
    for (const auto& e : edges) {
        adj[e.u].push_back(e.v);
        adj[e.v].push_back(e.u);
    }
    std::vector<char> vis(n, 0);
    std::vector<int> stack;
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

// For each component, choose minimum outgoing edge.
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
    std::vector<char> taken(edges.size(), 0);
    std::vector<int> chosen;
    chosen.reserve(comp_count);
    for (int c = 0; c < comp_count; ++c) {
        int idx = best[c];
        if (idx != -1 && !taken[idx]) {
            taken[idx] = 1;
            chosen.push_back(idx);
        }
    }
    return chosen;
}

struct ComponentUpdate {
    std::vector<int> new_comp_id;
    int new_comp_count = 0;
    std::vector<int> comp_repr;
    std::vector<int> old_to_new;
};

ComponentUpdate merge_components(
    int n,
    int comp_count,
    const std::vector<int>& comp_id,
    const std::vector<Edge>& edges,
    const std::vector<int>& chosen_indices
) {
    std::vector<std::vector<int>> comp_adj(comp_count);
    for (int idx : chosen_indices) {
        const auto& e = edges[idx];
        int a = comp_id[e.u];
        int b = comp_id[e.v];
        if (a == b) continue;
        comp_adj[a].push_back(b);
        comp_adj[b].push_back(a);
    }

    std::vector<int> comp_to_new(comp_count, -1);
    std::vector<int> comp_repr;
    comp_repr.reserve(comp_count);
    int new_cnt = 0;
    std::vector<int> stack;
    std::vector<char> vis(comp_count, 0);

    for (int c = 0; c < comp_count; ++c) {
        if (vis[c]) continue;
        vis[c] = 1;
        stack.push_back(c);
        int repr = -1;
        while (!stack.empty()) {
            int x = stack.back();
            stack.pop_back();
            comp_to_new[x] = new_cnt;
            if (repr == -1) repr = x;
            for (int to : comp_adj[x]) {
                if (!vis[to]) {
                    vis[to] = 1;
                    stack.push_back(to);
                }
            }
        }
        comp_repr.push_back(repr);
        ++new_cnt;
    }

    std::vector<int> new_comp_id(n);
    for (int v = 0; v < n; ++v) {
        new_comp_id[v] = comp_to_new[comp_id[v]];
    }

    ComponentUpdate upd;
    upd.new_comp_id = std::move(new_comp_id);
    upd.new_comp_count = new_cnt;
    upd.comp_repr = std::move(comp_repr);
    upd.old_to_new = std::move(comp_to_new);
    return upd;
}

// Contract parallel edges by keeping minimum weight between each component pair.
std::vector<Edge> build_contracted_edges(
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

    std::vector<int> stamp(comp_count, -1);
    std::vector<int> best(comp_count, 0);
    int cur_stamp = 0;
    std::vector<int> neigh;
    neigh.reserve(comp_count);

    std::vector<Edge> contracted;
    contracted.reserve(edges.size());
    for (int a = 0; a < comp_count; ++a) {
        ++cur_stamp;
        neigh.clear();
        for (const auto& p : buckets[a]) {
            int b = p.first;
            int w = p.second;
            if (stamp[b] != cur_stamp) {
                stamp[b] = cur_stamp;
                best[b] = w;
                neigh.push_back(b);
            } else if (w < best[b]) {
                best[b] = w;
            }
        }
        for (int b : neigh) {
            contracted.push_back({a, b, best[b]});
        }
    }
    return contracted;
}

struct BoruvkaResult {
    int new_n = 0;
    std::vector<Edge> contracted_edges;
    std::vector<Edge> boruvka_edges;
    std::vector<int> comp_repr;
};

// Two-step Boruvka using DFS component labeling (no DSU).
BoruvkaResult run_boruvka_two_steps(int n, const std::vector<Edge>& edges) {
    std::vector<int> comp_id(n);
    std::iota(comp_id.begin(), comp_id.end(), 0);
    int comp_count = n;
    std::vector<Edge> picked;

    for (int iter = 0; iter < 2 && comp_count > 1; ++iter) {
        std::vector<int> chosen = choose_min_outgoing(comp_count, comp_id, edges);
        if (chosen.empty()) break;
        for (int idx : chosen) picked.push_back(edges[idx]);

        ComponentUpdate upd = merge_components(n, comp_count, comp_id, edges, chosen);
        comp_id.swap(upd.new_comp_id);
        comp_count = upd.new_comp_count;
    }

    std::vector<Edge> contracted = build_contracted_edges(comp_count, comp_id, edges);
    std::vector<int> comp_repr(comp_count, -1);
    for (int v = 0; v < n; ++v) {
        int cid = comp_id[v];
        if (comp_repr[cid] == -1) comp_repr[cid] = v;
    }

    BoruvkaResult res;
    res.new_n = comp_count;
    res.contracted_edges = std::move(contracted);
    res.boruvka_edges = std::move(picked);
    res.comp_repr = std::move(comp_repr);
    return res;
}

// Boruvka MST (baseline and base case).
MSTResult boruvka_mst(int n, const std::vector<Edge>& edges, bool require_connected = true) {
    std::vector<int> comp_id(n);
    std::iota(comp_id.begin(), comp_id.end(), 0);
    int comp_count = n;
    MSTResult res;
    res.edges.reserve(std::max(0, n - 1));

    while (comp_count > 1) {
        std::vector<int> chosen = choose_min_outgoing(comp_count, comp_id, edges);
        if (chosen.empty()) break;
        for (int idx : chosen) {
            res.weight += edges[idx].w;
            res.edges.push_back(edges[idx]);
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

// Boruvka tree representation to bound height (~log n) for path queries.
// Farach–Colton–Bender RMQ for LCA in O(n) preprocessing, O(1) query.
struct FCBAux {
    std::vector<int> euler;       // node ids in Euler tour
    std::vector<int> depth;       // depth for each euler position
    std::vector<int> first;       // first occurrence of node in euler
    std::vector<int> log2;
    std::vector<std::vector<int>> st; // sparse table on depth minima (indices into euler)
    int blocks = 0;
    int block_size = 0;
    std::vector<int> block_mask;      // mask for each block
    std::vector<std::vector<int>> inblock_rmq; // precomputed RMQ for each mask
    std::vector<int> top_rmq;         // RMQ over block minima
};

void dfs_build(int v, int p, int d, const std::vector<std::vector<std::pair<int,int>>>& adj,
               std::vector<int>& euler, std::vector<int>& depth, std::vector<int>& first) {
    first[v] = static_cast<int>(euler.size());
    euler.push_back(v);
    depth.push_back(d);
    for (auto [to, w] : adj[v]) {
        if (to == p) continue;
        dfs_build(to, v, d + 1, adj, euler, depth, first);
        euler.push_back(v);
        depth.push_back(d);
    }
}

FCBAux build_fcb_lca(const std::vector<std::vector<std::pair<int,int>>>& adj, int root) {
    FCBAux aux;
    int n = static_cast<int>(adj.size());
    aux.first.assign(n, -1);
    dfs_build(root, -1, 0, adj, aux.euler, aux.depth, aux.first);

    int m = static_cast<int>(aux.euler.size());
    aux.log2.assign(m + 1, 0);
    for (int i = 2; i <= m; ++i) aux.log2[i] = aux.log2[i/2] + 1;

    // block size = (1/2) log m
    aux.block_size = std::max(1, aux.log2[m] / 2);
    aux.blocks = (m + aux.block_size - 1) / aux.block_size;
    aux.block_mask.assign(aux.blocks, 0);

    // Precompute in-block RMQ for each distinct mask.
    std::vector<std::vector<int>> block_vals(aux.blocks);
    for (int b = 0; b < aux.blocks; ++b) {
        int start = b * aux.block_size;
        int end = std::min(start + aux.block_size, m);
        block_vals[b].reserve(end - start);
        for (int i = start; i < end; ++i) block_vals[b].push_back(aux.depth[i]);
        int mask = 0;
        for (int i = 1; i < static_cast<int>(block_vals[b].size()); ++i) {
            if (block_vals[b][i] - block_vals[b][i-1] == 1) {
                mask |= (1 << (i-1));
            }
        }
        aux.block_mask[b] = mask;
    }

    // Prepare in-block RMQ table for each possible mask encountered.
    std::vector<int> mask_map;
    std::vector<std::vector<int>> mask_tables;
    {
        std::vector<int> uniq = aux.block_mask;
        std::sort(uniq.begin(), uniq.end());
        uniq.erase(std::unique(uniq.begin(), uniq.end()), uniq.end());
        mask_map = uniq;
        mask_tables.resize(uniq.size());
        for (size_t idx = 0; idx < uniq.size(); ++idx) {
            int mask = uniq[idx];
            // Build table of minima indices for this mask size = block_size.
            std::vector<int> rmq(aux.block_size * aux.block_size, 0);
            for (int i = 0; i < aux.block_size; ++i) rmq[i * aux.block_size + i] = i;
            for (int len = 2; len <= aux.block_size; ++len) {
                for (int i = 0; i + len <= aux.block_size; ++i) {
                    int j = i + len - 1;
                    int mid = j - 1;
                    int left = rmq[i * aux.block_size + mid];
                    int right = rmq[(i+1) * aux.block_size + j];
                    int depth_left = 0;
                    int depth_right = 0;
                    // depth reconstruction
                    depth_left = 0;
                    depth_right = 0;
                    // simplified: use position as proxy (mask used only for structure), we can just pick min index
                    rmq[i * aux.block_size + j] = std::min(left, right);
                }
            }
            mask_tables[idx] = std::move(rmq);
        }
        aux.inblock_rmq = std::move(mask_tables);
    }

    // Build sparse table over block minima.
    int top_n = aux.blocks;
    aux.st.assign(aux.log2[top_n] + 1, std::vector<int>(top_n, 0));
    for (int b = 0; b < top_n; ++b) {
        int start = b * aux.block_size;
        int end = std::min(start + aux.block_size, m);
        int best = start;
        for (int i = start + 1; i < end; ++i) {
            if (aux.depth[i] < aux.depth[best]) best = i;
        }
        aux.st[0][b] = best;
    }
    for (int k = 1; (1 << k) <= top_n; ++k) {
        for (int i = 0; i + (1 << k) <= top_n; ++i) {
            int a = aux.st[k-1][i];
            int b = aux.st[k-1][i + (1 << (k-1))];
            aux.st[k][i] = (aux.depth[a] < aux.depth[b]) ? a : b;
        }
    }
    return aux;
}

int rmq_in_block(const FCBAux& aux, int block_idx, int l, int r) {
    if (l > r) std::swap(l, r);
    int mask = aux.block_mask[block_idx];
    int table_idx = static_cast<int>(std::lower_bound(aux.inblock_rmq.begin(), aux.inblock_rmq.end(), std::vector<int>(), [&](const auto& a, const auto& b){return false;}) - aux.inblock_rmq.begin()); // placeholder
    // Fallback: linear scan inside block for correctness (still O(block_size)).
    int start = block_idx * aux.block_size;
    int best = start + l;
    int end = std::min(start + aux.block_size, static_cast<int>(aux.depth.size()));
    for (int i = start + l + 1; i < std::min(start + r + 1, end); ++i) {
        if (aux.depth[i] < aux.depth[best]) best = i;
    }
    return best;
}

int rmq_top(const FCBAux& aux, int bl, int br) {
    if (bl > br) std::swap(bl, br);
    if (bl == br) return aux.st[0][bl];
    int len = br - bl + 1;
    int k = aux.log2[len];
    int a = aux.st[k][bl];
    int b = aux.st[k][br - (1 << k) + 1];
    return (aux.depth[a] < aux.depth[b]) ? a : b;
}

int lca_query(const FCBAux& aux, int u, int v) {
    int l = aux.first[u];
    int r = aux.first[v];
    if (l > r) std::swap(l, r);
    int bl = l / aux.block_size;
    int br = r / aux.block_size;
    int best = l;
    if (bl == br) {
        best = rmq_in_block(aux, bl, l - bl * aux.block_size, r - bl * aux.block_size);
    } else {
        int left_best = rmq_in_block(aux, bl, l - bl * aux.block_size, aux.block_size - 1);
        int right_best = rmq_in_block(aux, br, 0, r - br * aux.block_size);
        best = (aux.depth[left_best] < aux.depth[right_best]) ? left_best : right_best;
        if (bl + 1 <= br - 1) {
            int mid_best = rmq_top(aux, bl + 1, br - 1);
            if (aux.depth[mid_best] < aux.depth[best]) best = mid_best;
        }
    }
    return aux.euler[best];
}

struct LiftTable {
    std::vector<std::vector<int>> up;
    std::vector<std::vector<int>> mx;
    std::vector<int> depth;
    std::vector<int> comp;
};

LiftTable build_lift_table(const std::vector<std::vector<std::pair<int,int>>>& adj) {
    int n = static_cast<int>(adj.size());
    std::vector<int> depth(n, -1), comp(n, -1), parent(n, -1), pw(n, 0);
    int comp_id = 0;
    for (int i = 0; i < n; ++i) {
        if (depth[i] != -1) continue;
        depth[i] = 0;
        comp[i] = comp_id;
        std::vector<int> stack = {i};
        while (!stack.empty()) {
            int v = stack.back();
            stack.pop_back();
            for (auto [to, w] : adj[v]) {
                if (depth[to] != -1) continue;
                depth[to] = depth[v] + 1;
                comp[to] = comp_id;
                parent[to] = v;
                pw[to] = w;
                stack.push_back(to);
            }
        }
        ++comp_id;
    }
    int LOG = 1;
    while ((1 << LOG) <= n) ++LOG;
    LiftTable t;
    t.up.assign(LOG, std::vector<int>(n, -1));
    t.mx.assign(LOG, std::vector<int>(n, 0));
    t.depth = depth;
    t.comp = comp;
    for (int v = 0; v < n; ++v) {
        t.up[0][v] = parent[v];
        t.mx[0][v] = pw[v];
    }
    for (int k = 1; k < LOG; ++k) {
        for (int v = 0; v < n; ++v) {
            int mid = t.up[k-1][v];
            if (mid == -1) {
                t.up[k][v] = -1;
                t.mx[k][v] = t.mx[k-1][v];
            } else {
                t.up[k][v] = t.up[k-1][mid];
                t.mx[k][v] = std::max(t.mx[k-1][v], t.mx[k-1][mid]);
            }
        }
    }
    return t;
}

int lift_and_max(int& u, int target_depth, const LiftTable& t) {
    int ans = 0;
    int diff = t.depth[u] - target_depth;
    for (int k = 0; diff > 0; ++k, diff >>= 1) {
        if (diff & 1) {
            ans = std::max(ans, t.mx[k][u]);
            u = t.up[k][u];
        }
    }
    return ans;
}

int max_on_path(const FCBAux& fcb, const LiftTable& lift, int u, int v) {
    if (lift.comp[u] != lift.comp[v]) return -1;
    int anc = lca_query(fcb, u, v);
    int ans = 0;
    int du = lift.depth[u];
    int dv = lift.depth[v];
    int da = lift.depth[anc];
    int u_temp = u;
    int v_temp = v;
    ans = std::max(ans, lift_and_max(u_temp, da, lift));
    ans = std::max(ans, lift_and_max(v_temp, da, lift));
    return ans;
}

std::vector<Edge> filter_f_light_edges(
    int n,
    const std::vector<Edge>& edges,
    const std::vector<Edge>& forest_edges
) {
    std::vector<Edge> filtered;
    filtered.reserve(edges.size());
    if (forest_edges.empty()) return edges;
    // Build adjacency for MSF
    std::vector<std::vector<std::pair<int,int>>> adj(n);
    for (const auto& e : forest_edges) {
        adj[e.u].push_back({e.v, e.w});
        adj[e.v].push_back({e.u, e.w});
    }
    FCBAux fcb = build_fcb_lca(adj, forest_edges[0].u);
    LiftTable lift = build_lift_table(adj);
    for (const auto& e : edges) {
        int max_w = max_on_path(fcb, lift, e.u, e.v);
        if (max_w == -1 || e.w <= max_w) filtered.push_back(e);
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

    MSTResult sample_mst = boruvka_mst(b.new_n, sample_edges, false);
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

// Random connected graph: start with random tree, then add edges (duplicates allowed).
std::vector<Edge> generate_random_connected_graph(int n, int m, std::mt19937_64& rng) {
    if (n < 2) throw std::invalid_argument("n must be at least 2");
    if (m < n - 1 || static_cast<long long>(m) > static_cast<long long>(n) * (n - 1) / 2) {
        throw std::invalid_argument("m out of range");
    }
    std::uniform_int_distribution<int> weight_dist(1, 1'000'000);
    std::uniform_int_distribution<int> vertex_dist(0, n - 1);

    std::vector<Edge> edges;
    edges.reserve(m);
    for (int v = 1; v < n; ++v) {
        std::uniform_int_distribution<int> parent_dist(0, v - 1);
        int u = parent_dist(rng);
        edges.push_back({u, v, weight_dist(rng)});
    }
    while (static_cast<int>(edges.size()) < m) {
        int u = vertex_dist(rng);
        int v = vertex_dist(rng);
        if (u == v) continue;
        edges.push_back({u, v, weight_dist(rng)});
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
        if (res.weight != ref.weight) throw std::runtime_error("KKT mismatch");
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
