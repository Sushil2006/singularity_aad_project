#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

/**
 * Global counter for swap operations performed by instrumented algorithms.
 */
long long g_swap_count = 0;

/**
 * Swap two integers and increment the global swap counter.
 *
 * @param a Reference to first integer.
 * @param b Reference to second integer.
 */
inline void counted_swap(int& a, int& b) {
    g_swap_count++;
    std::swap(a, b);
}

/**
 * Simple timer utility that measures elapsed time in nanoseconds.
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
 * Check whether a vector is sorted in non-decreasing order.
 *
 * @param a Reference to the vector to check.
 * @return bool true if sorted, else false.
 */
bool is_sorted_non_decreasing(const std::vector<int>& a) {
    for (size_t i = 1; i < a.size(); ++i) {
        if (a[i - 1] > a[i]) {
            return false;
        }
    }
    return true;
}

/**
 * Perform insertion sort on a half-open interval [lo, hi] of the array.
 *
 * @param a Vector to sort.
 * @param lo Inclusive lower index.
 * @param hi Inclusive upper index.
 */
void insertion_sort(std::vector<int>& a, int lo, int hi) {
    for (int i = lo + 1; i <= hi; ++i) {
        int key = a[i];
        int j = i - 1;
        while (j >= lo && a[j] > key) {
            a[j + 1] = a[j];
            --j;
        }
        a[j + 1] = key;
    }
}

/**
 * Partition the array segment around a randomly chosen pivot and return its final index.
 *
 * @param a Vector being sorted.
 * @param lo Inclusive lower bound.
 * @param hi Inclusive upper bound.
 * @param rng Random number generator used for pivot selection.
 * @return int Index of pivot after partitioning.
 */
int partition_range(std::vector<int>& a, int lo, int hi, std::mt19937_64& rng) {
    std::uniform_int_distribution<int> dist(lo, hi);
    int pivot_index = dist(rng);
    counted_swap(a[pivot_index], a[hi]);  // Move pivot to end for convenience.
    int pivot_value = a[hi];
    int store = lo;
    for (int i = lo; i < hi; ++i) {
        if (a[i] < pivot_value) {
            counted_swap(a[i], a[store]);
            ++store;
        }
    }
    counted_swap(a[store], a[hi]);
    return store;
}

/**
 * Vanilla QuickSort using straightforward recursion on both subarrays.
 *
 * @param a Vector to sort.
 * @param lo Inclusive lower bound.
 * @param hi Inclusive upper bound.
 * @param rng Random generator for pivot selection.
 * @param cutoff Cutoff size for insertion sort (0 disables cutoff).
 */
void quicksort_recursive(std::vector<int>& a, int lo, int hi, std::mt19937_64& rng, int cutoff) {
    if (lo >= hi) {
        return;
    }
    if (cutoff > 0 && (hi - lo + 1) <= cutoff) {
        insertion_sort(a, lo, hi);
        return;
    }
    int p = partition_range(a, lo, hi, rng);
    quicksort_recursive(a, lo, p - 1, rng, cutoff);
    quicksort_recursive(a, p + 1, hi, rng, cutoff);
}

/**
 * Tail-recursive QuickSort that recurses on the smaller partition and loops on the larger one.
 *
 * @param a Vector to sort.
 * @param lo Inclusive lower bound.
 * @param hi Inclusive upper bound.
 * @param rng Random generator for pivot selection.
 * @param cutoff Cutoff size for insertion sort (0 disables cutoff).
 */
void quicksort_tail_opt(std::vector<int>& a, int lo, int hi, std::mt19937_64& rng, int cutoff) {
    while (lo < hi) {
        int size = hi - lo + 1;
        if (cutoff > 0 && size <= cutoff) {
            insertion_sort(a, lo, hi);
            return;
        }
        int p = partition_range(a, lo, hi, rng);
        int left_size = p - lo;
        int right_size = hi - p;

        if (left_size < right_size) {
            if (left_size > 0) {
                quicksort_tail_opt(a, lo, p - 1, rng, cutoff);
            }
            lo = p + 1;  // Tail-call elimination on the larger side.
        } else {
            if (right_size > 0) {
                quicksort_tail_opt(a, p + 1, hi, rng, cutoff);
            }
            hi = p - 1;
        }
    }
}

/**
 * Public wrapper for vanilla QuickSort (no cutoff, standard recursion).
 *
 * @param a Vector to sort in place.
 * @param rng Random generator for pivot selection.
 */
void quicksort_vanilla(std::vector<int>& a, std::mt19937_64& rng) {
    if (a.empty()) {
        return;
    }
    quicksort_recursive(a, 0, static_cast<int>(a.size()) - 1, rng, 0);
}

/**
 * QuickSort variant that recurses on the smaller partition to keep stack depth O(log n).
 *
 * @param a Vector to sort in place.
 * @param rng Random generator for pivot selection.
 * @param cutoff Cutoff size for insertion sort; use 0 to disable.
 */
void quicksort_smaller_subtree(std::vector<int>& a, std::mt19937_64& rng, int cutoff) {
    if (a.empty()) {
        return;
    }
    quicksort_tail_opt(a, 0, static_cast<int>(a.size()) - 1, rng, cutoff);
}

/**
 * QuickSort variant with cutoff = 16 that reuses the smaller-subtree strategy.
 *
 * @param a Vector to sort in place.
 * @param rng Random generator for pivot selection.
 */
void quicksort_cutoff16(std::vector<int>& a, std::mt19937_64& rng) {
    quicksort_smaller_subtree(a, rng, 16);
}

/**
 * Merge function for merge sort using a preallocated buffer.
 *
 * @param a Vector being sorted.
 * @param buffer Temporary buffer of the same size.
 * @param lo Inclusive lower index.
 * @param mid Middle index (end of left half).
 * @param hi Inclusive upper index.
 */
void merge(std::vector<int>& a, std::vector<int>& buffer, int lo, int mid, int hi) {
    int i = lo;
    int j = mid + 1;
    int k = lo;
    while (i <= mid && j <= hi) {
        if (a[i] <= a[j]) {
            buffer[k++] = a[i++];
        } else {
            buffer[k++] = a[j++];
        }
    }
    while (i <= mid) {
        buffer[k++] = a[i++];
    }
    while (j <= hi) {
        buffer[k++] = a[j++];
    }
    for (int idx = lo; idx <= hi; ++idx) {
        a[idx] = buffer[idx];
    }
}

/**
 * Recursive helper for merge sort.
 *
 * @param a Vector to sort.
 * @param buffer Temporary buffer reused across calls.
 * @param lo Inclusive lower index.
 * @param hi Inclusive upper index.
 */
void mergesort_impl(std::vector<int>& a, std::vector<int>& buffer, int lo, int hi) {
    if (lo >= hi) {
        return;
    }
    int mid = lo + (hi - lo) / 2;
    mergesort_impl(a, buffer, lo, mid);
    mergesort_impl(a, buffer, mid + 1, hi);
    merge(a, buffer, lo, mid, hi);
}

/**
 * Public entry for merge sort.
 *
 * @param a Vector to sort in place.
 */
void mergesort_int(std::vector<int>& a) {
    if (a.empty()) {
        return;
    }
    std::vector<int> buffer(a.size());
    mergesort_impl(a, buffer, 0, static_cast<int>(a.size()) - 1);
}

/**
 * Wrapper for std::sort baseline.
 *
 * @param a Vector to sort in place.
 */
void std_sort_int(std::vector<int>& a) {
    std::sort(a.begin(), a.end());
}

/**
 * Generate a sorted vector of size n with values 0..n-1.
 *
 * @param n Desired vector size.
 * @return std::vector<int> Sorted vector.
 */
std::vector<int> generate_sorted(int n) {
    std::vector<int> a(n);
    for (int i = 0; i < n; ++i) {
        a[i] = i;
    }
    return a;
}

/**
 * Generate an almost-sorted vector with 5% random swaps applied to a sorted baseline.
 *
 * @param n Desired vector size.
 * @param seed Random seed for reproducibility.
 * @return std::vector<int> Almost-sorted vector.
 */
std::vector<int> generate_almost_sorted(int n, uint64_t seed) {
    std::vector<int> a = generate_sorted(n);
    int k = static_cast<int>(std::floor(0.05 * n));
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> dist(0, n - 1);
    for (int iter = 0; iter < k; ++iter) {
        int i = dist(rng);
        int j = dist(rng);
        std::swap(a[i], a[j]);
    }
    return a;
}

/**
 * Generate a vector with uniform integer values in [LOW, HIGH].
 *
 * @param n Desired vector size.
 * @param seed Random seed for reproducibility.
 * @return std::vector<int> Vector filled with uniform random integers.
 */
std::vector<int> generate_uniform(int n, uint64_t seed) {
    constexpr int LOW = -1000000000;
    constexpr int HIGH = 1000000000;
    std::vector<int> a(n);
    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<int> dist(LOW, HIGH);
    for (int i = 0; i < n; ++i) {
        a[i] = dist(rng);
    }
    return a;
}

/**
 * Generate a vector with integer values drawn from a normal distribution.
 *
 * @param n Desired vector size.
 * @param seed Random seed for reproducibility.
 * @return std::vector<int> Vector filled with normally distributed integers.
 */
std::vector<int> generate_normal(int n, uint64_t seed) {
    constexpr int LOW = -1000000000;
    constexpr int HIGH = 1000000000;
    std::vector<int> a(n);
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> dist(0.0, 1000.0);
    for (int i = 0; i < n; ++i) {
        int v = static_cast<int>(std::llround(dist(rng)));
        if (v < LOW) v = LOW;
        if (v > HIGH) v = HIGH;
        a[i] = v;
    }
    return a;
}

/**
 * Load the first n integer prices from the stock data file.
 *
 * @param n Number of values to read.
 * @param stock_path Path to the space-separated integer file.
 * @return std::vector<int> Vector of length n with stock prices.
 */
std::vector<int> generate_stock(int n, const std::string& stock_path) {
    std::ifstream in(stock_path);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open stock data file: " + stock_path);
    }
    std::vector<int> a;
    a.reserve(n);
    int value;
    while (static_cast<int>(a.size()) < n && in >> value) {
        a.push_back(value);
    }
    if (static_cast<int>(a.size()) < n) {
        throw std::runtime_error("Stock data file has fewer than n values");
    }
    return a;
}

/**
 * Generate a vector according to the requested distribution.
 *
 * @param dist_id Distribution identifier.
 * @param n Desired vector size.
 * @param seed Seed used for pseudo-random generators (ignored for stock).
 * @param stock_path Path to stock data file for dist_id == 4.
 * @return std::vector<int> Generated vector.
 */
std::vector<int> generate_data(int dist_id, int n, uint64_t seed, const std::string& stock_path) {
    switch (dist_id) {
        case 0:
            return generate_sorted(n);
        case 1:
            return generate_almost_sorted(n, seed);
        case 2:
            return generate_uniform(n, seed);
        case 3:
            return generate_normal(n, seed);
        case 4:
            return generate_stock(n, stock_path);
        default:
            throw std::invalid_argument("Invalid dist_id");
    }
}

/**
 * Dispatch to the selected sorting algorithm with instrumentation.
 *
 * @param algo_id Algorithm identifier.
 * @param a Vector to sort.
 * @param seed Seed used for algorithmic randomness.
 */
void run_algorithm(int algo_id, std::vector<int>& a, uint64_t seed) {
    std::mt19937_64 rng(seed);
    switch (algo_id) {
        case 0:
            quicksort_vanilla(a, rng);
            break;
        case 1:
            quicksort_smaller_subtree(a, rng, 0);
            break;
        case 2:
            quicksort_cutoff16(a, rng);
            break;
        case 3:
            mergesort_int(a);
            break;
        case 4:
            g_swap_count = -1;  // Sentinel value; std::sort not instrumented.
            std_sort_int(a);
            break;
        default:
            throw std::invalid_argument("Invalid algo_id");
    }
}

/**
 * Entry point: parse CLI arguments and run benchmarks.
 *
 * @param argc Argument count.
 * @param argv Argument vector.
 * @return int Process exit code (0 on success, non-zero on error).
 */
int main(int argc, char** argv) {
    if (argc != 7) {
        std::cerr << "Usage: ./quicksort_bench <algo_id> <dist_id> <n> <seed_base> <reps> <stock_path>\n";
        return 1;
    }

    int algo_id = std::stoi(argv[1]);
    int dist_id = std::stoi(argv[2]);
    int n = std::stoi(argv[3]);
    uint64_t seed_base = static_cast<uint64_t>(std::stoull(argv[4]));
    int reps = std::stoi(argv[5]);
    std::string stock_path = argv[6];

    const bool kVerifySorted = false;  // Enable for debugging correctness.

    for (int rep = 0; rep < reps; ++rep) {
        uint64_t seed = seed_base + static_cast<uint64_t>(rep);
        std::vector<int> data = generate_data(dist_id, n, seed, stock_path);

        g_swap_count = 0;
        Timer timer;
        timer.start();
        run_algorithm(algo_id, data, seed);
        long long elapsed = timer.elapsed_ns();

        if (kVerifySorted && !is_sorted_non_decreasing(data)) {
            std::cerr << "Validation failed: output not sorted for rep " << rep << "\n";
            return 2;
        }

        std::cout << elapsed << " " << g_swap_count << "\n";
    }

    return 0;
}
