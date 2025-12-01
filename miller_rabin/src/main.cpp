#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

/**
 * Lightweight timer for benchmarking.
 */
class Timer {
  public:
    /**
     * Start or reset the timer.
     */
    void start() { start_time_ = std::chrono::steady_clock::now(); }

    /**
     * Return elapsed nanoseconds since last start.
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
 * Multiply two uint64_t values modulo mod using 128-bit intermediate.
 *
 * @param a Multiplicand.
 * @param b Multiplier.
 * @param mod Modulus.
 * @return uint64_t (a * b) % mod without overflow.
 */
uint64_t mulmod(uint64_t a, uint64_t b, uint64_t mod) {
    return static_cast<uint64_t>((__uint128_t(a) * __uint128_t(b)) % mod);
}

/**
 * Modular exponentiation by squaring.
 *
 * @param base Base value.
 * @param exp Exponent.
 * @param mod Modulus (must be > 0).
 * @return uint64_t base^exp % mod.
 */
uint64_t powmod(uint64_t base, uint64_t exp, uint64_t mod) {
    uint64_t result = 1 % mod;
    uint64_t p = base % mod;
    uint64_t e = exp;
    while (e > 0) {
        if (e & 1ULL) {
            result = mulmod(result, p, mod);
        }
        p = mulmod(p, p, mod);
        e >>= 1;
    }
    return result;
}

/**
 * Trial division primality test (for small bit-lengths).
 *
 * @param n Number to test.
 * @return bool true if prime else false.
 */
bool is_prime_td(uint64_t n) {
    if (n < 2) return false;
    if ((n % 2ULL) == 0ULL) return n == 2;
    for (uint64_t d = 3; d * d <= n; d += 2) {
        if (n % d == 0ULL) return false;
    }
    return true;
}

/**
 * Compute n-1 = 2^s * d with d odd.
 *
 * @param n Odd integer > 2.
 * @return std::pair<uint64_t, uint64_t> {s, d}.
 */
std::pair<uint64_t, uint64_t> factor_n_minus_one(uint64_t n) {
    uint64_t s = 0;
    uint64_t d = n - 1;
    while ((d & 1ULL) == 0ULL) {
        d >>= 1;
        ++s;
    }
    return {s, d};
}

/**
 * Single Miller–Rabin round for base 'a'.
 *
 * @param n Odd integer > 2.
 * @param a Base where 2 <= a <= n-2.
 * @param s Exponent of two in n-1.
 * @param d Odd component of n-1.
 * @return bool true if round passes, false if composite is detected.
 */
bool miller_rabin_round(uint64_t n, uint64_t a, uint64_t s, uint64_t d) {
    uint64_t x = powmod(a, d, n);
    if (x == 1ULL || x == n - 1) {
        return true;
    }
    for (uint64_t r = 1; r < s; ++r) {
        x = mulmod(x, x, n);
        if (x == n - 1) {
            return true;
        }
        if (x == 1ULL) {
            return false;
        }
    }
    return false;
}

/**
 * Deterministic Miller–Rabin for all 64-bit unsigned integers.
 *
 * @param n Number to test.
 * @return bool true if prime else false.
 */
bool is_prime_mr_det64(uint64_t n) {
    if (n < 2) return false;
    if ((n % 2ULL) == 0ULL) return n == 2;
    if (n % 3ULL == 0ULL) return n == 3;

    const uint64_t bases[] = {2ULL, 3ULL, 5ULL, 7ULL, 11ULL, 13ULL, 17ULL, 19ULL, 23ULL, 29ULL, 31ULL, 37ULL};
    auto [s, d] = factor_n_minus_one(n);
    for (uint64_t a : bases) {
        if (a % n == 0ULL) continue;
        if (!miller_rabin_round(n, a, s, d)) {
            return false;
        }
    }
    return true;
}

/**
 * Fermat primality test with k random bases.
 *
 * @param n Number to test.
 * @param k Number of rounds.
 * @param rng Random generator.
 * @return bool true if probably prime else false.
 */
bool is_probable_prime_fermat(uint64_t n, int k, std::mt19937_64& rng) {
    if (n < 4) return n == 2 || n == 3;
    if ((n & 1ULL) == 0ULL) return false;
    std::uniform_int_distribution<uint64_t> dist(2ULL, n - 2ULL);
    for (int i = 0; i < k; ++i) {
        uint64_t a = dist(rng);
        if (powmod(a, n - 1, n) != 1ULL) {
            return false;
        }
    }
    return true;
}

/**
 * Miller–Rabin primality test with k random bases.
 *
 * @param n Number to test.
 * @param k Number of rounds.
 * @param rng Random generator.
 * @return bool true if probably prime else false.
 */
bool is_probable_prime_mr(uint64_t n, int k, std::mt19937_64& rng) {
    if (n < 4) return n == 2 || n == 3;
    if ((n & 1ULL) == 0ULL) return false;
    auto [s, d] = factor_n_minus_one(n);
    std::uniform_int_distribution<uint64_t> dist(2ULL, n - 2ULL);
    for (int i = 0; i < k; ++i) {
        uint64_t a = dist(rng);
        if (!miller_rabin_round(n, a, s, d)) {
            return false;
        }
    }
    return true;
}

/**
 * Compute inclusive range for a given bit-length.
 *
 * @param bits Desired bit-length.
 * @return std::pair<uint64_t, uint64_t> {low, high}.
 */
std::pair<uint64_t, uint64_t> bit_range(int bits) {
    if (bits <= 1 || bits > 64) {
        throw std::invalid_argument("bits must be in [2,64]");
    }
    uint64_t low = bits == 64 ? (1ULL << 63) : (1ULL << (bits - 1));
    uint64_t high = bits == 64 ? UINT64_MAX : ((1ULL << bits) - 1ULL);
    return {low, high};
}

/**
 * Sample a random odd integer within the bit range.
 *
 * @param bits Desired bit-length.
 * @param rng Random generator.
 * @return uint64_t Odd integer.
 */
uint64_t sample_rand_odd(int bits, std::mt19937_64& rng) {
    auto [low, high] = bit_range(bits);
    std::uniform_int_distribution<uint64_t> dist(low, high);
    uint64_t n = dist(rng);
    n |= 1ULL;  // Force odd.
    return n;
}

/**
 * Sample composite with a small prime factor.
 *
 * @param bits Desired bit-length.
 * @param rng Random generator.
 * @return uint64_t Composite integer.
 */
uint64_t sample_comp_small_factor(int bits, std::mt19937_64& rng) {
    static const uint64_t small_primes[] = {3ULL, 5ULL, 7ULL, 11ULL, 13ULL, 17ULL, 19ULL, 23ULL, 29ULL, 31ULL};
    std::uniform_int_distribution<size_t> pick_prime(0, std::size(small_primes) - 1);
    uint64_t p = small_primes[pick_prime(rng)];
    auto [low, high] = bit_range(bits);
    uint64_t min_m = low / p + 1;
    uint64_t max_m = high / p;
    if (min_m % 2 == 0) ++min_m;
    if (max_m % 2 == 0) --max_m;
    if (min_m > max_m) {
        min_m = 3;
        max_m = 5;
    }
    std::uniform_int_distribution<uint64_t> pick_m(min_m, max_m);
    uint64_t m = pick_m(rng);
    m |= 1ULL;
    return p * m;
}

/**
 * Sample a prime using deterministic MR.
 *
 * @param bits Desired bit-length.
 * @param rng Random generator.
 * @return uint64_t Prime integer.
 */
uint64_t sample_prime_via_mr_det(int bits, std::mt19937_64& rng) {
    while (true) {
        uint64_t n = sample_rand_odd(bits, rng);
        if (is_prime_mr_det64(n)) {
            return n;
        }
    }
}

/**
 * Load Carmichael numbers from file once.
 *
 * @param path File path containing one number per line.
 * @return std::vector<uint64_t> Loaded numbers.
 */
std::vector<uint64_t> load_carmichael(const std::string& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open Carmichael file: " + path);
    }
    std::vector<uint64_t> nums;
    uint64_t x;
    while (in >> x) {
        nums.push_back(x);
    }
    if (nums.empty()) {
        throw std::runtime_error("Carmichael file is empty");
    }
    return nums;
}

/**
 * Sample a Carmichael number by bit-length, falling back to nearest available.
 *
 * @param bits Target bit-length.
 * @param rng Random generator.
 * @param pool Preloaded Carmichael numbers.
 * @return uint64_t Carmichael-like composite.
 */
uint64_t sample_carmichael(int bits, std::mt19937_64& rng, const std::vector<uint64_t>& pool) {
    std::vector<uint64_t> candidates;
    auto [low, high] = bit_range(bits);
    for (uint64_t n : pool) {
        if (n >= low && n <= high) {
            candidates.push_back(n);
        }
    }
    if (candidates.empty()) {
        candidates = pool;
    }
    std::uniform_int_distribution<size_t> dist(0, candidates.size() - 1);
    return candidates[dist(rng)];
}

/**
 * Dispatch sampler by distribution id.
 *
 * @param dist_id Distribution identifier.
 * @param bits Target bit-length.
 * @param rng Random generator.
 * @param carmichael_pool Preloaded Carmichael numbers.
 * @return uint64_t Sampled integer.
 */
uint64_t sample_number(int dist_id, int bits, std::mt19937_64& rng, const std::vector<uint64_t>& carmichael_pool) {
    switch (dist_id) {
        case 0:
            return sample_rand_odd(bits, rng);
        case 1:
            return sample_carmichael(bits, rng, carmichael_pool);
        case 2:
            return sample_comp_small_factor(bits, rng);
        case 3:
            return sample_prime_via_mr_det(bits, rng);
        default:
            throw std::invalid_argument("Invalid dist_id");
    }
}

/**
 * Run the chosen primality algorithm.
 *
 * @param algo_id Algorithm identifier.
 * @param n Number to test.
 * @param rounds Rounds parameter for Fermat/MR.
 * @param rng Random generator.
 * @return bool Result of the test: true if prime/probably prime.
 */
bool run_algo(int algo_id, uint64_t n, int rounds, std::mt19937_64& rng) {
    switch (algo_id) {
        case 0:
            return is_prime_td(n);
        case 1:
            return is_probable_prime_fermat(n, rounds, rng);
        case 2:
            return is_probable_prime_mr(n, rounds, rng);
        default:
            throw std::invalid_argument("Invalid algo_id");
    }
}

/**
 * Entry point: parse CLI, run benchmark repetitions, and emit results.
 *
 * @param argc Argument count.
 * @param argv Argument array.
 * @return int Exit code.
 */
int main(int argc, char** argv) {
    if (argc != 8) {
        std::cerr << "Usage: ./mr_bench <algo_id> <dist_id> <bits> <sample_count> <rounds> <seed_base> <reps>\n";
        return 1;
    }

    int algo_id = std::stoi(argv[1]);
    int dist_id = std::stoi(argv[2]);
    int bits = std::stoi(argv[3]);
    int sample_count = std::stoi(argv[4]);
    int rounds = std::stoi(argv[5]);
    uint64_t seed_base = static_cast<uint64_t>(std::stoull(argv[6]));
    int reps = std::stoi(argv[7]);

    const std::string exec_path = std::string(argv[0]);
    std::string::size_type last_slash = exec_path.find_last_of("/\\");
    std::string base_dir = (last_slash == std::string::npos) ? "." : exec_path.substr(0, last_slash);
    std::string carmichael_file = base_dir + "/data/carmichael_odd.txt";
    std::vector<uint64_t> carmichael_pool = load_carmichael(carmichael_file);

    for (int rep = 0; rep < reps; ++rep) {
        uint64_t seed = seed_base + static_cast<uint64_t>(rep);
        std::mt19937_64 rng(seed);
        long long error_count = 0;

        Timer timer;
        timer.start();
        for (int i = 0; i < sample_count; ++i) {
            uint64_t n = sample_number(dist_id, bits, rng, carmichael_pool);
            bool truth = is_prime_mr_det64(n);
            bool guess = run_algo(algo_id, n, rounds, rng);
            if (truth != guess) {
                ++error_count;
            }
        }
        long long elapsed = timer.elapsed_ns();
        std::cout << elapsed << " " << error_count << "\n";
    }

    return 0;
}
