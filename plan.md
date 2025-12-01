# Quicksort:

- **1. Algorithmic analysis and variants (to be described in report)**

  - **1.1 QuickSort baseline (vanilla QS)**

    - Input: `std::vector<int>& a`.
    - Pivot: index chosen uniformly at random in `[lo, hi]` using a fixed RNG.
    - Partition: in-place 2-way partition around pivot (elements `< pivot` and `>= pivot`).
    - Recursion: recursively sort `[lo, p-1]` and `[p+1, hi]` in any fixed order.
    - Complexity:

      - Worst-case time: `Θ(n²)` if partitions are highly unbalanced (e.g., already sorted + unlucky pivots).
      - Expected time: `Θ(n log n)` under uniform random pivot selection.
      - Best-case time: `Θ(n log n)` for balanced partitions.
      - Extra space: `O(1)` auxiliary + `O(n)` recursion stack in the worst case.

  - **1.2 QuickSort with recursion on smaller subtree (QS-small)**

    - Same as vanilla QS except recursion order:

      - After partitioning into `[lo, p-1]` and `[p+1, hi]`, identify smaller segment `S` and larger segment `L`.
      - Recursively call QS-small on `S`.
      - Replace recursion on `L` by a loop that updates `(lo, hi)` to cover `L`.

    - Stack-depth analysis:

      - Let `D(n)` be recursion depth.
      - Each recursive call reduces `n` to at most `(n−1)/2` (size of smaller side).
      - Recurrence: `D(n) ≤ 1 + D(⌊(n−1)/2⌋)` ⇒ `D(n) = O(log n)` for all inputs.
      - Time complexity: remains `Θ(n log n)` expected, `Θ(n²)` worst-case; only stack usage changes from `O(n)` to `O(log n)`.

  - **1.3 QuickSort with cutoff = 16 (QS-cut16)**

    - Same as vanilla QS, except:

      - If subarray size `(hi − lo + 1) ≤ 16`, do not partition further.
      - After top-level QS-cut16 completes, perform a final insertion sort over the entire array, or sort small segments immediately when threshold reached (pick one; document choice).

    - Complexity sketch:

      - Partitioning cost down to size 16 is `Θ(n log(n/16)) = Θ(n log n)`.
      - Total cost of insertion sorts over all small segments is `O(n * 16) = O(n)`.
      - Overall time: still `Θ(n log n)` with smaller constants than vanilla QS in practice.
      - Stack usage: same as vanilla QS (`O(n)` worst-case) unless combined with “smaller-subtree” optimization (in this spec, we treat QS-cut16 as separate from QS-small).

  - **1.4 Merge Sort (MS)**

    - Standard top-down merge sort on `std::vector<int>`.
    - Recursively split into halves, merge two sorted halves using a temporary vector.
    - Time: `Θ(n log n)` in all cases.
    - Extra space: `Θ(n)` for the temporary buffer.

  - **1.5 `std::sort` (STD)**

    - Uses C++ STL `std::sort` on `std::vector<int>`.
    - Typical library implementation is introsort: QuickSort + HeapSort + Insertion Sort; worst-case `O(n log n)`.
    - No control over internal behavior; used as a realistic baseline.

---

- **2. Objectives and scope**

  - Benchmark 5 algorithms on integer data:

    - A0: vanilla QuickSort (QS).
    - A1: QuickSort with recursion on smaller subtree (QS-small).
    - A2: QuickSort with cutoff 16 (QS-cut16).
    - A3: Merge Sort (MS).
    - A4: `std::sort` (STD).

  - Distributions:

    - `sorted` ascending.
    - `almost_sorted` (5% random swaps).
    - `uniform` random.
    - `normal` distribution (integers).
    - `stock` (NIFTY 1-minute close prices, real-world).

  - Metrics collected per run (from C++):

    - Wall-clock time in nanoseconds (`time_ns`).
    - Number of swaps (`swaps`) for A0–A3 (A4 uses sentinel value, e.g. `-1`).

  - End products:

    - CSV file with all runs.
    - Plots comparing time vs `n` and swaps vs `n` for each distribution and algorithm.
    - Short interpretation in report: relationship between theory (Section 1) and observed performance.

---

- **3. Datasets and distributions**

  - **3.1 Real-world stock data (stock)**

    - Source: Kaggle “Indian Stock Market Index Intraday Data (2008–2020)”.
    - Preprocessing (Python, one-time):

      - Filter to NIFTY 1-minute data.
      - Sort by timestamp.
      - Extract `Close` column and convert to integer: `price_int = round(close * 100)`.
      - Take first 1,000,000 values.
      - Write to `quicksort/data/nifty_1m_int_1M.txt` as space-separated integers (no header).

    - Use in C++:

      - For `stock` distribution and size `n`, read first `n` integers from `nifty_1m_int_1M.txt` into `std::vector<int>`.

  - **3.2 Synthetic distributions (deterministic given `(n, seed)`)**

    - `sorted`:

      - `a[i] = i` for `0 ≤ i < n`.

    - `almost_sorted`:

      - Start from `sorted`.
      - Let `k = floor(0.05 * n)`.
      - Perform `k` iterations: select two indices `i,j` uniformly at random in `[0, n-1]` and swap `a[i], a[j]`.

    - `uniform`:

      - Use `std::mt19937_64 rng(seed)` and `std::uniform_int_distribution<int>(LOW, HIGH)` with fixed `LOW = -1e9`, `HIGH = 1e9`.
      - Fill `a[i]` with draws from this distribution.

    - `normal`:

      - Use `std::mt19937_64 rng(seed)` and `std::normal_distribution<double>(0.0, 1000.0)`.
      - For each draw `x`, set `a[i] = (int)std::round(x)`; optionally clamp to `[LOW, HIGH]`.

    - All synthetic generators must be pure functions of `(n, seed)` to make experiments reproducible.

---

- **4. Single C++ benchmark program (one `.cpp` file)**

  - **4.1 File and build**

    - File: `quicksort/src/main.cpp`.
    - Contains:

      - Implementations of A0–A4.
      - Timer utility class (using `std::chrono::steady_clock`).
      - Distribution generators.
      - `main()` which acts as benchmark driver.

    - Build example (for README):

      - `g++ -O2 -std=c++17 src/main.cpp -o quicksort_bench`.

  - **4.2 Algorithm interfaces (inside `main.cpp`)**

    - Data type: `std::vector<int> a`.
    - Function signatures (internal):

      - `void quicksort_vanilla(std::vector<int>& a);` // A0
      - `void quicksort_smaller_subtree(std::vector<int>& a);` // A1
      - `void quicksort_cutoff16(std::vector<int>& a);` // A2
      - `void mergesort_int(std::vector<int>& a);` // A3
      - `void std_sort_int(std::vector<int>& a);` // A4

    - Global instrumentation (or struct passed by reference):

      - `long long g_swap_count;` set to `0` before each algorithm call.
      - A4 (`std::sort`) sets `g_swap_count = -1` or leaves it unchanged; Python script handles this.

  - **4.3 Command-line interface for `main()`**

    - Binary: `./quicksort_bench`.
    - Arguments (in fixed order):

      - `<algo_id>`: integer in `{0,1,2,3,4}` for `{QS, QS-small, QS-cut16, MS, STD}`.
      - `<dist_id>`: integer in `{0,1,2,3,4}` for `{sorted, almost_sorted, uniform, normal, stock}`.
      - `<n>`: integer, problem size.
      - `<seed_base>`: 64-bit seed base.
      - `<reps>`: integer ≥ 1, number of repetitions.
      - `<stock_path>`: path to `nifty_1m_int_1M.txt` (used only if `dist_id == 4`).

  - **4.4 Runtime behavior**

    - For `rep` from `0` to `reps−1`:

      - Compute `seed = seed_base + rep`.
      - Generate base vector `a` of size `n` using `(dist_id, n, seed)`; for `stock`, ignore `seed` and read from `stock_path`.
      - Reset `g_swap_count = 0`.
      - Start timer.
      - Dispatch to algorithm specified by `algo_id` on `a`.
      - Stop timer.
      - Optional: verify `a` is sorted (can be disabled in final runs).
      - Print a single line to `stdout`:

        - `<time_ns> <swaps>` with a single space separator.

    - No extra output, logging, or prompts; exactly `reps` lines are printed.

---

- **5. Python scripts and visualization workflow**

  - **5.1 `prepare_stock_data.py`**

    - Input: Kaggle CSV files for NIFTY intraday data.
    - Output: `quicksort/data/nifty_1m_int_1M.txt` as specified.
    - Run once before benchmarks.

  - **5.2 `run_benchmarks.py`**

    - Configuration at top:

      - `algos = ["QS", "QS_small", "QS_cut16", "MS", "STD"]` with mapping to `{0..4}`.
      - `dists = ["sorted", "almost_sorted", "uniform", "normal", "stock"]` with mapping to `{0..4}`.
      - `sizes = [1_000, 10_000, 100_000, 1_000_000]` (or final chosen list).
      - `reps = R` (e.g., 5 or 10).
      - `seed_base_global` as fixed integer.

    - Loop over `(algo, dist, n)` and call:

      - `./quicksort_bench algo_id dist_id n seed_base reps path/to/nifty_1m_int_1M.txt`.

    - For each call:

      - Read `reps` lines from `stdout`.
      - For each line, parse `time_ns` and `swaps`.
      - Append one CSV row per line to `quicksort/results/raw_results.csv` with columns:

        - `algo, dist, n, rep, time_ns, swaps`.

  - **5.3 `plot_results.py`**

    - Input: `quicksort/results/raw_results.csv`.
    - Aggregation:

      - Group by `(algo, dist, n)`.
      - Compute `mean_time_ms = mean(time_ns / 1e6)`, `std_time_ms`, `mean_swaps` (ignore `swaps == -1`).

    - Required visualizations (for report):

      - For each `dist` separately:

        - Line plot, x-axis = `n` (log scale), y-axis = `mean_time_ms`.
        - One line per algorithm (5 curves).
        - Optional error bars for `std_time_ms`.

      - For a fixed large `n` (e.g., `1_000_000`):

        - Bar chart with x-axis = algorithms, y-axis = `mean_time_ms`, one chart per distribution.

      - Optional diagnostic plots:

        - For QuickSort variants and Merge Sort, line plot of `mean_swaps` vs `n` for each distribution.

    - Outputs: save plots as `PNG` or `PDF` to `quicksort/results/plots/` with clear filenames (e.g., `time_vs_n_uniform.png`).

---

- **6. Points to cover in “Why QuickSort is good for real-world data” section of report**

  - In-place nature: `O(1)` extra memory makes it suitable for large in-memory arrays compared to `O(n)`-space Merge Sort.
  - Cache behavior: partition-based scanning over contiguous `std::vector<int>` improves cache locality; fewer passes over data than some alternatives.
  - Randomization: expected `Θ(n log n)` performance is robust across non-adversarial inputs, which is typical for real market data and logs.
  - Empirical dominance: standard library implementations for general-purpose sorting (e.g., `std::sort`) rely on QuickSort-based hybrids, indicating strong practical performance.
  - Connection to results: relate the above points to observed performance on `stock` data and synthetic distributions in the analysis section.

---

- **7. Directory structure (`quicksort/`)**

  - `quicksort/README.md`

    - Purpose of module.
    - How to build `quicksort_bench`.
    - How to run `prepare_stock_data.py`, `run_benchmarks.py`, `plot_results.py`.

  - `quicksort/src/`

    - `main.cpp`

      - All algorithm implementations (A0–A4).
      - Distribution generators.
      - `main()` and Timer class.

  - `quicksort/data/`

    - `nifty_1m_int_1M.txt`

      - Preprocessed stock prices as integers (space-separated).

  - `quicksort/scripts/`

    - `prepare_stock_data.py`
    - `run_benchmarks.py`
    - `plot_results.py`

  - `quicksort/results/`

    - `raw_results.csv`
    - `plots/` (all generated figures for the report).

---

# Miller Rabin:

- **1. Algorithmic analysis and proof-flow (for report)**

  - **1.1 Problem definition**

    - Input: integer (n \ge 2); in implementation we restrict to `uint64_t`.
    - Task: decide whether (n) is composite or prime / “probably prime”.
    - Computational model: 64-bit word RAM; modular multiplication done via 128-bit intermediate (`__uint128_t`) to avoid overflow. ([Codeforces][1])

  - **1.2 Preprocessing step for odd (n)**

    - For odd (n>2), write (n-1 = 2^s \cdot d) with (d) odd.
    - This factorization is unique and is reused across all bases for a given (n). ([Codeforces][1])

  - **1.3 Single-round Miller–Rabin test for base (a)** ([Codeforces][1])

    - Preconditions: (n) odd, (n>2), base (a) with (2 \le a \le n-2) and (\gcd(a,n)=1).
    - Steps:

      - Compute (x = a^d \bmod n) using fast exponentiation by repeated squaring.
      - If (x = 1) or (x = n-1), the round **passes**.
      - Otherwise, for (r = 1,\dots,s-1):

        - Set (x = x^2 \bmod n).
        - If (x = n-1), round passes.
        - If (x = 1) before reaching (n-1), declare (n) composite (non-trivial square root of 1).

      - If loop ends without seeing (n-1), declare (n) composite.

  - **1.4 k-round randomized MR (MR-k)** ([Codeforces][1])

    - Repeat the single-round test for (k) independently chosen random bases (a_1,\dots,a_k) in ({2,\dots,n-2}).
    - If any round returns “composite” → output “composite”.
    - If all rounds pass → output “probably prime”.

  - **1.5 Correctness & error bound – flow of the proof** ([Codeforces][1])

    - Define “strong probable prime to base (a)” via the MR congruences above.
    - For **prime** (n):

      - Work in group ((\mathbb{Z}/n\mathbb{Z})^\times).
      - Show the only square roots of 1 mod (n) are (\pm1).
      - Show that for all bases (a) coprime to (n), the MR sequence must hit 1 or (-1) exactly where the test expects → MR never labels a prime composite (no false negatives).

    - For **composite** (n):

      - Show the set of bases that behave “prime-like” (strong liars) forms a proper subgroup/coset structure inside ((\mathbb{Z}/n\mathbb{Z})^\times).
      - Prove that at most 1/4 of the bases are strong liars and at least 3/4 are strong witnesses.
      - Therefore, with random base choice, for composite (n):
        [
        \Pr[\text{one MR round says “probably prime”}] \le \tfrac14.
        ]
      - For (k) independent bases, probability composite (n) passes all rounds is at most ((1/4)^k).

  - **1.6 Deterministic MR for 64-bit integers** ([Codeforces][1])

    - There is a known finite set of bases (e.g. ({2,3,5,7,11,13,17,19,23,29,31,37})) such that if an odd (n<2^{64}) passes MR for all these bases, then (n) is provably prime. ([Codeforces][1])
    - We use MR with this fixed base set as a deterministic `is_prime_mr_det64(n)` oracle for all (n < 2^{64}).

  - **1.7 Fermat test vs Carmichael numbers (for comparison)** ([Stack Overflow][2])

    - Fermat test checks (a^{n-1} \equiv 1 \pmod{n}) for random bases (a).
    - Carmichael numbers are composite (n) such that (a^{n-1} \equiv 1 \pmod{n}) holds for **all** (a) coprime to (n); these defeat Fermat completely but are detected by MR for most bases.

---

- **2. Objectives and scope**

  - Algorithms to implement and benchmark:

    - A0: Trial Division (TD) up to (\lfloor\sqrt{n}\rfloor) – **only for small bit-lengths (16, 32)**, never used for 64-bit numbers.
    - A1: Fermat-k probabilistic test (Fermat).
    - A2: Miller–Rabin with k random bases (MR-k) – main focus.
    - A3: Deterministic MR-det-64 (fixed 12 bases) – ground-truth oracle and optional benchmark. ([Codeforces][1])

  - Goals:

    - Compare **runtime** of TD, Fermat-k, MR-k across bit-lengths and distributions.
    - Measure **empirical error rate** of each algorithm against MR-det-64 truth labels.
    - Build a small **RSA keygen + communication demo** that uses MR for prime generation and showcases a toy RSA protocol over sockets.
    - Connect implementation to real-world usage: RSA key generation in OpenSSL, Java `BigInteger.isProbablePrime`, and CAS / libraries that rely on MR-style tests. ([Wikipedia][3])

---

- **3. Number distributions and test sets**

  - **3.1 Bit-lengths and per-run sample size**

    - Bit-lengths (B \in {16, 32, 48, 64}).
    - Per configuration `(algo, dist, B, k)`, test `S` numbers (e.g. `S = 100000`).
    - Restriction for TD: run A0 **only** for (B \in {16, 32}); skip TD for 48 and 64 bits.

  - **3.2 Distributions**

    - `rand_odd_B` (ID 0):

      - Sample `S` integers uniformly in `[2^{B-1}, 2^B - 1]`, force oddness (`n |= 1`).
      - Mix of primes & composites with approximate natural density.

    - `carmichael_B` (ID 1):

      - Carmichael numbers loaded from precomputed file `carmichael_odd.txt` (generated by Python/SymPy or downloaded from known lists). ([Stack Overflow][2])
      - For given `B`, sample uniformly among Carmichael numbers with bit-length in `[B-1,B]` (or nearest available).
      - Truth label: always composite.

    - `comp_small_factor_B` (ID 2):

      - For each sample: choose a small prime `p` from precomputed list; generate odd `m` so that `n = p * m` falls in `[2^{B-1}, 2^B - 1]`.
      - Truth label: composite.

    - `primes_B` (ID 3):

      - Generate primes online: repeatedly sample random odd `n` in `[2^{B-1},2^B-1]`, test with `is_prime_mr_det64`, and keep only primes until `S` collected.
      - Truth label: prime.

---

- **4. Algorithms to implement and benchmark**

  - **4.1 Trial division (TD)**

    - Use for (B \in {16,32}) only.
    - Steps:

      - Handle (n < 2) and even `n` explicitly.
      - For odd `d = 3,5,7,...` while `d*d <= n`:

        - If `n % d == 0` → composite; if loop finishes → prime.

  - **4.2 Fermat-k**

    - For odd `n > 2`, parameter `k`:

      - Repeat `k` times: choose random `a` in `[2, n-2]`; compute `x = a^(n-1) mod n`.
      - If any `x != 1` → composite.
      - If all pass → “probably prime”.

  - **4.3 MR-k**

    - Implement as in Section 1; parameter `k` controls error bound (\le 4^{-k}). ([Codeforces][1])

  - **4.4 MR-det-64**

    - Fixed bases `{2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37}` for all `n < 2^64`. ([Codeforces][1])
    - If `n` passes MR for all bases → prime; else composite.
    - Used as **oracle** to label truth for all benchmark samples.

---

- **5. C++ benchmark program (`miller_rabin/src/main.cpp`)**

  - **5.1 Build**

    - Single file: `miller_rabin/src/main.cpp`.
    - Contains:

      - Implementations of TD, Fermat-k, MR-k, MR-det-64.
      - Modular arithmetic helpers (`mulmod`, `powmod` using `__uint128_t`).
      - Distribution sampling functions.
      - Timer class using `std::chrono::steady_clock`.
      - `main()` acting as benchmark driver.

    - Example build:

      - `g++ -O2 -std=c++17 src/main.cpp -o mr_bench`.

  - **5.2 Internal interfaces**

    - Primality functions:

      - `bool is_prime_td(uint64_t n);`
      - `bool is_probable_prime_fermat(uint64_t n, int k, std::mt19937_64& rng);`
      - `bool is_probable_prime_mr(uint64_t n, int k, std::mt19937_64& rng);`
      - `bool is_prime_mr_det64(uint64_t n);`

    - Distribution samplers:

      - `uint64_t sample_rand_odd(int bits, std::mt19937_64& rng);`
      - `uint64_t sample_comp_small_factor(int bits, std::mt19937_64& rng);`
      - `uint64_t sample_prime_via_mr_det(int bits, std::mt19937_64& rng);`
      - `uint64_t sample_carmichael(int bits, std::mt19937_64& rng);` (sampling from in-memory array loaded from file).

  - **5.3 CLI interface**

    - Binary: `./mr_bench`.
    - Arguments:

      - `<algo_id>`: `0=TD`, `1=Fermat`, `2=MR`.
      - `<dist_id>`: `0=rand_odd`, `1=carmichael`, `2=comp_small_factor`, `3=primes`.
      - `<bits>`: one of `16, 32, 48, 64`.
      - `<sample_count>`: `S`.
      - `<rounds>`: `k` for Fermat/MR, ignored for TD.
      - `<seed_base>`: 64-bit seed base.
      - `<reps>`: number of repetitions.

  - **5.4 Per-invocation behavior**

    - For `rep = 0..reps-1`:

      - Compute `seed = seed_base + rep`; construct `std::mt19937_64 rng(seed)`.
      - Set `error_count = 0`.
      - Start timer.
      - For `i = 1..sample_count`:

        - Sample `n` according to `(dist_id, bits)` with `rng`.
        - Compute `truth = is_prime_mr_det64(n);`
        - If `algo_id == 0` and `bits > 32`, **skip** TD: either do not schedule such runs from Python, or early-exit with `error` (specify convention in README; recommended: Python never calls TD with bits>32).
        - Compute `guess` = output of chosen algorithm (`true` = “prime/probably prime”, `false` = “composite”).
        - If `guess != truth`, increment `error_count`.

      - Stop timer; compute `time_ns_total`.
      - Print single line: `<time_ns_total> <error_count>`.

---

- **6. Python scripts and visualization**

  - **6.1 `run_benchmarks.py`**

    - Config:

      - `algos = ["TD","Fermat","MR"]` → `{0,1,2}`.
      - `dists = ["rand_odd","carmichael","comp_small_factor","primes"]` → `{0..3}`.
      - `bits_list_per_algo`:

        - For TD: `[16, 32]`.
        - For Fermat/MR: `[16, 32, 48, 64]`.

      - `sample_count = S`, `reps = R`.
      - `rounds_list` for Fermat/MR (e.g. `k ∈ {3,5,10}`).
      - `seed_base_global`.

    - For each valid `(algo, dist, bits, k)` combination:

      - Compute `seed_base` as deterministic function of `(algo, dist, bits, k)`.
      - Call `./mr_bench algo_id dist_id bits sample_count k seed_base reps`.
      - For each of the `reps` lines, parse `<time_ns_total> <error_count>` and append row to `miller_rabin/results/raw_results.csv` with columns:

        - `algo,dist,bits,rounds,sample_count,rep,time_ns_total,error_count`.

  - **6.2 `plot_results.py`**

    - Load `raw_results.csv`.
    - For each `(algo,dist,bits,rounds)` group compute:

      - `mean_time_per_test_ns = mean(time_ns_total / sample_count)`.
      - `std_time_per_test_ns`.
      - `mean_error_rate = mean(error_count / sample_count)`.

    - Plots (saved in `miller_rabin/results/plots/`):

      - For each `dist` and fixed `rounds`:

        - Time vs bits: line plot, x=`bits`, y=`mean_time_per_test_ns`, curves for TD/Fermat/MR.
        - Error vs bits: line plot, x=`bits`, y=`mean_error_rate`, curves for Fermat/MR (TD should have 0 error where defined).

      - Highlight plot: `dist=carmichael`, `bits=32` or `48`: bar chart of `mean_error_rate` per algo → Fermat ≈ 1, MR ≈ 0.

---

- **7. Real-world applications of Miller–Rabin (keep only the big ones)**

  - **RSA key generation (OpenSSL and friends)**

    - OpenSSL’s prime generation and `BN_is_prime_ex` functions perform multiple rounds of Miller–Rabin; documentation gives error bounds like ≤ (2^{-64}) or lower depending on key size. ([OpenSSL Documentation][4])
    - When generating RSA keys (`openssl genrsa`/`genpkey`), candidate primes that pass sieving are subjected to MR rounds; progress markers (`+`) in the CLI reflect MR tests. ([OpenSSL Documentation][5])

  - **General big-integer libraries**

    - GMP (`mpz_probab_prime_p`) runs Miller–Rabin (and in recent versions Baillie-PSW) to test primality of multi-precision integers. ([Reddit][6])
    - Java’s `BigInteger.isProbablePrime` uses MR (and, for ≥100-bit numbers, a Lucas test) as its core algorithm. ([Reddit][6])

  - **Computer algebra systems and number-theory software**

    - Maple `isprime`, Mathematica `PrimeQ`, PARI/GP, SageMath, FLINT, etc. use MR and related tests (often Baillie-PSW based on MR to base 2 plus a Lucas test). ([Reddit][6])

  - In the report, explicitly connect these uses to MR-k and MR-det-64 in your implementation and to observed error/runtime tradeoffs.

---

- **8. RSA demo (toy, C++, with sockets)**

  - **8.1 Scope**

    - Implement a **toy RSA demo** in C++, separate from benchmark runs, with two components:

      - Local CLI demo: generate small RSA keypair, let user enter message `m`, show every step of keygen, encryption and decryption.
      - Simple client–server demo over TCP sockets: send file data encrypted with toy RSA; MR is used to generate primes.

    - Security is **not** the goal; this is purely educational and will use small key sizes and textbook RSA (no padding).

  - **8.2 Key generation (shared logic)**

    - Key size: choose e.g. 32 or 40-bit primes so that `N = p*q` fits in `uint64_t` and computations are fast.
    - Algorithm:

      - Use `is_probable_prime_mr` or `is_prime_mr_det64` with small bit-length to generate random primes (p,q).
      - Compute (N = pq), (\phi = (p-1)(q-1)).
      - Fix public exponent (e = 65537) (or small odd `e` co-prime to (\phi)).
      - Compute private exponent `d` using extended Euclidean algorithm: `d ≡ e^{-1} (mod φ)`.

    - All intermediate values (`p,q,N,φ,e,d`) are logged to terminal with clear labels.

  - **8.3 Local CLI RSA demo (`rsa_demo_local` mode)**

    - Program flow:

      - Generate RSA keypair via above procedure, logging each step.
      - Prompt user for a small integer `m` with `0 < m < N`.
      - Compute ciphertext `c = m^e mod N`, log base, exponent, modulus, intermediate exponentiation steps (at a coarse granularity).
      - Compute decrypted message `m' = c^d mod N`, log steps.
      - Display `m'` and verify `m' == m`.

  - **8.4 Client–server RSA file transfer demo (`rsa_server` / `rsa_client`)**

    - Implementation assumption: POSIX sockets (`AF_INET`, `SOCK_STREAM`), blocking I/O.
    - Server (`rsa_server`):

      - On start, generate RSA keypair as above; logs `p,q,N,e,d`.
      - Bind to `localhost:PORT`, listen and accept a single client.
      - Send public key `(N,e)` to client (e.g., `N` and `e` as ASCII decimal separated by newline).
      - Receive encrypted chunks from client, decrypt each chunk with `m_chunk = c_chunk^d mod N`, and write to output file (e.g., `received.bin`).
      - Log each received ciphertext and corresponding plaintext chunk index.

    - Client (`rsa_client`):

      - Connect to server at `localhost:PORT`.
      - Receive public key `(N,e)` from server; log it.
      - Open input file; read in fixed-size chunks, interpret each chunk as an integer `< N` (e.g., treat up to 2 bytes or 3 bytes per chunk depending on `N`), or simple 1-byte chunks if you want to keep it trivial.
      - For each chunk value `m_chunk`, compute `c_chunk = m_chunk^e mod N` using same `powmod` routine.
      - Send `c_chunk` to server as ASCII decimal plus newline or as fixed-size binary, with clear, documented framing.
      - Log each plaintext chunk and ciphertext sent.

    - Both sides use the **same MR code** as in the benchmark module for prime generation (linked via common functions or copied with identical logic).

---

- **9. Directory structure (`miller_rabin/`)**

  - `miller_rabin/README.md`

    - Short description.
    - Build commands for `mr_bench`, `rsa_server`, `rsa_client`, `rsa_demo_local` (if separate binaries).
    - Example commands for running benchmarks and RSA demo.

  - `miller_rabin/src/`

    - `main.cpp` – benchmark driver + primality algorithms (TD, Fermat-k, MR-k, MR-det-64).
    - `rsa_demo.cpp` (or `rsa_server.cpp` + `rsa_client.cpp`, depending on how you split):

      - Shared RSA helper code (keygen, powmod).
      - CLI RSA demo and simple socket-based client/server.

  - `miller_rabin/data/`

    - `carmichael_odd.txt` – precomputed list of Carmichael numbers (one per line).

  - `miller_rabin/scripts/`

    - `prepare_carmichael.py` – generate/populate `carmichael_odd.txt`.
    - `run_benchmarks.py` – orchestrate benchmarking, writes `results/raw_results.csv`.
    - `plot_results.py` – generate plots under `results/plots/`.

  - `miller_rabin/results/`

    - `raw_results.csv` – collected benchmark data.
    - `plots/` – PNG/PDF plots for time vs bits and error vs bits.

---

# Karger Min Cut:

- **1. Algorithmic analysis and proof-flow (for report)**

  - **1.1 Problem definition**

    - Input: connected undirected unweighted graph (G = (V,E)).
    - Output: a global minimum cut ((S,\bar S)), i.e., a non-trivial partition of (V) minimizing the number of edges crossing between (S) and (\bar S).
    - Parameters:

      - (n = |V|), (m = |E|).
      - Contractions operate on a multigraph (parallel edges allowed, self-loops ignored).

  - **1.2 Basic Karger contraction algorithm (single run)**

    - While current graph has more than 2 supernodes:

      - Pick an edge uniformly at random from the current edge set.
      - Contract that edge: merge its endpoints into a single supernode; remove self-loops; keep parallel edges.

    - When only 2 supernodes remain, the number of edges between them is the cut value returned by this run.

  - **1.3 Success probability (single run)**

    - Let (\lambda) be the global min-cut value.
    - At any stage, minimum degree (\delta(G) \ge \lambda), hence (m \ge n\lambda/2).
    - A min-cut consists of exactly (\lambda) edges.
    - On an (i)-vertex graph, probability that Karger contracts an edge from a fixed min-cut in that step is at most
      [
      \frac{\lambda}{m_i} \le \frac{\lambda}{i\lambda/2} = \frac{2}{i}.
      ]
    - Therefore probability that first contraction avoids that min-cut is at least (1 - 2/n); conditioning on survival, repeat on an ((n-1))-vertex graph, etc.
    - Probability that **no contraction ever hits that min-cut** over the entire run is at least
      [
      \prod_{i= n}^{3} \left(1 - \frac{2}{i}\right) = \frac{2}{n(n-1)} = \binom{n}{2}^{-1}.
      ]
    - So in worst case, a _single_ Karger run finds a particular min-cut with probability at least (p\_{\min}(n) = 2/(n(n-1))).

  - **1.4 Repetition and target error rate**

    - We fix a **global target failure probability** per graph:

      - `delta_target = 0.1` (i.e., want success probability ≥ 0.9 for every graph).

    - For each graph with (n) vertices, we run independent Karger trials `R(n)` times and take the minimum cut value observed.
    - With per-run success probability (p \ge p\_{\min}(n) = 2/(n(n-1))), probability all (R) runs fail is
      [
      \Pr[\text{all fail}] \le (1-p)^R \le e^{-pR}.
      ]
    - To ensure (e^{-pR} \le \delta*{\text{target}}), it suffices that
      [
      R \ge \frac{\ln(1/\delta*{\text{target}})}{p*{\min}(n)} = \frac{n(n-1)}{2}\ln\frac{1}{\delta*{\text{target}}}.
      ]
    - We therefore define in code:

      - `R(n) = max(1, ceil(0.5 * n * (n - 1) * log(1.0 / delta_target)))`.
      - This is a **theoretical bound**; in practice we will choose small `n` (e.g., up to ~100) so that `R(n)` remains implementable.

  - **1.5 Deterministic Stoer–Wagner min-cut algorithm**

    - Stoer–Wagner solves global min-cut in undirected weighted graphs in (O(nm + n^2 \log n)) time.
    - It repeatedly runs “min-cut phases” (maximum adjacency search) and contracts the last two vertices of each phase, tracking the lightest cut encountered.
    - For unweighted graphs we treat all edges as weight 1 and run Stoer–Wagner as is.
    - Output is exact global min-cut; we use this as deterministic baseline and ground-truth oracle.

---

- **2. Objectives and scope (MinCut module)**

  - Implement and benchmark only:

    - A0: **Repeated basic Karger** (with `R(n)` derived from `delta_target = 0.1`).
    - A1: **Stoer–Wagner** deterministic min-cut.

  - Graph classes to test (only three):

    - `"two_cluster_gnp"` – planted bisection community graph.
    - `"gnp_sparse"` – sparse Erdős–Rényi (G(n,p)).
    - `"gnp_dense"` – dense Erdős–Rényi (G(n,p)).

  - Keep implementation and benchmarks **minimal** and **uniform**:

    - Same `delta_target` for all graphs.
    - Same `n` values and number of graph samples per distribution.
    - Record only runtime and error metrics that are easy to compute:

      - Mean time per graph.
      - Mean squared error (MSE) of cut value.

---

- **3. Graph distributions (final simplified set)**

  - **3.1 `two_cluster_gnp`** (ID 0, planted community)

    - Parameters:

      - `n` vertices; partition fixed as `V1 = {0..n/2-1}`, `V2 = {n/2..n-1}` (assume even `n`).
      - Inside-block edge probability `p_in` (e.g., `p_in = 0.3`).
      - Cross-block edge probability `p_out` (e.g., `p_out = 0.05`).

    - Generation:

      - For each unordered pair in `V1` and `V2`, add edge with probability `p_in`.
      - For each pair across `V1,V2`, add edge with probability `p_out`.
      - Regenerate if graph is disconnected (to keep Stoer–Wagner happy).

  - **3.2 `gnp_sparse`** (ID 1)

    - Parameters:

      - Edge probability `p_sparse = c/n` with small constant `c` (e.g., `c = 4` or `6`).

    - Generation:

      - Erdős–Rényi (G(n,p\_{\text{sparse}})) with independent edges; regenerate until connected.

  - **3.3 `gnp_dense`** (ID 2)

    - Parameters:

      - Edge probability `p_dense` fixed (e.g., `0.2` or `0.5`).

    - Generation:

      - Erdős–Rényi (G(n,p\_{\text{dense}})); regenerate until connected.

  - **3.4 Size and sample choices (keep small and fixed)**

    - Example (to be frozen in doc):

      - `n_list = [30, 60, 90]`.
      - `graphs_per_rep = 20` or `50` (depending on time budget).

    - We do **not** vary any other parameters; everything else is fixed in code.

---

- **4. Algorithms and C++ implementation plan**

  - **4.1 Shared graph representation**

    - Vertices: integers `0..n-1`.
    - Edges: `std::vector<std::pair<int,int>> edges` (for Karger).
    - Stoer–Wagner can either:

      - Work from an adjacency matrix `std::vector<std::vector<int>> w` (weights = 1); or
      - Build its own adjacency data from `edges`.

    - When generating graphs, we build both `edges` and `adj` once per graph; both algorithms read from these structures.

  - **4.2 Repeated Karger (A0)**

    - Helper: DSU (Union–Find) for contractions.
    - Single run on a fixed edge list `edges` and `n`:

      - Initialize DSU with `n` sets, component count `comp = n`.
      - While `comp > 2`:

        - Sample uniform random edge `(u,v)` from `edges`.
        - Let `fu = find(u), fv = find(v)`; if `fu == fv`, continue; else union and decrement `comp`.

      - After loop, scan `edges` and count edges whose endpoints are in different DSU components → `cut_value_run`.

    - Repeated runs:

      - Compute `R(n)` from `n` and `delta_target` as in Section 1.4.
      - Run the above single-run `R(n)` times on the same `edges`, each with a “fresh” DSU.
      - Track `best_cut = min(best_cut, cut_value_run)`.
      - Return `best_cut`.

  - **4.3 Stoer–Wagner (A1)**

    - Implement standard Stoer–Wagner algorithm for undirected weighted graphs:

      - Maintain adjacency matrix or weight structure.
      - Run `n-1` phases, each using “maximum adjacency search” to find an s–t min-cut and merging s,t.
      - Track the best cut weight across phases and return it as global min-cut.

    - For unweighted graphs, every edge weight = 1.
    - This output is taken as the **exact min-cut** for benchmarking.

---

- **5. Benchmark driver and metrics**

  - **5.1 C++ benchmark driver**

    - Single file: `mincut/src/main.cpp` → binary `mincut_bench`.
    - Responsibilities:

      - Generate graphs for a given `(dist_id, n)` combination.
      - For each graph, compute true min-cut value `lambda_truth` with Stoer–Wagner (A1).
      - Run either Karger (A0) or Stoer–Wagner (A1) inside a timed region.
      - Compare estimated cut `lambda_hat` with `lambda_truth`.

    - CLI arguments:

      - `<algo_id>`: `0 = Karger`, `1 = StoerWagner`.
      - `<dist_id>`: `0 = two_cluster_gnp`, `1 = gnp_sparse`, `2 = gnp_dense`.
      - `<n>`: graph size from predefined `n_list`.
      - `<graphs_per_rep>`: number of graphs `G` per repetition.
      - `<seed_base>`: base random seed (64-bit).
      - `<reps>`: number of repetitions.

  - **5.2 Per-invocation behavior**

    - For `rep = 0..reps-1`:

      - Initialize RNG with `seed = seed_base + rep`.
      - Set `total_time_ns = 0`.
      - Set accumulators for errors: `sum_sq_error = 0`.
      - For `g = 1..graphs_per_rep`:

        - Generate graph `G` according to `(dist_id, n, rng)`.
        - Compute `lambda_truth` using Stoer–Wagner (not timed).
        - Start timer.
        - If `algo_id == 0` → run repeated Karger and get `lambda_hat`.
        - If `algo_id == 1` → run Stoer–Wagner and get `lambda_hat`.
        - Stop timer, add to `total_time_ns`.
        - Let `err = lambda_hat - lambda_truth`; accumulate `sum_sq_error += err * err`.

      - After all graphs:

        - Print one line: `<total_time_ns> <sum_sq_error>`.

---

- **6. Python scripts and visualization**

  - **6.1 `run_benchmarks.py`**

    - Fixed configuration at top:

      - `algos = ["Karger","StoerWagner"]` → `{0,1}`.
      - `dists = ["two_cluster_gnp","gnp_sparse","gnp_dense"]` → `{0,1,2}`.
      - `n_list = [30, 60, 90]`.
      - `graphs_per_rep = 20` (or `50` if time permits).
      - `reps = 5` (e.g.).
      - `seed_base_global` fixed.

    - For each `(algo, dist, n)` combination:

      - Compute `seed_base` as deterministic function of `(algo, dist, n)`.
      - Call `./mincut_bench algo_id dist_id n graphs_per_rep seed_base reps`.
      - Capture `reps` lines; for each line, parse `total_time_ns`, `sum_sq_error`.
      - Append row to `mincut/results/raw_results.csv` with columns:

        - `algo,dist,n,graphs_per_rep,rep,total_time_ns,sum_sq_error`.

  - **6.2 `plot_results.py`**

    - Load `raw_results.csv`.
    - For each group `(algo, dist, n)` compute:

      - `mean_time_per_graph_ns = mean(total_time_ns / graphs_per_rep)`.
      - `std_time_per_graph_ns`.
      - `mean_sq_error = mean(sum_sq_error / graphs_per_rep)`.

    - Required plots:

      - **Time vs n** (per distribution):

        - x-axis: `n`.
        - y-axis: `mean_time_per_graph_ns`.
        - Lines for Karger and Stoer–Wagner.

      - **MSE vs n** (per distribution):

        - x-axis: `n`.
        - y-axis: `mean_sq_error`.
        - Lines for Karger and Stoer–Wagner (Stoer–Wagner should be identically zero).

---

- **7. Real-world applications of minimum cut (for discussion; no implementation)**

  - **Network reliability and connectivity**

    - Edge connectivity of a network equals its global min-cut: minimum number of links that must fail to disconnect the network.
    - Used to design robust telecommunication and transportation networks by ensuring min-cut exceeds expected failure levels.

  - **VLSI partitioning and placement**

    - Hypergraph/graph partitioning for chip design often uses cut minimization: divide the circuit into blocks with few inter-block connections (low cut) to improve routability and timing.

  - **Image segmentation / vision**

    - s–t min-cut and global cuts are fundamental in graph-based image segmentation; pixels/superpixels as nodes, edges capture similarity, and cut gives object/background separation.

  - **Graph clustering / communities**

    - Small cuts separating weakly connected portions of a graph are used as building blocks for clustering and community detection; global min-cut is a basic structural descriptor.

---

- **8. Directory structure (`mincut/`)**

  - `mincut/README.md`

    - Summary of module and its scope.
    - Exact compile command for `mincut_bench`.
    - Example commands for `run_benchmarks.py` and `plot_results.py`.

  - `mincut/src/`

    - `main.cpp`

      - Graph generators for `two_cluster_gnp`, `gnp_sparse`, `gnp_dense`.
      - Implementations of repeated Karger and Stoer–Wagner.
      - Timer utility.
      - `main()` that parses CLI, runs benchmarks, prints `<total_time_ns> <sum_sq_error>`.

  - `mincut/scripts/`

    - `run_benchmarks.py` – orchestrates all runs, writes `results/raw_results.csv`.
    - `plot_results.py` – loads CSV, generates time vs `n` and MSE vs `n` plots in `results/plots/`.

  - `mincut/results/`

    - `raw_results.csv` – aggregated benchmark data.
    - `plots/` – generated figures for the report.

---

# Randomized cycle finding with color coding

- **1. Algorithmic analysis and proof-flow (for report)**

  - **1.1 Problem definition**

    - Input:

      - Undirected, simple graph (G=(V,E)) with (n = |V|), (m = |E|).
      - Integer (k \ge 3) (target cycle length).

    - Output:

      - Either: a simple cycle of length exactly (k) (k distinct vertices, k edges, first = last), or report that no such cycle exists.

    - Goal:

      - Design a randomized FPT algorithm (color-coding + DP over color subsets) that runs in time (f(k)\cdot \text{poly}(n+m)), and compare it against a naive DFS/backtracking baseline.

  - **1.2 Color-coding framework**

    - Randomly assign each vertex (v) a color `color(v) ∈ {0,…,k−1}` independently and uniformly.
    - A k-cycle (C = (v*0,v_1,…,v*{k-1},v*0)) is **colorful** if all (v_0,…,v*{k-1}) receive pairwise distinct colors.
    - Any simple k-cycle in the graph becomes colorful under some colorings; color-coding restricts the search to colorful cycles, which can be detected via DP over color-subsets.

  - **1.3 Success probability of a single random coloring**

    - Fix a particular simple k-cycle with distinct vertices (v*0,…,v*{k-1}).
    - Probability that its vertices receive pairwise distinct colors under a random coloring:

      - Number of injective colorings: (k! ) (all permutations of colors on these vertices).
      - Number of all possible colorings: (k^k).
      - So
        [
        \Pr[\text{cycle is colorful}] = \frac{k!}{k^k} \approx e^{-k}.
        ]

    - If the graph contains at least one k-cycle, a single random coloring makes **some** k-cycle colorful with probability (\Omega(k!/k^k)).

  - **1.4 DP over color subsets for colorful k-cycles**

    - Representation:

      - After coloring, define `mask` as a k-bit subset of colors; `mask` indicates which colors are used.
      - DP state: `DP[mask][v] = true` iff there exists a simple path that:

        - Uses exactly the colors in `mask` (each at most once), and
        - Ends at vertex `v`.

    - Initialization:

      - For each vertex `v`, let `c = color(v)`; set `DP[1<<c][v] = true`.

    - Transitions:

      - For each `mask` and vertex `v` with `DP[mask][v] = true`:

        - For each neighbor `u` of `v`:

          - Let `c = color(u)`; if `(mask & (1<<c)) == 0` then set `DP[mask | (1<<c)][u] = true`.

      - Process masks in increasing order of popcount.

    - Detecting a colorful k-cycle:

      - For each vertex `r` treated as an anchor:

        - Require that `r`’s color is included in `mask`.
        - Look for states where `popcount(mask) = k` and `DP[mask][v] = true` and edge `(v,r) ∈ E`.
        - Then the path represented by `DP[mask][v]` plus edge `(v,r)` forms a simple colorful k-cycle.

    - Complexity per coloring (for one anchor set):

      - Number of DP states: (O(2^k \cdot n)).
      - Each state relaxes over incident edges: (O(2^k \cdot m)) operations in worst case.
      - For small k (≤ ~15–18) and moderate n this is practical.

  - **1.5 Repetition for high success probability**

    - Let (p(k) = k!/k^k). For a graph with at least one k-cycle, each coloring has success probability ≥ (p(k)) of making some k-cycle colorful and detectable by DP.
    - With (R) independent colorings, failure probability ≤ ((1 - p(k))^R \le e^{-p(k) R}).
    - For a target failure probability `delta_target` (same for all experiments), choose
      [
      R(k) = \left\lceil \frac{\ln(1/\text{delta_target})}{p(k)} \right\rceil
      \approx \left\lceil e^k \ln\frac{1}{\text{delta_target}} \right\rceil.
      ]
    - In code:

      - Precompute or hardcode `R(k)` for each k in the experimental range.
      - For each graph instance, run color-coding DP for exactly `R(k)` colorings and return the best cycle found (or “none” if all fail).

  - **1.6 Baseline DFS/backtracking algorithm for k-cycles**

    - Deterministic, exact algorithm used for comparison and for small sanity tests.
    - Logic:

      - For each start vertex `s`:

        - Run DFS with depth limit `k`, maintaining visited set.
        - At depth `d` at current vertex `v`:

          - If `d == k` and `v == s` and all vertices in the path are distinct (except the repeated `s`), report a k-cycle.
          - Else expand neighbors `u` not yet visited, except that at depth `k` we only allow `u == s`.

    - Complexity:

      - Roughly (O(n \cdot d^{k-1})) where `d` is average branching factor; exponential in k but fine for small `n,k`.

---

- **2. Objectives and scope (Color-coding for k-cycle)**

  - Algorithms to implement and benchmark:

    - A0: `cc_k_cycle` – randomized color-coding + subset DP for simple k-cycles, with `R(k)` repetitions chosen for fixed error target.
    - A1: `dfs_k_cycle` – naive DFS/backtracking detection of simple k-cycles.

  - Graph model:

    - Undirected, simple, unweighted graphs.
    - Input always generated by a **single synthetic generator** with a planted k-cycle and random noise edges (see Section 3).

  - Evaluation:

    - Existence ground truth: generator guarantees at least one k-cycle in every graph instance.
    - We only care about **detection** (“did we find some simple k-cycle?”), not about identifying the planted one specifically.
    - Metrics per algorithm:

      - Runtime per graph (mean, std).
      - Error rate = fraction of graphs where algorithm fails to output any valid k-cycle (false negative).
      - MSE = average of `(truth − found)^2` per graph; since `truth=1`, this equals the error rate.

---

- **3. Graph instance generation (single distribution)**

  - **3.1 Distribution: `planted_cycle_noise` (only dataset used)**

    - Parameters:

      - `n`: total number of vertices.
      - `k`: length of planted simple cycle (3 ≤ k ≤ n).
      - `p_noise`: probability of adding extra “noise” edges between non-cycle vertex pairs.

    - Generation procedure:

      - Step 1: create vertex set `V = {0,1,…,n-1}`.
      - Step 2: pick a random subset `C` of size k:

        - Sample `k` distinct vertices uniformly from `V`; store as `c[0..k-1]` in random order.

      - Step 3: add the planted k-cycle:

        - For `i = 0..k-1`, add edge `(c[i], c[(i+1) mod k])`.

      - Step 4: add noise edges:

        - For each unordered pair `{u,v}` with `u < v`:

          - If `(u,v)` is **not** in the planted cycle edge set, then with probability `p_noise` add edge `(u,v)`.

      - Step 5 (optional connectivity step):

        - Optionally check connectivity; if disconnected, add a few random edges between components (documented behavior in code).

    - Guarantees:

      - Every graph has at least one simple k-cycle (the planted one).
      - For `p_noise > 0`, graphs may have multiple k-cycles; algorithms are considered successful if they find **any** simple k-cycle of length k.

  - **3.2 Parameter ranges (to be fixed in config)**

    - Typical experimental ranges (to be decided when scoping down):

      - `n ∈ {30, 60, 90, 120}`.
      - `k ∈ {5, 6, 7, 8}`.
      - `p_noise ∈ {0.05, 0.1, 0.2}`.

    - All graphs in the project will be generated solely from this distribution with different `(n,k,p_noise)` combinations.

---

- **4. Algorithms and C++ implementation details**

  - **4.1 Graph representation**

    - Vertices: `0..n-1`.
    - Adjacency list: `std::vector<std::vector<int>> adj`.
    - No generic graph class; use simple structs or global helpers for speed.

  - **4.2 Color-coding k-cycle (`cc_k_cycle`)**

    - Signature (internal):

      - `bool cc_k_cycle(const Graph& G, int k, int R, std::vector<int>& cycle_out);`

    - High-level steps for each of the `R` colorings:

      - Color vertices:

        - For each `v`, assign `color[v]` uniformly in `[0, k-1]` using `std::mt19937_64`.

      - DP allocation:

        - Assume `k ≤ 20` so `1 << k` fits in 32-bit.
        - Use `std::vector<std::vector<char>> DP(1<<k, std::vector<char>(n, 0));` or bitsets to save memory.

      - Initialization:

        - For each `v`, set `DP[1 << color[v]][v] = 1`.

      - DP transitions:

        - Iterate masks in increasing popcount order.
        - For each `mask` and `v` with `DP[mask][v] = 1`:

          - For each neighbor `u` of `v`:

            - Let `c = color[u]`.
            - If `(mask & (1<<c)) == 0`, set `DP[mask | (1<<c)][u] = 1`.

      - Cycle detection:

        - For each vertex `r` (anchor):

          - Let `cr = color[r]`.
          - For each `mask` with `popcount(mask) = k` and `(mask & (1<<cr)) != 0`:

            - For each neighbor `v` of `r`:

              - If `DP[mask][v] = 1` and both `r` and `v` are in the vertex set encoded by `mask` (handled by reconstruction), then `(path represented by DP, plus edge (v,r))` is a simple colorful k-cycle.

        - On first success, reconstruct the cycle using parent pointers and return `true`.

      - If no cycle found after all `R` colorings, return `false`.

    - For this project, correctness is checked by verifying that `cycle_out` is:

      - k distinct vertices;
      - consecutive vertices adjacent;
      - last vertex adjacent to first;
      - all vertices distinct (except closure).

  - **4.3 DFS-based k-cycle (`dfs_k_cycle`)**

    - Signature:

      - `bool dfs_k_cycle(const Graph& G, int k, std::vector<int>& cycle_out);`

    - Implementation:

      - For each start vertex `s` from `0..n-1`:

        - Run recursive DFS with parameters `(current_vertex, depth, start_vertex, visited[], path[])`.
        - Base cases:

          - If `depth == k` and `current_vertex == start_vertex` and `path` contains k distinct vertices: record `cycle_out` and return `true`.
          - If `depth == k` and `current_vertex != start_vertex`: backtrack.

        - Recurrence:

          - For each neighbor `u` of `current_vertex`:

            - If `depth+1 < k` and `u` not visited: mark visited, extend path, recurse.
            - If `depth+1 == k` and `u == start_vertex`: record cycle.

      - Return `false` if no k-cycle found.

    - This algorithm is exponential in k; used as a baseline and for smaller `(n,k)` combinations.

  - **4.4 Randomization and repetition policy for cc_k_cycle**

    - Fix a global `delta_target` (e.g., 0.1) and allowed k-range in code.
    - For each k in range:

      - Precompute `p(k) = k! / k^k` (in double) and `R(k) = ceil(log(1/delta_target) / p(k))`.
      - Optionally cap with `R_max` to keep runs practical (e.g., `R_max = 1000`).

    - For each graph instance with length parameter k:

      - Call `cc_k_cycle(G, k, R(k), cycle_out)`.

---

- **5. Benchmark driver and metrics**

  - **5.1 C++ benchmark driver (`color_coding/src/main.cpp`)**

    - Binary: `kcycle_bench`.
    - CLI arguments (in order):

      - `<algo_id>`: `0 = cc_k_cycle`, `1 = dfs_k_cycle`.
      - `<n>`: number of vertices.
      - `<k>`: cycle length.
      - `<p_noise>`: noise probability as floating-point (e.g., `0.05`).
      - `<graphs_per_rep>`: number of graph instances per repetition (G).
      - `<seed_base>`: 64-bit seed base.
      - `<reps>`: number of repetitions.

    - Per-invocation behavior:

      - For `rep = 0..reps−1`:

        - Initialize `std::mt19937_64 rng(seed_base + rep)`.
        - Set `total_time_ns = 0`, `error_count = 0`.
        - For `g = 1..graphs_per_rep`:

          - Generate graph using `planted_cycle_noise(n, k, p_noise, rng)`.
          - Start timer.
          - Run algorithm chosen by `algo_id` to obtain `found` ∈ {true,false}.
          - Stop timer; add elapsed time to `total_time_ns`.
          - Validate output (if `found`): check returned cycle is a simple k-cycle; if invalid, treat as `found = false`.
          - If `found == false`, increment `error_count`.

        - Print one line to stdout:

          - `<total_time_ns> <error_count>` (space-separated).

      - No other output.

  - **5.2 Python scripts**

    - `run_benchmarks.py`:

      - Configure lists:

        - `algos = ["cc_k_cycle","dfs_k_cycle"]` → `{0,1}`.
        - `n_list`, `k_list`, `p_noise_list`.
        - `graphs_per_rep`, `reps`, `seed_base_global`.

      - Loop over `(algo, n, k, p_noise)` combinations:

        - Compute `seed_base` deterministically from these parameters.
        - Call `./kcycle_bench algo_id n k p_noise graphs_per_rep seed_base reps`.
        - For each of the `reps` lines, parse `total_time_ns`, `error_count`.
        - Append rows to `color_coding/results/raw_results.csv` with columns:

          - `algo,n,k,p_noise,graphs_per_rep,rep,total_time_ns,error_count`.

    - `plot_results.py`:

      - Load `raw_results.csv`.
      - For each group `(algo,n,k,p_noise)`:

        - `mean_time_per_graph_ns = mean(total_time_ns / graphs_per_rep)`.
        - `std_time_per_graph_ns`.
        - `mean_error_rate = mean(error_count / graphs_per_rep)`.

      - Produce plots (PNG/PDF in `color_coding/results/plots/`):

        - Time vs `n` for fixed `(k,p_noise)`, curves for both algorithms.
        - Time vs `k` for fixed `(n,p_noise)`.
        - Error rate vs `k` for `cc_k_cycle` (dfs_k_cycle should be ~0 error in the feasible regime).

---

- **6. Directory structure (`color_coding/`)**

  - `color_coding/README.md`

    - Short description of module and goals.
    - Example compile command for `kcycle_bench`.
    - Example `run_benchmarks.py` invocation and notes on interpreting plots.

  - `color_coding/src/`

    - `main.cpp`

      - Graph generator for `planted_cycle_noise`.
      - Implementation of `cc_k_cycle` and `dfs_k_cycle`.
      - Timer utility.
      - CLI parsing and benchmark loop printing `<total_time_ns> <error_count>`.

  - `color_coding/scripts/`

    - `run_benchmarks.py` – orchestrates experiments, writes `results/raw_results.csv`.
    - `plot_results.py` – produces time/error plots into `results/plots/`.

  - `color_coding/results/`

    - `raw_results.csv` – aggregated benchmark data.
    - `plots/` – generated figures (time vs n, time vs k, error vs k).

---
