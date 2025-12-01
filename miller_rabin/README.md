# Miller–Rabin Benchmarks and RSA Demo

This module benchmarks trial division, Fermat tests, and Miller–Rabin variants on 64-bit integers and includes a toy RSA demonstration.

## Preparing data

Generate Carmichael list (already committed, re-run if needed):
```bash
python3 scripts/prepare_carmichael.py
```

## Build

```bash
g++ -O2 -std=c++17 src/main.cpp -o mr_bench
g++ -O2 -std=c++17 src/rsa_demo.cpp -o rsa_demo
```

## Benchmark CLI

```
./mr_bench <algo_id> <dist_id> <bits> <sample_count> <rounds> <seed_base> <reps>
```

- `algo_id`: `0=TD`, `1=Fermat`, `2=MR`
- `dist_id`: `0=rand_odd`, `1=carmichael`, `2=comp_small_factor`, `3=primes`
- `bits`: `16/32` for TD; `16/32/48/64` for Fermat/MR
- Output per repetition: `<time_ns_total> <error_count>`

## Running all benchmarks (15s budget)

```bash
python3 scripts/run_benchmarks.py
```

Defaults:
- `sample_count = 1`, `reps = 100,000` (TD at 48 bits uses 100 reps)
- `rounds = 5` for Fermat/MR (TD uses a single pass)
- No timeout enforced; runs can be long with these settings.

Results are written to `results/raw_results.csv`.

## Plotting

```bash
python3 scripts/plot_results.py
```

Outputs time/error plots per distribution and a Carmichael highlight bar chart to `results/plots/`.

## Toy RSA demo

```bash
./rsa_demo local [prime_bits]              # Local encrypt/decrypt demo (default 16-bit primes)
./rsa_demo server <port> [prime_bits]      # Start server, stays running
./rsa_demo client <host> <port>            # Start client, interactive
```

Notes:
- Educational only; uses small primes and textbook RSA (no padding).
- Socket code targets POSIX systems (AF_INET, blocking I/O).

### Suggested folder layout and usage

Run the binaries from separate `server/` and `client/` folders so file transfers stay organized:
```
mkdir -p server client
cp rsa_demo server/ && cp rsa_demo client/
cd server && ./rsa_demo server 9000        # in one terminal
# in another terminal
cd client && ./rsa_demo client 127.0.0.1 9000
```

- Put any files you want to send inside the `client/` directory. From the client prompt, use:
  - `-m <message>` to send a text message.
  - `-f <filename>` to send a file from the `client/` directory; it will be written with the same name in the `server/` directory.
- The connection stays open; you can issue multiple `-m` or `-f` commands until you exit the client.
