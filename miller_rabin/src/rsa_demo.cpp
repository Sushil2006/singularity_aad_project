#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>

#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

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
        if (e & 1ULL) result = mulmod(result, p, mod);
        p = mulmod(p, p, mod);
        e >>= 1;
    }
    return result;
}

/**
 * Send a line (appends '\n') over a socket.
 */
void send_line(int fd, const std::string& line) {
    std::string msg = line + "\n";
    send(fd, msg.c_str(), msg.size(), 0);
}

/**
 * Receive a single line terminated by '\n'. Returns false on EOF.
 */
bool recv_line(int fd, std::string& out) {
    out.clear();
    char ch;
    while (true) {
        ssize_t r = recv(fd, &ch, 1, 0);
        if (r <= 0) {
            return !out.empty();
        }
        if (ch == '\n') {
            return true;
        }
        out.push_back(ch);
    }
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
 * @return bool true if round passes, false if composite detected.
 */
bool miller_rabin_round(uint64_t n, uint64_t a, uint64_t s, uint64_t d) {
    uint64_t x = powmod(a, d, n);
    if (x == 1ULL || x == n - 1) return true;
    for (uint64_t r = 1; r < s; ++r) {
        x = mulmod(x, x, n);
        if (x == n - 1) return true;
        if (x == 1ULL) return false;
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
        if (!miller_rabin_round(n, a, s, d)) return false;
    }
    return true;
}

/**
 * Compute gcd via Euclid.
 *
 * @param a First integer.
 * @param b Second integer.
 * @return uint64_t gcd(a, b).
 */
uint64_t gcd(uint64_t a, uint64_t b) {
    while (b != 0) {
        uint64_t t = a % b;
        a = b;
        b = t;
    }
    return a;
}

/**
 * Extended gcd for modular inverse.
 *
 * @param a First integer.
 * @param b Second integer.
 * @return std::tuple<int64_t, int64_t, int64_t> (g, x, y) with g=gcd(a,b), ax+by=g.
 */
std::tuple<int64_t, int64_t, int64_t> extended_gcd(int64_t a, int64_t b) {
    if (b == 0) return {a, 1, 0};
    auto [g, x1, y1] = extended_gcd(b, a % b);
    int64_t x = y1;
    int64_t y = x1 - (a / b) * y1;
    return {g, x, y};
}

/**
 * Compute modular inverse of a modulo m.
 *
 * @param a Value to invert.
 * @param m Modulus (must be > 0).
 * @return uint64_t Inverse such that (a * inv) % m == 1.
 */
uint64_t modinv(uint64_t a, uint64_t m) {
    auto [g, x, _] = extended_gcd(static_cast<int64_t>(a), static_cast<int64_t>(m));
    if (g != 1) {
        throw std::runtime_error("modinv: values not coprime");
    }
    int64_t res = x % static_cast<int64_t>(m);
    if (res < 0) res += static_cast<int64_t>(m);
    return static_cast<uint64_t>(res);
}

/**
 * Generate a random prime of given bit length using deterministic MR.
 *
 * @param bits Bit-length target.
 * @param rng Random generator.
 * @return uint64_t Prime.
 */
uint64_t generate_prime(int bits, std::mt19937_64& rng) {
    if (bits < 4 || bits > 62) {
        throw std::invalid_argument("bits out of supported range");
    }
    uint64_t low = 1ULL << (bits - 1);
    uint64_t high = (1ULL << bits) - 1ULL;
    std::uniform_int_distribution<uint64_t> dist(low, high);
    while (true) {
        uint64_t candidate = dist(rng) | 1ULL;
        if (is_prime_mr_det64(candidate)) {
            return candidate;
        }
    }
}

/**
 * RSA keypair container.
 */
struct RSAKeypair {
    uint64_t n;
    uint64_t e;
    uint64_t d;
    uint64_t p;
    uint64_t q;
};

/**
 * Generate a toy RSA keypair with primes of the requested bit size.
 *
 * @param bits Bit-length per prime.
 * @param rng Random generator.
 * @return RSAKeypair Generated keypair.
 */
RSAKeypair generate_keypair(int bits, std::mt19937_64& rng) {
    uint64_t p = generate_prime(bits, rng);
    uint64_t q = generate_prime(bits, rng);
    while (q == p) {
        q = generate_prime(bits, rng);
    }
    uint64_t n = p * q;
    uint64_t phi = (p - 1) * (q - 1);
    uint64_t e = 65537;
    if (gcd(e, phi) != 1) {
        e = 3;
    }
    uint64_t d = modinv(e, phi);
    return {n, e, d, p, q};
}

/**
 * Print key details to stdout.
 *
 * @param key RSA keypair.
 */
void log_key(const RSAKeypair& key) {
    std::cout << "p=" << key.p << "\nq=" << key.q << "\nn=" << key.n << "\nphi=" << (key.p - 1) * (key.q - 1)
              << "\ne=" << key.e << "\nd=" << key.d << "\n";
}

/**
 * Interactive local RSA demo.
 *
 * @param bits Bit-length per prime.
 */
void demo_local(int bits) {
    std::mt19937_64 rng(std::random_device{}());
    RSAKeypair key = generate_keypair(bits, rng);
    std::cout << "Generated keypair:\n";
    log_key(key);

    std::cout << "Enter message m (0 < m < n): ";
    uint64_t m;
    std::cin >> m;
    if (m == 0 || m >= key.n) {
        std::cerr << "Message out of range\n";
        return;
    }
    uint64_t c = powmod(m, key.e, key.n);
    uint64_t m_dec = powmod(c, key.d, key.n);

    std::cout << "Ciphertext c = " << c << "\n";
    std::cout << "Decrypted m' = " << m_dec << "\n";
    if (m_dec == m) {
        std::cout << "Success: m' == m\n";
    } else {
        std::cout << "Mismatch detected\n";
    }
}

/**
 * Start a simple RSA server that accepts one client and writes decrypted bytes to a file.
 *
 * @param port TCP port.
 * @param bits Bit-length per prime.
 */
void run_server(int port, int bits) {
    std::mt19937_64 rng(std::random_device{}());
    RSAKeypair key = generate_keypair(bits, rng);
    std::cout << "Server key:\n";
    log_key(key);

    int server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd < 0) {
        throw std::runtime_error("Failed to create socket");
    }
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = INADDR_ANY;
    addr.sin_port = htons(static_cast<uint16_t>(port));

    int opt = 1;
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
    if (bind(server_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        close(server_fd);
        throw std::runtime_error("Bind failed");
    }
    if (listen(server_fd, 1) < 0) {
        close(server_fd);
        throw std::runtime_error("Listen failed");
    }
    std::cout << "Listening on port " << port << "...\n";
    int client_fd = accept(server_fd, nullptr, nullptr);
    if (client_fd < 0) {
        close(server_fd);
        throw std::runtime_error("Accept failed");
    }
    std::string pub = std::to_string(key.n) + " " + std::to_string(key.e) + "\n";
    send(client_fd, pub.c_str(), pub.size(), 0);

    std::cout << "Receiving commands (-m/-f)...\n";
    std::string line;
    while (recv_line(client_fd, line)) {
        if (line == "MSG") {
            std::cout << "----- Incoming message -----\n";
            std::string msg_bytes;
            while (recv_line(client_fd, line) && line != "END") {
                uint64_t c = std::stoull(line);
                uint64_t m = powmod(c, key.d, key.n);
                unsigned char byte = static_cast<unsigned char>(m & 0xFF);
                std::cout << "c=" << c << " -> m=" << static_cast<int>(byte) << " ('"
                          << static_cast<char>(byte) << "')\n";
                msg_bytes.push_back(static_cast<char>(m & 0xFF));
            }
            std::cout << "[msg] " << msg_bytes << "\n";
            std::cout << "----- End message -----\n";
        } else if (line.rfind("FILE ", 0) == 0) {
            std::string filename = line.substr(5);
            std::ofstream out(filename, std::ios::binary);
            if (!out.is_open()) {
                std::cerr << "Cannot open file for writing: " << filename << "\n";
                continue;
            }
            std::cout << "----- Incoming file: " << filename << " -----\n";
            while (recv_line(client_fd, line) && line != "END") {
                uint64_t c = std::stoull(line);
                uint64_t m = powmod(c, key.d, key.n);
                char byte = static_cast<char>(m & 0xFF);
                out.write(&byte, 1);
                std::cout << "c=" << c << " -> m=" << static_cast<int>(static_cast<unsigned char>(byte)) << " ('"
                          << byte << "')\n";
            }
            std::cout << "[file] wrote to " << filename << "\n";
            std::cout << "----- End file -----\n";
        }
    }

    close(client_fd);
    close(server_fd);
    std::cout << "Server done; connection closed\n";
}

/**
 * Interactive client loop that sends messages or files without closing the connection.
 */
void interactive_loop(int sock, uint64_t n, uint64_t e) {
    while (true) {
        std::cout << "> ";
        std::string line;
        if (!std::getline(std::cin, line)) {
            break;
        }
        if (line.rfind("-m ", 0) == 0) {
            std::string message = line.substr(3);
            send_line(sock, "MSG");
            for (unsigned char byte : message) {
                uint64_t c = powmod(static_cast<uint64_t>(byte), e, n);
                std::cout << "client: m=" << static_cast<int>(byte) << " ('" << static_cast<char>(byte)
                          << "') -> c=" << c << "\n";
                send_line(sock, std::to_string(c));
            }
            send_line(sock, "END");
        } else if (line.rfind("-f ", 0) == 0) {
            std::string filename = line.substr(3);
            std::ifstream in(filename, std::ios::binary);
            if (!in.is_open()) {
                std::cerr << "Failed to open file: " << filename << "\n";
                continue;
            }
            send_line(sock, "FILE " + filename);
            char byte;
            while (in.get(byte)) {
                uint64_t c = powmod(static_cast<uint64_t>(static_cast<unsigned char>(byte)), e, n);
                std::cout << "client: m=" << static_cast<int>(static_cast<unsigned char>(byte)) << " ('" << byte
                          << "') -> c=" << c << "\n";
                send_line(sock, std::to_string(c));
            }
            send_line(sock, "END");
            std::cout << "Sent file: " << filename << "\n";
        } else {
            std::cout << "Commands: -m <message>, -f <filename>\n";
        }
    }
}

/**
 * Run a simple RSA client sending plaintext bytes.
 *
 * @param host Target host (IPv4).
 * @param port TCP port.
 * @param message Message to send.
 */
void run_client(const std::string& host, int port) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        throw std::runtime_error("Failed to create socket");
    }
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(static_cast<uint16_t>(port));
    if (inet_pton(AF_INET, host.c_str(), &addr.sin_addr) <= 0) {
        close(sock);
        throw std::runtime_error("Invalid host");
    }
    if (connect(sock, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) < 0) {
        close(sock);
        throw std::runtime_error("Connect failed");
    }

    std::string line;
    if (!recv_line(sock, line)) {
        close(sock);
        throw std::runtime_error("Failed to read public key");
    }
    size_t space_pos = line.find(' ');
    if (space_pos == std::string::npos) {
        close(sock);
        throw std::runtime_error("Failed to parse public key");
    }
    uint64_t n = std::stoull(line.substr(0, space_pos));
    uint64_t e = std::stoull(line.substr(space_pos + 1));
    std::cout << "Received public key n=" << n << " e=" << e << "\n";

    interactive_loop(sock, n, e);
    close(sock);
}

/**
 * Entry point for RSA demo modes.
 *
 * @param argc Argument count.
 * @param argv Argument array.
 * @return int Exit code.
 */
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage:\n"
                  << "  ./rsa_demo local [prime_bits]\n"
                  << "  ./rsa_demo server <port> [prime_bits]\n"
                  << "  ./rsa_demo client <host> <port>\n";
        return 1;
    }

    std::string mode = argv[1];
    if (mode == "local") {
        int bits = (argc >= 3) ? std::stoi(argv[2]) : 16;
        demo_local(bits);
    } else if (mode == "server") {
        if (argc < 3) {
            std::cerr << "server mode requires <port>\n";
            return 1;
        }
        int port = std::stoi(argv[2]);
        int bits = (argc >= 4) ? std::stoi(argv[3]) : 16;
        run_server(port, bits);
    } else if (mode == "client") {
        if (argc < 4) {
            std::cerr << "client mode requires <host> <port>\n";
            return 1;
        }
        std::string host = argv[2];
        int port = std::stoi(argv[3]);
        run_client(host, port);
    } else {
        std::cerr << "Unknown mode\n";
        return 1;
    }

    return 0;
}
