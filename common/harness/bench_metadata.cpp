/**
 * @file bench_metadata.cpp
 * @brief Collects the run metadata embedded in every result file's context.
 *
 * Build-time facts (rpp-bench commit, ROCm version) come from the generated
 * bench_build_info.hpp; runtime facts (loaded librpp, GPU, driver, clock
 * policy) are probed here. The context keys deliberately avoid Google
 * Benchmark's own keys (host_name, num_cpus, date, ...), which it records
 * itself.
 */
#include "harness/bench_metadata.hpp"

#include <benchmark/benchmark.h>
#include <nlohmann/json.hpp>
#include <rpp/rpp.h>
#include <rpp/rpp_version.h>

#include "bench_build_info.hpp"

#include <dlfcn.h>
#include <sys/utsname.h>

#include <cctype>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>

#if RPP_BENCH_HIP
#include <hip/hip_runtime.h>
#endif

namespace rppbench {

namespace {

constexpr const char *kUnknown = "unknown";

/**
 * @brief 64-bit FNV-1a hash, as fixed-width hex.
 *
 * Used instead of std::hash, whose value is implementation-defined and so
 * can't be compared across machines or toolchains.
 * @param s Bytes to hash.
 * @return 16 lowercase hex digits.
 */
std::string fnv1a64_hex(const std::string &s) {
    std::uint64_t h = 0xcbf29ce484222325ULL;
    for (unsigned char c : s) {
        h ^= c;
        h *= 0x100000001b3ULL;
    }
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(h));
    return buf;
}

/**
 * @brief First line of a text file with surrounding whitespace trimmed.
 * @param path File to read (typically a sysfs/procfs node).
 * @return The line, or empty if the file can't be read.
 */
std::string read_first_line(const std::string &path) {
    std::ifstream f(path);
    std::string line;
    if (!f || !std::getline(f, line))
        return {};
    const auto b = line.find_first_not_of(" \t\r\n");
    const auto e = line.find_last_not_of(" \t\r\n");
    return b == std::string::npos ? std::string{} : line.substr(b, e - b + 1);
}

/**
 * @brief The CPU model string from /proc/cpuinfo (Google Benchmark doesn't record it).
 * @return The model name, or "unknown".
 */
std::string cpu_model() {
    std::ifstream f("/proc/cpuinfo");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("model name", 0) == 0) {
            const auto colon = line.find(':');
            if (colon != std::string::npos && colon + 2 <= line.size())
                return line.substr(colon + 2);
        }
    }
    return kUnknown;
}

/**
 * @brief Kernel name and release, e.g. "Linux 6.8.0-117-generic".
 * @return The OS string, or "unknown".
 */
std::string os_kernel() {
    utsname u{};
    if (uname(&u) != 0)
        return kUnknown;
    return std::string(u.sysname) + " " + u.release;
}

/**
 * @brief Resolved path of the librpp actually loaded into this process.
 *
 * Several RPP versions can be installed side by side, and the dynamic loader
 * (LD_LIBRARY_PATH, rpath) decides which one runs - not the headers the binary
 * was compiled against. Asking the loader which file defines rppCreate settles it.
 * @return The canonical library path, or "unknown".
 */
std::string loaded_rpp_library() {
    Dl_info info{};
    if (dladdr(reinterpret_cast<void *>(&rppCreate), &info) == 0 || info.dli_fname == nullptr)
        return kUnknown;
    char resolved[PATH_MAX];
    if (realpath(info.dli_fname, resolved) != nullptr)
        return resolved;
    return info.dli_fname;
}

/**
 * @brief Format a HIP version integer (major*10000000 + minor*100000 + patch).
 * @param v The raw version from hipRuntimeGetVersion/hipDriverGetVersion.
 * @return "major.minor.patch (raw)".
 */
[[maybe_unused]] std::string hip_version_string(int v) {
    return std::to_string(v / 10000000) + "." + std::to_string((v / 100000) % 100) + "." +
           std::to_string(v % 100000) + " (" + std::to_string(v) + ")";
}

/**
 * @brief Parse a JSON file without throwing.
 * @param path File to parse.
 * @return The document, or a discarded value if unreadable/invalid.
 */
nlohmann::json parse_json_file(const std::string &path) {
    std::ifstream f(path);
    return nlohmann::json::parse(f, nullptr, /*allow_exceptions=*/false);
}

/**
 * @brief Record the sweep config's path and a hash of its content.
 *
 * The hash is taken over the compact form, so whitespace-only edits don't
 * change it; equal hashes mean the same sweep was requested. The content itself
 * is embedded after the run by embed_config_in_results() - as context it would
 * be one multi-KB line in Google Benchmark's console header.
 * @param configPath Path of the loaded config.
 */
void add_config_context(const std::string &configPath) {
    benchmark::AddCustomContext("config_path", configPath);
    const nlohmann::json j = parse_json_file(configPath);
    benchmark::AddCustomContext("config_hash", j.is_discarded() ? kUnknown : fnv1a64_hex(j.dump()));
}

/**
 * @brief Record the environment variables that change what or where the run measures.
 *
 * Unset variables are skipped, so their absence in the context means "unset".
 */
void add_env_context() {
    for (const char *name :
         {"OMP_NUM_THREADS", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "LD_LIBRARY_PATH"}) {
        if (const char *v = std::getenv(name))
            benchmark::AddCustomContext(std::string("env_") + name, v);
    }
}

#if RPP_BENCH_HIP
/**
 * @brief Record the HIP runtime/driver versions and the active GPU's identity.
 *
 * Covers the device the benchmarks run on (the current device - the harness
 * never calls hipSetDevice, so device 0 after any *_VISIBLE_DEVICES masking).
 * Also records the amdgpu clock policy from sysfs: "auto" means clocks float
 * with load, so GPU timings are noisier than under a fixed/profiling level.
 */
void add_gpu_context() {
    int v = 0;
    benchmark::AddCustomContext("hip_runtime_version", hipRuntimeGetVersion(&v) == hipSuccess
                                                           ? hip_version_string(v)
                                                           : kUnknown);
    benchmark::AddCustomContext("hip_driver_version", hipDriverGetVersion(&v) == hipSuccess
                                                          ? hip_version_string(v)
                                                          : kUnknown);

    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess)
        count = 0;
    benchmark::AddCustomContext("gpu_count", std::to_string(count));
    if (count == 0)
        return;

    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess)
        dev = 0;
    hipDeviceProp_t p{};
    if (hipGetDeviceProperties(&p, dev) != hipSuccess)
        return;

    benchmark::AddCustomContext("gpu_index", std::to_string(dev));
    benchmark::AddCustomContext("gpu_name", p.name);
    benchmark::AddCustomContext("gpu_arch", p.gcnArchName);
    benchmark::AddCustomContext("gpu_compute_units", std::to_string(p.multiProcessorCount));
    benchmark::AddCustomContext("gpu_max_sclk_mhz", std::to_string(p.clockRate / 1000));
    benchmark::AddCustomContext("gpu_max_mclk_mhz", std::to_string(p.memoryClockRate / 1000));
    benchmark::AddCustomContext("gpu_vram_mib", std::to_string(p.totalGlobalMem >> 20));

    char bus[64] = {};
    if (hipDeviceGetPCIBusId(bus, sizeof(bus), dev) == hipSuccess) {
        std::string busId(bus);
        for (char &c : busId)
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        benchmark::AddCustomContext("gpu_pci_bus_id", busId);
        const std::string level =
            read_first_line("/sys/bus/pci/devices/" + busId + "/power_dpm_force_performance_level");
        benchmark::AddCustomContext("gpu_perf_level", level.empty() ? kUnknown : level);
    }
}
#endif

/**
 * @brief Join argv into one shell-like string (arguments with spaces are quoted).
 * @param args The original arguments.
 * @return The joined command line.
 */
std::string join_command_line(const std::vector<std::string> &args) {
    std::ostringstream os;
    for (size_t i = 0; i < args.size(); ++i) {
        if (i)
            os << ' ';
        if (args[i].find_first_of(" \t'\"") != std::string::npos)
            os << '\'' << args[i] << '\'';
        else
            os << args[i];
    }
    return os.str();
}

} // namespace

void add_run_metadata(const std::string &configPath, const std::vector<std::string> &commandLine) {
    benchmark::AddCustomContext("rpp_bench_schema", kResultSchemaVersion);
    benchmark::AddCustomContext("rpp_bench_commit", RPP_BENCH_GIT_SHA);
    benchmark::AddCustomContext("command_line", join_command_line(commandLine));

    // RPP: the headers compiled against vs the library actually loaded.
    benchmark::AddCustomContext("rpp_version", std::to_string(RPP_VERSION_MAJOR) + "." +
                                                   std::to_string(RPP_VERSION_MINOR) + "." +
                                                   std::to_string(RPP_VERSION_PATCH));
    benchmark::AddCustomContext("rpp_backend", RPP_BENCH_RPP_BACKEND);
    benchmark::AddCustomContext("rpp_library", loaded_rpp_library());
    benchmark::AddCustomContext("rocm_path", RPP_BENCH_ROCM_PATH);
    benchmark::AddCustomContext("rocm_version", RPP_BENCH_ROCM_VERSION);

    benchmark::AddCustomContext("cpu_model", cpu_model());
    benchmark::AddCustomContext("os_kernel", os_kernel());
#if RPP_BENCH_HIP
    add_gpu_context();
#endif
    add_config_context(configPath);
    add_env_context();
}

bool embed_config_in_results(const std::string &resultsPath, const std::string &configPath) {
    nlohmann::json results = parse_json_file(resultsPath);
    const nlohmann::json config = parse_json_file(configPath);
    if (!results.is_object() || config.is_discarded())
        return false;
    results["rpp_bench_config"] = config;
    // Write a sibling file and rename over the original, so an interrupted
    // write can't destroy the results.
    const std::string tmp = resultsPath + ".tmp";
    {
        std::ofstream out(tmp);
        out << results.dump(2) << '\n';
        if (!out)
            return false;
    }
    return std::rename(tmp.c_str(), resultsPath.c_str()) == 0;
}

} // namespace rppbench
