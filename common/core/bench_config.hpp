/**
 * @file bench_config.hpp
 * @brief The runtime sweep description, parsed from a JSON file.
 *
 * A config has "defaults" (a base matrix + default library) plus a list of
 * "ops". Each op inherits the defaults and may override its library, any matrix
 * axis, add per-op params, and (for resize-like ops) list destination sizes.
 * Changing the sweep needs no rebuild. This is library-neutral: it speaks the
 * vocabulary enums in core/bench_types.hpp, never a library's own types.
 */
#ifndef RPP_BENCH_CORE_CONFIG_HPP
#define RPP_BENCH_CORE_CONFIG_HPP

#include <nlohmann/json.hpp>
#include "core/bench_types.hpp"

#include <string>
#include <utility>
#include <vector>

namespace rppbench {

// One matrix = the cartesian axes to sweep. Empty vectors mean "unset" so an op
// can inherit the value from defaults.
struct Matrix {
    std::vector<std::string> backends; // "HOST" / "HIP"
    std::vector<DataType> dtypes;
    std::vector<Layout> layouts;
    std::vector<int> batchSizes;
    std::vector<std::pair<int, int>> imageSizes; // (width, height)
};

struct OpSpec {
    std::string name;                          // op registry key, e.g. "brightness"
    std::string library;                       // which library, e.g. "rpp"
    Matrix matrix;                             // fully-resolved (defaults merged)
    std::vector<std::pair<int, int>> dstSizes; // optional; for resize/crop ops
    // One or more op-specific param sets, each swept as its own axis. Populated
    // from "param_sets" (a list) or "params" (a single set); always >= 1 entry
    // (an empty object when neither is given).
    std::vector<nlohmann::json> paramSets;
};

struct BenchConfig {
    std::vector<OpSpec> ops;
    // Optional global knobs applied to every registered benchmark.
    double minTimeSec = 0.0;  // 0 => use Google Benchmark default
    int iterations = 0;       // 0 => unset; fixed iteration count (excludes minTimeSec)
    int repetitions = 0;      // 0 => use Google Benchmark default
    int warmupIterations = 0; // 0 => none; untimed iterations run once per case first
};

/**
 * @brief Parse a config file.
 * @param path Path to the JSON config file.
 * @return The parsed sweep description.
 * @throws std::runtime_error with a helpful message on any malformed field.
 */
BenchConfig load_config(const std::string &path);

} // namespace rppbench

#endif // RPP_BENCH_CORE_CONFIG_HPP
