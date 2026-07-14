// ============================================================================
// bench_config.hpp - the runtime sweep description, parsed from a JSON file.
//
// A config has "defaults" (a base matrix) plus a list of "ops". Each op inherits
// the defaults and may override any matrix axis, add per-op params, and (for
// resize-like ops) list destination sizes. Changing the sweep needs no rebuild.
// ============================================================================
#ifndef RPP_BENCH_CONFIG_HPP
#define RPP_BENCH_CONFIG_HPP

#include <rpp/rppdefs.h>
#include <nlohmann/json.hpp>
#include "config/bench_enums.hpp"

#include <string>
#include <utility>
#include <vector>

namespace rppbench {

// One matrix = the cartesian axes to sweep. Empty vectors mean "unset" so an op
// can inherit the value from defaults.
struct Matrix {
    std::vector<std::string> backends; // "HOST" / "HIP"
    std::vector<RpptDataType> dtypes;
    std::vector<Layout> layouts;
    std::vector<int> batchSizes;
    std::vector<std::pair<int, int>> imageSizes; // (width, height)
};

struct OpSpec {
    std::string name;                          // registry key, e.g. "brightness"
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
    double minTimeSec = 0.0;   // 0 => use Google Benchmark default
    int iterations = 0;        // 0 => unset; fixed iteration count (excludes minTimeSec)
    int repetitions = 0;       // 0 => use Google Benchmark default
    int warmupIterations = 0;  // 0 => none; untimed iterations run once per case first
};

// Parse a config file. Throws std::runtime_error with a helpful message on any
// malformed field.
BenchConfig load_config(const std::string &path);

} // namespace rppbench

#endif // RPP_BENCH_CONFIG_HPP
