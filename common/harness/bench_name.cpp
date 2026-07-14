// ============================================================================
// bench_name.cpp - the benchmark-name encoding (see bench_name.hpp).
// ============================================================================
#include "harness/bench_name.hpp"

#include "config/bench_enums.hpp"

#include <algorithm>
#include <string>
#include <vector>

namespace rppbench {

namespace {

// A JSON scalar as a bare string ("BILINEAR", "5", "1.75") - no quotes for
// strings, compact repr for numbers/bools.
std::string json_scalar(const nlohmann::json &v) {
    return v.is_string() ? v.get<std::string>() : v.dump();
}

// Compact, deterministic "k1=v1,k2=v2" encoding of a param set (empty if none).
std::string encode_params(const nlohmann::json &p) {
    if (!p.is_object() || p.empty())
        return "";
    std::vector<std::string> kv;
    for (auto it = p.begin(); it != p.end(); ++it)
        kv.push_back(it.key() + "=" + json_scalar(it.value()));
    std::sort(kv.begin(), kv.end());
    std::string s;
    for (size_t i = 0; i < kv.size(); ++i) {
        if (i)
            s += ",";
        s += kv[i];
    }
    return s;
}

} // namespace

std::string encode_name(const BenchContext &c) {
    std::string s = "op:" + c.opName + "/backend:" + c.backendName +
                    "/dtype:" + dtype_name(c.dtype) + "/layout:" + layout_name(c.layout) +
                    "/batch:" + std::to_string(c.batch) + "/size:" + std::to_string(c.width) + "x" +
                    std::to_string(c.height);
    if (c.dstWidth != c.width || c.dstHeight != c.height)
        s += "/dst:" + std::to_string(c.dstWidth) + "x" + std::to_string(c.dstHeight);
    // Encode params so swept sets get unique names and appear in the output.
    if (c.params) {
        std::string ps = encode_params(*c.params);
        if (!ps.empty())
            s += "/params:" + ps;
    }
    return s;
}

} // namespace rppbench
