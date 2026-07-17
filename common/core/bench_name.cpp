/**
 * @file bench_name.cpp
 * @brief The benchmark-name encoding (see bench_name.hpp).
 */
#include "core/bench_name.hpp"

#include "core/bench_types.hpp"

#include <algorithm>
#include <string>
#include <vector>

namespace rppbench {

namespace {

/**
 * @brief Render a JSON scalar as a bare string ("BILINEAR", "5", "1.75").
 *
 * No quotes for strings, compact repr for numbers/bools.
 * @param v The JSON scalar.
 * @return The bare-string rendering.
 */
std::string json_scalar(const nlohmann::json &v) {
    return v.is_string() ? v.get<std::string>() : v.dump();
}

/**
 * @brief Compact, deterministic "k1=v1,k2=v2" encoding of a param set.
 * @param p The param set object.
 * @return The encoded params, or empty if none.
 */
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

std::string encode_name(const BenchPoint &p) {
    std::string s = "op:" + p.opName + "/library:" + p.library +
                    "/backend:" + backend_name(p.backend) + "/dtype:" + dtype_name(p.dtype) +
                    "/layout:" + layout_name(p.layout) + "/batch:" + std::to_string(p.batch) +
                    "/size:" + std::to_string(p.width) + "x" + std::to_string(p.height);
    if (p.dstWidth != p.width || p.dstHeight != p.height)
        s += "/dst:" + std::to_string(p.dstWidth) + "x" + std::to_string(p.dstHeight);
    // Encode params so swept sets get unique names and appear in the output.
    if (p.params) {
        std::string ps = encode_params(*p.params);
        if (!ps.empty())
            s += "/params:" + ps;
    }
    return s;
}

} // namespace rppbench
