/**
 * @file bench_point.hpp
 * @brief One fully-resolved, library-neutral point of the sweep.
 *
 * The core builds a BenchPoint for every surviving matrix combo and hands it to
 * a Library to construct a runnable case (and to encode_name() to render). It is
 * the neutral counterpart of a library's own typed context (e.g. RPP's
 * BenchContext), carrying only vocabulary types from bench_types.hpp plus the
 * opaque JSON param blob. No library or device headers.
 */
#ifndef RPP_BENCH_CORE_POINT_HPP
#define RPP_BENCH_CORE_POINT_HPP

#include <nlohmann/json.hpp>
#include "core/bench_types.hpp"

#include <string>

namespace rppbench {

// One fully-resolved point of the sweep, neutral over the imaging library.
struct BenchPoint {
    std::string opName;
    std::string library;             // which implementation, e.g. "rpp"
    Backend backend = Backend::Host; // where it runs (HOST/HIP)
    DataType dtype = DataType::U8;
    Layout layout = Layout::PKD3;
    int batch = 0;
    int width = 0, height = 0;              // source dimensions
    int dstWidth = 0, dstHeight = 0;        // destination dimensions (== src unless resized)
    const nlohmann::json *params = nullptr; // op-specific knobs from config

    /**
     * @brief Read a param with a default when absent/null.
     * @param key JSON key to look up in the op's param set.
     * @param fallback Value returned when @p key is absent or null.
     * @return The parsed param value, or @p fallback.
     */
    template <typename T>
    T param(const char *key, T fallback) const {
        if (params && params->contains(key) && !params->at(key).is_null())
            return params->at(key).get<T>();
        return fallback;
    }
};

} // namespace rppbench

#endif // RPP_BENCH_CORE_POINT_HPP
