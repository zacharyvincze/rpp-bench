/**
 * @file bench_context.hpp
 * @brief The RPP-typed, fully-resolved sweep point handed to an adapter.
 *
 * The typed bridge between the neutral core and the RPP adapters: the core
 * resolves a BenchPoint (neutral enums), and each RPP case converts it - once -
 * into a BenchContext whose fields are RPP's own types (ctx.backend is an
 * RppBackend, ctx.dtype an RpptDataType), so the ~70 adapters keep using the
 * exact context surface they always had. Adapters never see a BenchPoint.
 */
#ifndef RPP_BENCH_RPP_CONTEXT_HPP
#define RPP_BENCH_RPP_CONTEXT_HPP

#include <nlohmann/json.hpp>
#include "core/bench_point.hpp"
#include "rpp/rpp_enums.hpp"

#include <string>

namespace rppbench {

// One fully-resolved point of the sweep, in RPP's own vocabulary.
struct BenchContext {
    std::string opName;
    std::string backendName; // "HOST" / "HIP"
    RppBackend backend = RppBackend::RPP_HOST_BACKEND;
    RpptDataType dtype = RpptDataType::U8;
    Layout layout = Layout::PKD3;
    int batch = 0;
    int width = 0, height = 0;       // source dimensions
    int dstWidth = 0, dstHeight = 0; // destination dimensions (== src unless resized)
    bool isHip = false;
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

    /**
     * @brief Build the RPP-typed context from a neutral sweep point.
     * @param p The resolved, library-neutral sweep point.
     * @return The equivalent RPP-typed context.
     */
    static BenchContext from(const BenchPoint &p) {
        BenchContext c;
        c.opName = p.opName;
        c.backendName = backend_name(p.backend);
        c.backend = to_rpp_backend(p.backend);
        c.dtype = to_rpp_dtype(p.dtype);
        c.layout = p.layout;
        c.batch = p.batch;
        c.width = p.width;
        c.height = p.height;
        c.dstWidth = p.dstWidth;
        c.dstHeight = p.dstHeight;
        c.isHip = (p.backend == Backend::Hip);
        c.params = p.params;
        return c;
    }
};

} // namespace rppbench

#endif // RPP_BENCH_RPP_CONTEXT_HPP
