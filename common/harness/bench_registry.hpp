// ============================================================================
// bench_registry.hpp - the OpAdapter interface and the global op registry.
//
// Every rppt_* op has a distinct C signature, so each op needs a small compiled
// adapter that (a) declares what it supports, (b) builds its op-specific param
// tensors in setup(), and (c) issues the timed call in run(). Adapters live in
// src/ops/*.cpp and self-register via REGISTER_RPP_BENCH.
// ============================================================================
#ifndef RPP_BENCH_REGISTRY_HPP
#define RPP_BENCH_REGISTRY_HPP

#include <rpp/rpp.h>
#include <nlohmann/json.hpp>
#include "config/bench_enums.hpp"
#include "harness/bench_tensor.hpp"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace rppbench {

// One fully-resolved point of the sweep, handed to an adapter.
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

    // Convenience: read a param with a default when absent/null.
    template <typename T>
    T param(const char *key, T fallback) const {
        if (params && params->contains(key) && !params->at(key).is_null())
            return params->at(key).get<T>();
        return fallback;
    }
};

// Adapter base. One instance is created per registered benchmark, so adapters
// may hold their per-run param tensors as members.
class OpAdapter {
public:
    virtual ~OpAdapter() = default;

    // Restrict the sweep to what the op actually supports. Defaults to "all".
    virtual std::vector<RpptDataType> supportedDtypes() const {
        return {RpptDataType::U8, RpptDataType::F32, RpptDataType::F16, RpptDataType::I8};
    }
    virtual std::vector<Layout> supportedLayouts() const {
        return {Layout::PKD3, Layout::PLN3, Layout::PLN1};
    }

    // Source-buffer halo padding the op requires (HIP filter kernels need a
    // leading offset of 12*(kernelSize/2) bytes plus kernelSize/2 halo columns).
    // Defaults to none; the runner allocates the source accordingly.
    virtual int srcOffsetBytes(const BenchContext &) const { return 0; }
    virtual int srcAdditionalStride(const BenchContext &) const { return 0; }

    // Allocate op-specific param tensors. Called once, outside the timing loop.
    virtual void setup(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst) = 0;

    // The hot path - issues the rppt_* call. Timed. For HIP the runner
    // synchronizes the stream after each call, so run() need only enqueue.
    virtual RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                          rppHandle_t handle) = 0;

    // Free op-specific param tensors. Called once, after the timing loop.
    virtual void teardown() {}
};

using AdapterFactory = std::function<std::unique_ptr<OpAdapter>()>;

// Global registry (singleton). Adapters register at static-init time.
class OpRegistry {
public:
    static OpRegistry &instance();
    void add(const std::string &name, AdapterFactory factory);
    // Returns nullptr factory-wrapper miss via has(); find throws if absent.
    bool has(const std::string &name) const;
    AdapterFactory find(const std::string &name) const;
    std::vector<std::string> names() const;

private:
    std::vector<std::pair<std::string, AdapterFactory>> entries_;
};

// Helper used by the registration macro.
struct AdapterRegistrar {
    AdapterRegistrar(const std::string &name, AdapterFactory f) {
        OpRegistry::instance().add(name, std::move(f));
    }
};

// Register an adapter class under a config name. Place at file scope in an
// ops/*.cpp. Example: REGISTER_RPP_BENCH("brightness", BrightnessAdapter)
#define REGISTER_RPP_BENCH(cfg_name, AdapterClass)                                                 \
    namespace {                                                                                    \
    const ::rppbench::AdapterRegistrar _rpp_bench_reg_##AdapterClass(cfg_name, [] {                \
        return std::make_unique<AdapterClass>();                                                   \
    });                                                                                            \
    }

} // namespace rppbench

#endif // RPP_BENCH_REGISTRY_HPP
