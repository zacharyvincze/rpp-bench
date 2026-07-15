/**
 * @file bench_registry.hpp
 * @brief The OpAdapter interface and the global op registry.
 *
 * Every rppt_* op has a distinct C signature, so each op needs a small compiled
 * adapter that (a) declares what it supports, (b) builds its op-specific param
 * tensors in setup(), and (c) issues the timed call in run(). Adapters live in
 * src/ops/*.cpp and self-register via REGISTER_RPP_BENCH.
 */
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

// Adapter base. One instance is created per registered benchmark, so adapters
// may hold their per-run param tensors as members.
class OpAdapter {
public:
    virtual ~OpAdapter() = default;

    /**
     * @brief Restrict the dtype sweep to what the op actually supports.
     * @return Supported dtypes; defaults to all (U8/F32/F16/I8).
     */
    virtual std::vector<RpptDataType> supportedDtypes() const {
        return {RpptDataType::U8, RpptDataType::F32, RpptDataType::F16, RpptDataType::I8};
    }
    /**
     * @brief Restrict the layout sweep to what the op actually supports.
     * @return Supported layouts; defaults to all (PKD3/PLN3/PLN1).
     */
    virtual std::vector<Layout> supportedLayouts() const {
        return {Layout::PKD3, Layout::PLN3, Layout::PLN1};
    }
    /**
     * @brief Restrict the sweep to backends the op actually implements.
     *
     * Defaults to both; ops with a GPU-only kernel (e.g. erode/dilate) narrow
     * this to HIP.
     * @return Supported backends.
     */
    virtual std::vector<RppBackend> supportedBackends() const {
        return {RppBackend::RPP_HOST_BACKEND, RppBackend::RPP_HIP_BACKEND};
    }

    /**
     * @brief Source-buffer leading halo padding, in bytes, the op requires.
     *
     * HIP filter kernels need a leading offset of 12*(kernelSize/2) bytes plus
     * kernelSize/2 halo columns (see srcAdditionalStride()). Defaults to none;
     * the runner allocates the source accordingly.
     * @return Leading offset in bytes; 0 when no halo is needed.
     */
    virtual int srcOffsetBytes(const BenchContext &) const { return 0; }
    /**
     * @brief Extra halo columns to add to the source width.
     * @return Additional stride columns; 0 when no halo is needed.
     */
    virtual int srcAdditionalStride(const BenchContext &) const { return 0; }

    /**
     * @brief Allocate op-specific param tensors. Called once, outside the timing loop.
     * @param ctx The fully-resolved sweep point.
     * @param src Ready-to-use source buffer.
     * @param dst Ready-to-use destination buffer.
     */
    virtual void setup(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst) = 0;

    /**
     * @brief The hot path - issues the rppt_* call. Timed.
     *
     * For HIP the runner synchronizes the stream after each call, so run() need
     * only enqueue.
     * @param ctx The fully-resolved sweep point.
     * @param src Source buffer.
     * @param dst Destination buffer.
     * @param handle The RPP handle to issue the call against.
     * @return The rppt_* call's status.
     */
    virtual RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                          rppHandle_t handle) = 0;

    /**
     * @brief Free op-specific param tensors. Called once, after the timing loop.
     */
    virtual void teardown() {}
};

using AdapterFactory = std::function<std::unique_ptr<OpAdapter>()>;

// Global registry (singleton). Adapters register at static-init time.
class OpRegistry {
public:
    static OpRegistry &instance();
    void add(const std::string &name, AdapterFactory factory);
    /**
     * @brief Whether an adapter is registered under @p name.
     * @param name Config/registry key to look up.
     * @return True if present; use before find(), which throws on a miss.
     */
    bool has(const std::string &name) const;
    /**
     * @brief Look up an adapter factory by name.
     * @param name Config/registry key to look up.
     * @return The registered factory.
     * @throws std::runtime_error if no adapter is registered under @p name.
     */
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

/**
 * @brief Register an adapter class under a config name.
 *
 * Place at file scope in a src/ops adapter source file.
 * Example: REGISTER_RPP_BENCH("brightness", BrightnessAdapter)
 */
#define REGISTER_RPP_BENCH(cfg_name, AdapterClass)                                                 \
    namespace {                                                                                    \
    const ::rppbench::AdapterRegistrar _rpp_bench_reg_##AdapterClass(cfg_name, [] {                \
        return std::make_unique<AdapterClass>();                                                   \
    });                                                                                            \
    }

} // namespace rppbench

#endif // RPP_BENCH_REGISTRY_HPP
