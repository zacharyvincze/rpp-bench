/**
 * @file bench_registry.hpp
 * @brief The RPP OpAdapter interface and the RPP library's op registry.
 *
 * Every rppt_* op has a distinct C signature, so each op needs a small compiled
 * adapter that (a) declares what it supports, (b) builds its op-specific param
 * tensors in setup(), and (c) issues the timed call in run(). Adapters live in
 * libraries/rpp/ops/*.cpp and self-register via REGISTER_RPP_BENCH.
 *
 * This interface is RPP-typed on purpose - it is internal to the RPP module. The
 * neutral core never sees it: RppLibrary reads an adapter's capabilities and
 * RppCase drives its setup()/run(), translating to/from the core's neutral
 * BenchPoint at the module boundary (see rpp_library.* / rpp_case.*).
 */
#ifndef RPP_BENCH_REGISTRY_HPP
#define RPP_BENCH_REGISTRY_HPP

#include <rpp/rpp.h>
#include "rpp/bench_context.hpp"
#include "rpp/bench_tensor.hpp"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace rppbench {

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
     * RppCase allocates the source accordingly.
     * @return Leading offset in bytes; 0 when no halo is needed.
     */
    virtual int srcOffsetBytes(const BenchContext &) const { return 0; }
    /**
     * @brief Extra halo columns to add to the source width.
     * @return Additional stride columns; 0 when no halo is needed.
     */
    virtual int srcAdditionalStride(const BenchContext &) const { return 0; }

    /**
     * @brief Declare the shape of each source tensor (generic-descriptor ops only).
     *
     * The default (empty) selects the legacy image path: RppCase allocates
     * numSrc() image buffers via TensorBuffer::init() from ctx dims/layout, and
     * adapters use src[i].descPtr / src[i].roi. Generic-descriptor ops (transpose,
     * slice, normalize, the voxel/broadcast families) return one TensorSpec per
     * source instead; RppCase allocates each via TensorBuffer::initGeneric(),
     * and adapters use src[i].gdescPtr with src[i].roiTensor or src[i].roi3d.
     * @param ctx The fully-resolved sweep point.
     * @return One spec per source, or empty to use the image path.
     */
    virtual std::vector<TensorSpec> srcSpecs(const BenchContext &) const { return {}; }
    /**
     * @brief Declare the shape of each destination tensor (generic ops only).
     *
     * Empty selects the image path (see srcSpecs()). @return One spec per dest.
     */
    virtual std::vector<TensorSpec> dstSpecs(const BenchContext &) const { return {}; }

    /**
     * @brief How many source buffers to allocate for this op.
     *
     * Most ops read one image; two-source ops (blend, bitwise_and, magnitude, ...)
     * return >1. Every source is allocated with identical dims/dtype/layout and
     * filled independently, and they share one descriptor/ROI (which is what the
     * two-source rppt_* signatures expect). Defaults to 1. Ignored on the generic
     * path, where srcSpecs().size() decides the count.
     * @return Source buffer count (>= 1).
     */
    virtual int numSrc() const { return 1; }
    /**
     * @brief How many destination buffers to allocate for this op.
     * @return Destination buffer count (>= 1).
     */
    virtual int numDst() const { return 1; }

    /**
     * @brief Allocate op-specific param tensors. Called once, outside the timing loop.
     * @param ctx The fully-resolved sweep point.
     * @param src Ready-to-use source buffers (size == numSrc()).
     * @param dst Ready-to-use destination buffers (size == numDst()).
     */
    virtual void setup(const BenchContext &ctx, std::vector<TensorBuffer> &src,
                       std::vector<TensorBuffer> &dst) = 0;

    /**
     * @brief The hot path - issues the rppt_* call. Timed.
     *
     * For HIP the case synchronizes the stream after each call, so run() need
     * only enqueue.
     * @param ctx The fully-resolved sweep point.
     * @param src Source buffers (size == numSrc()).
     * @param dst Destination buffers (size == numDst()).
     * @param handle The RPP handle to issue the call against.
     * @return The rppt_* call's status.
     */
    virtual RppStatus run(const BenchContext &ctx, std::vector<TensorBuffer> &src,
                          std::vector<TensorBuffer> &dst, rppHandle_t handle) = 0;

    /**
     * @brief Free op-specific param tensors. Called once, after the timing loop.
     */
    virtual void teardown() {}
};

// Adapter bases build on the OpAdapter interface above and live in their own
// headers, one per descriptor dialect:
//   - rpp/bench_simple_adapter.hpp    - SimpleOpAdapter, the common
//     single-source/single-destination 4D image (RpptDesc) op.
//   - rpp/bench_generic_adapters.hpp  - GenericOpAdapter / VoxelOpAdapter /
//     BroadcastOpAdapter, the generic (RpptGenericDesc) ops, via the
//     srcSpecs()/dstSpecs() hooks above.
// Multi-source/destination image ops subclass OpAdapter directly (see
// libraries/rpp/ops/bench_bitwise_and.cpp).

using AdapterFactory = std::function<std::unique_ptr<OpAdapter>()>;

// The RPP library's op registry (singleton). Adapters register at static-init
// time; RppLibrary reads it. This table is owned by the RPP module - the core
// has no global op registry (see common/core/bench_library.hpp).
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
 * Place at file scope in a libraries/rpp/ops adapter source file.
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
