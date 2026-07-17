/**
 * @file bench_simple_adapter.hpp
 * @brief Adapter base for the common single-source, single-destination image op.
 *
 * The image-dialect counterpart to bench_generic_adapters.hpp: where those bases
 * serve the generic (RpptGenericDesc) ops, SimpleOpAdapter serves the 4D image
 * (RpptDesc) ops that read one source and write one destination - the vast
 * majority of `rppt_*` operators. It forwards the collection-based
 * setup()/run() hooks of OpAdapter to single-buffer signatures so those adapters
 * never touch the src/dst vectors. Multi-source/destination image ops (blend,
 * bitwise_and, crop_and_patch, ...) subclass OpAdapter directly instead and live
 * against bench_registry.hpp.
 */
#ifndef RPP_BENCH_SIMPLE_ADAPTER_HPP
#define RPP_BENCH_SIMPLE_ADAPTER_HPP

#include "harness/bench_registry.hpp"

#include <vector>

namespace rppbench {

/**
 * @brief Base for the common single-source, single-destination op.
 *
 * Forwards the collection-based setup()/run() to the single-buffer signatures
 * the vast majority of adapters use, so those adapters need only inherit this
 * instead of OpAdapter. Multi-source/destination ops subclass OpAdapter directly
 * and override numSrc()/numDst() plus the collection forms.
 */
class SimpleOpAdapter : public OpAdapter {
public:
    void setup(const BenchContext &ctx, std::vector<TensorBuffer> &src,
               std::vector<TensorBuffer> &dst) final {
        setup(ctx, src[0], dst[0]);
    }
    RppStatus run(const BenchContext &ctx, std::vector<TensorBuffer> &src,
                  std::vector<TensorBuffer> &dst, rppHandle_t handle) final {
        return run(ctx, src[0], dst[0], handle);
    }

    /**
     * @brief Allocate op-specific param tensors. Called once, outside the timing loop.
     * @param ctx The fully-resolved sweep point.
     * @param src Ready-to-use source buffer.
     * @param dst Ready-to-use destination buffer.
     */
    virtual void setup(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst) = 0;
    /**
     * @brief The hot path - issues the rppt_* call. Timed. See OpAdapter::run.
     * @param ctx The fully-resolved sweep point.
     * @param src Source buffer.
     * @param dst Destination buffer.
     * @param handle The RPP handle to issue the call against.
     * @return The rppt_* call's status.
     */
    virtual RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                          rppHandle_t handle) = 0;
};

} // namespace rppbench

#endif // RPP_BENCH_SIMPLE_ADAPTER_HPP
