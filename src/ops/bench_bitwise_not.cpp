/**
 * @file bench_bitwise_not.cpp
 * @brief Adapter for rppt_bitwise_not - per-pixel bitwise complement, no params.
 *
 * U8 only (the kernel is defined for 8-bit data); works on 1- or 3-channel
 * layouts. A single-source op with an ROI but no param tensors, so it clones
 * the copy adapter with a dtype constraint.
 *
 * Signature:
 *   rppt_bitwise_not(src, srcDesc, dst, dstDesc, roi, roiType, handle, backend)
 */
#include "harness/bench_registry.hpp"
#include <rpp/rppt_tensor_bitwise_operations.h>

namespace rppbench {

class BitwiseNotAdapter : public OpAdapter {
public:
    std::vector<RpptDataType> supportedDtypes() const override { return {RpptDataType::U8}; }

    void setup(const BenchContext &, TensorBuffer &, TensorBuffer &) override {}

    RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                  rppHandle_t handle) override {
        return rppt_bitwise_not(src.data, src.descPtr, dst.data, dst.descPtr, src.roi, src.roiType,
                                handle, ctx.backend);
    }
};

REGISTER_RPP_BENCH("bitwise_not", BitwiseNotAdapter)

} // namespace rppbench
