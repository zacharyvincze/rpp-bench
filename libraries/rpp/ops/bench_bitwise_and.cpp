/**
 * @file bench_bitwise_and.cpp
 * @brief Adapter for rppt_bitwise_and - per-pixel bitwise AND of two sources.
 *
 * The first two-source op: it reads srcPtr1 and srcPtr2 (both sharing a single
 * srcDesc/ROI) into one destination, so it overrides numSrc() to 2 and inherits
 * OpAdapter directly (not SimpleOpAdapter) to receive both source buffers. U8
 * only (the kernel is defined for 8-bit data); works on 1- or 3-channel layouts.
 *
 * Signature:
 *   rppt_bitwise_and(src1, src2, srcDesc, dst, dstDesc, roi, roiType, handle, backend)
 */
#include "rpp/bench_registry.hpp"
#include <rpp/rppt_tensor_bitwise_operations.h>

namespace rppbench {

class BitwiseAndAdapter : public OpAdapter {
public:
    int numSrc() const override { return 2; }
    std::vector<RpptDataType> supportedDtypes() const override { return {RpptDataType::U8}; }

    void setup(const BenchContext &, std::vector<TensorBuffer> &,
               std::vector<TensorBuffer> &) override {}

    RppStatus run(const BenchContext &ctx, std::vector<TensorBuffer> &src,
                  std::vector<TensorBuffer> &dst, rppHandle_t handle) override {
        return rppt_bitwise_and(src[0].data, src[1].data, src[0].descPtr, dst[0].data,
                                dst[0].descPtr, src[0].roi, src[0].roiType, handle, ctx.backend);
    }
};

REGISTER_RPP_BENCH("bitwise_and", BitwiseAndAdapter)

} // namespace rppbench
