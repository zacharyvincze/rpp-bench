/**
 * @file bench_median_filter.cpp
 * @brief Adapter for rppt_median_filter - a kernelSize-scalar median filter.
 *
 * Same HIP halo requirement as the other filters (12*(kernelSize/2) byte offset
 * + kernelSize/2 halo columns). kernelSize = 3/5/7/9 are the optimized sizes.
 * Only REPLICATE border is supported.
 *
 * Signature:
 *   rppt_median_filter(src, srcDesc, dst, dstDesc, kernelSize, borderType,
 *                      roi, roiType, handle, backend)
 */
#include "harness/bench_simple_adapter.hpp"
#include <rpp/rppt_tensor_filter_augmentations.h>

namespace rppbench {

class MedianFilterAdapter : public SimpleOpAdapter {
public:
    void setup(const BenchContext &ctx, TensorBuffer &, TensorBuffer &) override {
        kernelSize_ = static_cast<Rpp32u>(ctx.param<int>("kernel_size", 3));
    }

    RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                  rppHandle_t handle) override {
        return rppt_median_filter(src.data, src.descPtr, dst.data, dst.descPtr, kernelSize_,
                                  RpptImageBorderType::REPLICATE, src.roi, src.roiType, handle,
                                  ctx.backend);
    }

    int srcOffsetBytes(const BenchContext &ctx) const override {
        return 12 * (ctx.param<int>("kernel_size", 3) / 2);
    }
    int srcAdditionalStride(const BenchContext &ctx) const override {
        return ctx.param<int>("kernel_size", 3) / 2;
    }

private:
    Rpp32u kernelSize_ = 3;
};

REGISTER_RPP_BENCH("median_filter", MedianFilterAdapter)

} // namespace rppbench
