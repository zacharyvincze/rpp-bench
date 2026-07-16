/**
 * @file bench_box_filter.cpp
 * @brief Adapter for rppt_box_filter - a kernelSize-scalar box (mean) filter.
 *
 * Like all RPP filter kernels, the HIP path requires the source buffer to carry
 * a leading offset of 12*(kernelSize/2) bytes plus a kernelSize/2 halo column
 * margin for neighbor reads (see bench_gaussian_blur.cpp for the pattern).
 * kernelSize = 3/5/7/9 are the optimized sizes. Only REPLICATE border is supported.
 *
 * Signature:
 *   rppt_box_filter(src, srcDesc, dst, dstDesc, kernelSize, borderType,
 *                   roi, roiType, handle, backend)
 */
#include "harness/bench_registry.hpp"
#include <rpp/rppt_tensor_filter_augmentations.h>

namespace rppbench {

class BoxFilterAdapter : public SimpleOpAdapter {
public:
    void setup(const BenchContext &ctx, TensorBuffer &, TensorBuffer &) override {
        kernelSize_ = static_cast<Rpp32u>(ctx.param<int>("kernel_size", 3));
    }

    RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                  rppHandle_t handle) override {
        return rppt_box_filter(src.data, src.descPtr, dst.data, dst.descPtr, kernelSize_,
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

REGISTER_RPP_BENCH("box_filter", BoxFilterAdapter)

} // namespace rppbench
