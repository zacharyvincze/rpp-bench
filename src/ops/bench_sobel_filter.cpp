/**
 * @file bench_sobel_filter.cpp
 * @brief Adapter for rppt_sobel_filter - sobelType + kernelSize scalar gradient filter.
 *
 * The destination is single-channel (dstDescPtr restriction: layout = NCHW, c = 1),
 * while the harness allocates dst with the same layout as src. Restricting the
 * sweep to PLN1 keeps both src and dst single-channel NCHW and satisfies that.
 * Same HIP halo requirement as the other filters. kernelSize = 3/5/7;
 * sobelType = 0 (X) / 1 (Y) / 2 (XY).
 *
 * Signature:
 *   rppt_sobel_filter(src, srcDesc, dst, dstDesc, sobelType, kernelSize,
 *                     roi, roiType, handle, backend)
 */
#include "harness/bench_registry.hpp"
#include <rpp/rppt_tensor_filter_augmentations.h>

namespace rppbench {

class SobelFilterAdapter : public SimpleOpAdapter {
public:
    std::vector<Layout> supportedLayouts() const override { return {Layout::PLN1}; }

    void setup(const BenchContext &ctx, TensorBuffer &, TensorBuffer &) override {
        kernelSize_ = static_cast<Rpp32u>(ctx.param<int>("kernel_size", 3));
        sobelType_ = static_cast<Rpp32u>(ctx.param<int>("sobel_type", 0));
    }

    RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                  rppHandle_t handle) override {
        return rppt_sobel_filter(src.data, src.descPtr, dst.data, dst.descPtr, sobelType_,
                                 kernelSize_, src.roi, src.roiType, handle, ctx.backend);
    }

    int srcOffsetBytes(const BenchContext &ctx) const override {
        return 12 * (ctx.param<int>("kernel_size", 3) / 2);
    }
    int srcAdditionalStride(const BenchContext &ctx) const override {
        return ctx.param<int>("kernel_size", 3) / 2;
    }

private:
    Rpp32u kernelSize_ = 3;
    Rpp32u sobelType_ = 0;
};

REGISTER_RPP_BENCH("sobel_filter", SobelFilterAdapter)

} // namespace rppbench
