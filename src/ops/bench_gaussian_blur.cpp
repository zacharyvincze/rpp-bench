/**
 * @file bench_gaussian_blur.cpp
 * @brief Adapter for rppt_gaussian_filter - per-image std-dev + a kernel-size param.
 *
 * Signature:
 *   rppt_gaussian_filter(src, srcDesc, dst, dstDesc, stdDevTensor, kernelSize,
 *                        borderType, roi, roiType, handle, backend)
 */
#include "harness/bench_simple_adapter.hpp"
#include <rpp/rppt_tensor_filter_augmentations.h>

namespace rppbench {

class GaussianBlurAdapter : public SimpleOpAdapter {
public:
    void setup(const BenchContext &ctx, TensorBuffer &, TensorBuffer &) override {
        isHip_ = ctx.isHip;
        kernelSize_ = static_cast<Rpp32u>(ctx.param<int>("kernel_size", 3));
        const auto stdDev = ctx.param<float>("std_dev", 1.0F);
        stdDev_ = static_cast<float *>(bench_pinned_alloc(sizeof(float) * ctx.batch, ctx.isHip));
        for (int i = 0; i < ctx.batch; ++i)
            stdDev_[i] = stdDev;
    }

    RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                  rppHandle_t handle) override {
        return rppt_gaussian_filter(src.data, src.descPtr, dst.data, dst.descPtr, stdDev_,
                                    kernelSize_, RpptImageBorderType::REPLICATE, src.roi,
                                    src.roiType, handle, ctx.backend);
    }

    // HIP gaussian_filter requires srcDescPtr->offsetInBytes >= 12*(kernelSize/2)
    // and a matching halo of kernelSize/2 columns for neighbor reads.
    int srcOffsetBytes(const BenchContext &ctx) const override {
        return 12 * (ctx.param<int>("kernel_size", 3) / 2);
    }
    int srcAdditionalStride(const BenchContext &ctx) const override {
        return ctx.param<int>("kernel_size", 3) / 2;
    }

    void teardown() override {
        bench_pinned_free(stdDev_, isHip_);
        stdDev_ = nullptr;
    }

private:
    float *stdDev_ = nullptr;
    Rpp32u kernelSize_ = 3;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("gaussian_blur", GaussianBlurAdapter)

} // namespace rppbench
