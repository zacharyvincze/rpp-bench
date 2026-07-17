/**
 * @file bench_emboss.cpp
 * @brief Adapter for rppt_emboss - a per-image strength tensor + kernelSize scalar.
 *
 * Same HIP halo requirement as the other filters (12*(kernelSize/2) byte offset
 * + kernelSize/2 halo columns). kernelSize = 3/5/7/9. Only REPLICATE border.
 *
 * Signature:
 *   rppt_emboss(src, srcDesc, dst, dstDesc, strength, kernelSize, borderType,
 *               roi, roiType, handle, backend)
 */
#include "rpp/bench_simple_adapter.hpp"
#include <rpp/rppt_tensor_filter_augmentations.h>

namespace rppbench {

class EmbossAdapter : public SimpleOpAdapter {
public:
    void setup(const BenchContext &ctx, TensorBuffer &, TensorBuffer &) override {
        isHip_ = ctx.isHip;
        kernelSize_ = static_cast<Rpp32u>(ctx.param<int>("kernel_size", 3));
        const auto strength = ctx.param<float>("strength", 1.0F);
        strength_ = static_cast<float *>(bench_pinned_alloc(sizeof(float) * ctx.batch, ctx.isHip));
        for (int i = 0; i < ctx.batch; ++i)
            strength_[i] = strength;
    }

    RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                  rppHandle_t handle) override {
        return rppt_emboss(src.data, src.descPtr, dst.data, dst.descPtr, strength_, kernelSize_,
                           RpptImageBorderType::REPLICATE, src.roi, src.roiType, handle,
                           ctx.backend);
    }

    int srcOffsetBytes(const BenchContext &ctx) const override {
        return 12 * (ctx.param<int>("kernel_size", 3) / 2);
    }
    int srcAdditionalStride(const BenchContext &ctx) const override {
        return ctx.param<int>("kernel_size", 3) / 2;
    }

    void teardown() override {
        bench_pinned_free(strength_, isHip_);
        strength_ = nullptr;
    }

private:
    float *strength_ = nullptr;
    Rpp32u kernelSize_ = 3;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("emboss", EmbossAdapter)

} // namespace rppbench
