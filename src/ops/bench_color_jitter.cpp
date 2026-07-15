// Adapter for rppt_color_jitter - fused brightness/contrast/hue/saturation.
//
// Same four per-image float tensors as color_twist; the two ops differ in the
// order/math RPP applies internally, not in their adapter shape.
//
// Signature:
//   rppt_color_jitter(src, srcDesc, dst, dstDesc, brightnessTensor,
//                     contrastTensor, hueTensor, saturationTensor,
//                     roi, roiType, handle, backend)
#include "harness/bench_registry.hpp"
#include <rpp/rppt_tensor_color_augmentations.h>

namespace rppbench {

class ColorJitterAdapter : public OpAdapter {
public:
    void setup(const BenchContext &ctx, TensorBuffer &, TensorBuffer &) override {
        isHip_ = ctx.isHip;
        const auto brightness = ctx.param<float>("brightness", 1.5F);
        const auto contrast = ctx.param<float>("contrast", 1.2F);
        const auto hue = ctx.param<float>("hue", 90.0F);
        const auto saturation = ctx.param<float>("saturation", 1.5F);
        brightness_ =
            static_cast<float *>(bench_pinned_alloc(sizeof(float) * ctx.batch, ctx.isHip));
        contrast_ = static_cast<float *>(bench_pinned_alloc(sizeof(float) * ctx.batch, ctx.isHip));
        hue_ = static_cast<float *>(bench_pinned_alloc(sizeof(float) * ctx.batch, ctx.isHip));
        saturation_ =
            static_cast<float *>(bench_pinned_alloc(sizeof(float) * ctx.batch, ctx.isHip));
        for (int i = 0; i < ctx.batch; ++i) {
            brightness_[i] = brightness;
            contrast_[i] = contrast;
            hue_[i] = hue;
            saturation_[i] = saturation;
        }
    }

    RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                  rppHandle_t handle) override {
        return rppt_color_jitter(src.data, src.descPtr, dst.data, dst.descPtr, brightness_,
                                 contrast_, hue_, saturation_, src.roi, src.roiType, handle,
                                 ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(brightness_, isHip_);
        brightness_ = nullptr;
        bench_pinned_free(contrast_, isHip_);
        contrast_ = nullptr;
        bench_pinned_free(hue_, isHip_);
        hue_ = nullptr;
        bench_pinned_free(saturation_, isHip_);
        saturation_ = nullptr;
    }

private:
    float *brightness_ = nullptr;
    float *contrast_ = nullptr;
    float *hue_ = nullptr;
    float *saturation_ = nullptr;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("color_jitter", ColorJitterAdapter)

} // namespace rppbench
