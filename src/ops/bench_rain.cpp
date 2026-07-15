/**
 * @file bench_rain.cpp
 * @brief Adapter for rppt_rain - overlays a synthetic rain layer.
 *
 * Rain geometry is scalar (percentage, streak width/height, slant angle); only
 * the per-image alpha blend weight is a tensor.
 *
 * Memory note: rain stages its rain layer through the RPP handle's host scratch
 * buffer, which RPP sizes at ~0.4 GB * nBatchSize (one 8K RGB image per batch
 * element). rppCreate is called with the case's batch, so large batches demand
 * proportionally large host RAM; if that malloc fails RPP leaves the scratch
 * NULL and the kernel dereferences it. Keep rain's batch modest on low-RAM hosts
 * (config/example.json caps it) - this is an RPP allocation quirk, not a limit
 * of the op itself.
 *
 * Signature:
 *   rppt_rain(src, srcDesc, dst, dstDesc, rainPercentage, rainWidth, rainHeight,
 *             slantAngle, alpha, roi, roiType, handle, backend)
 */
#include "harness/bench_registry.hpp"
#include <rpp/rppt_tensor_effects_augmentations.h>

namespace rppbench {

class RainAdapter : public OpAdapter {
public:
    void setup(const BenchContext &ctx, TensorBuffer &, TensorBuffer &) override {
        isHip_ = ctx.isHip;
        rainPercentage_ = ctx.param<float>("rain_percentage", 0.03F);
        rainWidth_ = static_cast<Rpp32u>(ctx.param<int>("rain_width", 1));
        rainHeight_ = static_cast<Rpp32u>(ctx.param<int>("rain_height", 15));
        slantAngle_ = ctx.param<float>("slant_angle", 0.0F);
        const auto alpha = ctx.param<float>("alpha", 0.5F);
        alpha_ = static_cast<float *>(bench_pinned_alloc(sizeof(float) * ctx.batch, ctx.isHip));
        for (int i = 0; i < ctx.batch; ++i)
            alpha_[i] = alpha;
    }

    RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                  rppHandle_t handle) override {
        return rppt_rain(src.data, src.descPtr, dst.data, dst.descPtr, rainPercentage_, rainWidth_,
                         rainHeight_, slantAngle_, alpha_, src.roi, src.roiType, handle,
                         ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(alpha_, isHip_);
        alpha_ = nullptr;
    }

private:
    float *alpha_ = nullptr;
    Rpp32f rainPercentage_ = 0.0F;
    Rpp32u rainWidth_ = 0;
    Rpp32u rainHeight_ = 0;
    Rpp32f slantAngle_ = 0.0F;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("rain", RainAdapter)

} // namespace rppbench
