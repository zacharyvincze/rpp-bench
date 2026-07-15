/**
 * @file bench_brightness.cpp
 * @brief Adapter for rppt_brightness - scalar (alpha, beta) params per image.
 *
 * Signature:
 *   rppt_brightness(src, srcDesc, dst, dstDesc, alphaTensor, betaTensor,
 *                   roi, roiType, handle, backend)
 */
#include "harness/bench_registry.hpp"
#include <rpp/rppt_tensor_color_augmentations.h>

namespace rppbench {

class BrightnessAdapter : public OpAdapter {
public:
    void setup(const BenchContext &ctx, TensorBuffer &, TensorBuffer &) override {
        const auto alpha = ctx.param<float>("alpha", 1.75F);
        const auto beta = ctx.param<float>("beta", 50.0F);
        alpha_ = static_cast<float *>(bench_pinned_alloc(sizeof(float) * ctx.batch, ctx.isHip));
        beta_ = static_cast<float *>(bench_pinned_alloc(sizeof(float) * ctx.batch, ctx.isHip));
        isHip_ = ctx.isHip;
        for (int i = 0; i < ctx.batch; ++i) {
            alpha_[i] = alpha;
            beta_[i] = beta;
        }
    }

    RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                  rppHandle_t handle) override {
        return rppt_brightness(src.data, src.descPtr, dst.data, dst.descPtr, alpha_, beta_, src.roi,
                               src.roiType, handle, ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(alpha_, isHip_);
        alpha_ = nullptr;
        bench_pinned_free(beta_, isHip_);
        beta_ = nullptr;
    }

private:
    float *alpha_ = nullptr;
    float *beta_ = nullptr;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("brightness", BrightnessAdapter)

} // namespace rppbench
