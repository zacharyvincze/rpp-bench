/**
 * @file bench_snow.cpp
 * @brief Adapter for rppt_snow - per-image brightness coeff + threshold + dark-mode.
 *
 * darkMode is an Rpp32s tensor (0/1); the other two are float tensors.
 *
 * Signature:
 *   rppt_snow(src, srcDesc, dst, dstDesc, brightnessCoefficient, snowThreshold,
 *             darkMode, roi, roiType, handle, backend)
 */
#include "harness/bench_simple_adapter.hpp"
#include <rpp/rppt_tensor_effects_augmentations.h>

namespace rppbench {

class SnowAdapter : public SimpleOpAdapter {
public:
    void setup(const BenchContext &ctx, TensorBuffer &, TensorBuffer &) override {
        isHip_ = ctx.isHip;
        const auto brightness = ctx.param<float>("brightness_coefficient", 2.5F);  // (1, 4]
        const auto threshold = ctx.param<float>("threshold", 0.5F);                // (0, 1]
        const auto darkMode = static_cast<Rpp32s>(ctx.param<int>("dark_mode", 0)); // 0/1
        brightness_ =
            static_cast<float *>(bench_pinned_alloc(sizeof(float) * ctx.batch, ctx.isHip));
        threshold_ = static_cast<float *>(bench_pinned_alloc(sizeof(float) * ctx.batch, ctx.isHip));
        darkMode_ =
            static_cast<Rpp32s *>(bench_pinned_alloc(sizeof(Rpp32s) * ctx.batch, ctx.isHip));
        for (int i = 0; i < ctx.batch; ++i) {
            brightness_[i] = brightness;
            threshold_[i] = threshold;
            darkMode_[i] = darkMode;
        }
    }

    RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                  rppHandle_t handle) override {
        return rppt_snow(src.data, src.descPtr, dst.data, dst.descPtr, brightness_, threshold_,
                         darkMode_, src.roi, src.roiType, handle, ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(brightness_, isHip_);
        brightness_ = nullptr;
        bench_pinned_free(threshold_, isHip_);
        threshold_ = nullptr;
        bench_pinned_free(darkMode_, isHip_);
        darkMode_ = nullptr;
    }

private:
    float *brightness_ = nullptr;
    float *threshold_ = nullptr;
    Rpp32s *darkMode_ = nullptr;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("snow", SnowAdapter)

} // namespace rppbench
