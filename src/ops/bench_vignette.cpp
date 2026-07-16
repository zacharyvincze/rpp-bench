/**
 * @file bench_vignette.cpp
 * @brief Adapter for rppt_vignette - one scalar (intensity) param per image.
 *
 * Signature:
 *   rppt_vignette(src, srcDesc, dst, dstDesc, vignetteIntensityTensor,
 *                 roi, roiType, handle, backend)
 */
#include "harness/bench_registry.hpp"
#include <rpp/rppt_tensor_effects_augmentations.h>

namespace rppbench {

class VignetteAdapter : public SimpleOpAdapter {
public:
    void setup(const BenchContext &ctx, TensorBuffer &, TensorBuffer &) override {
        isHip_ = ctx.isHip;
        const auto intensity = ctx.param<float>("intensity", 6.0F);
        intensity_ = static_cast<float *>(bench_pinned_alloc(sizeof(float) * ctx.batch, ctx.isHip));
        for (int i = 0; i < ctx.batch; ++i)
            intensity_[i] = intensity;
    }

    RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                  rppHandle_t handle) override {
        return rppt_vignette(src.data, src.descPtr, dst.data, dst.descPtr, intensity_, src.roi,
                             src.roiType, handle, ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(intensity_, isHip_);
        intensity_ = nullptr;
    }

private:
    float *intensity_ = nullptr;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("vignette", VignetteAdapter)

} // namespace rppbench
