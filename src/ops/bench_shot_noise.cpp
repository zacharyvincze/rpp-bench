/**
 * @file bench_shot_noise.cpp
 * @brief Adapter for rppt_shot_noise - Poisson (shot) noise, one factor per image.
 *
 * The shot-noise factor scales the Poisson lambda; a fixed RNG seed keeps the
 * sweep deterministic. One f32 param tensor, exposure-style.
 *
 * Signature:
 *   rppt_shot_noise(src, srcDesc, dst, dstDesc, shotNoiseFactorTensor, seed,
 *                   roi, roiType, handle, backend)
 */
#include "harness/bench_registry.hpp"
#include <rpp/rppt_tensor_effects_augmentations.h>

namespace rppbench {

class ShotNoiseAdapter : public OpAdapter {
public:
    void setup(const BenchContext &ctx, TensorBuffer &, TensorBuffer &) override {
        isHip_ = ctx.isHip;
        seed_ = static_cast<Rpp32u>(ctx.param<int>("seed", 0));
        const auto factor = ctx.param<float>("factor", 80.0F);
        factor_ = static_cast<float *>(bench_pinned_alloc(sizeof(float) * ctx.batch, ctx.isHip));
        for (int i = 0; i < ctx.batch; ++i)
            factor_[i] = factor;
    }

    RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                  rppHandle_t handle) override {
        return rppt_shot_noise(src.data, src.descPtr, dst.data, dst.descPtr, factor_, seed_,
                               src.roi, src.roiType, handle, ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(factor_, isHip_);
        factor_ = nullptr;
    }

private:
    float *factor_ = nullptr;
    Rpp32u seed_ = 0;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("shot_noise", ShotNoiseAdapter)

} // namespace rppbench
