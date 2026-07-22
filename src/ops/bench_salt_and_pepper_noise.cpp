/**
 * @file bench_salt_and_pepper_noise.cpp
 * @brief Adapter for rppt_salt_and_pepper_noise - impulse noise per image.
 *
 * Four f32 param tensors (noise/salt probability, salt/pepper value), all in
 * [0,1], plus a fixed RNG seed for a deterministic sweep.
 *
 * Signature:
 *   rppt_salt_and_pepper_noise(src, srcDesc, dst, dstDesc, noiseProbabilityTensor,
 *                              saltProbabilityTensor, saltValueTensor,
 *                              pepperValueTensor, seed, roi, roiType, handle, backend)
 */
#include "harness/bench_simple_adapter.hpp"
#include <rpp/rppt_tensor_effects_augmentations.h>

namespace rppbench {

class SaltAndPepperNoiseAdapter : public SimpleOpAdapter {
public:
    void setup(const BenchContext &ctx, TensorBuffer &, TensorBuffer &) override {
        isHip_ = ctx.isHip;
        seed_ = static_cast<Rpp32u>(ctx.param<int>("seed", 0));
        const auto noiseProb = ctx.param<float>("noise_probability", 0.1F);
        const auto saltProb = ctx.param<float>("salt_probability", 0.5F);
        const auto saltValue = ctx.param<float>("salt_value", 1.0F);
        const auto pepperValue = ctx.param<float>("pepper_value", 0.0F);
        noiseProb_ = alloc(ctx, noiseProb);
        saltProb_ = alloc(ctx, saltProb);
        saltValue_ = alloc(ctx, saltValue);
        pepperValue_ = alloc(ctx, pepperValue);
    }

    RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                  rppHandle_t handle) override {
        return rppt_salt_and_pepper_noise(src.data, src.descPtr, dst.data, dst.descPtr, noiseProb_,
                                          saltProb_, saltValue_, pepperValue_, seed_, src.roi,
                                          src.roiType, handle, ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(noiseProb_, isHip_);
        noiseProb_ = nullptr;
        bench_pinned_free(saltProb_, isHip_);
        saltProb_ = nullptr;
        bench_pinned_free(saltValue_, isHip_);
        saltValue_ = nullptr;
        bench_pinned_free(pepperValue_, isHip_);
        pepperValue_ = nullptr;
    }

private:
    float *alloc(const BenchContext &ctx, float value) {
        auto *p = static_cast<float *>(bench_pinned_alloc(sizeof(float) * ctx.batch, ctx.isHip));
        for (int i = 0; i < ctx.batch; ++i)
            p[i] = value;
        return p;
    }

    float *noiseProb_ = nullptr;
    float *saltProb_ = nullptr;
    float *saltValue_ = nullptr;
    float *pepperValue_ = nullptr;
    Rpp32u seed_ = 0;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("salt_and_pepper_noise", SaltAndPepperNoiseAdapter)

} // namespace rppbench
