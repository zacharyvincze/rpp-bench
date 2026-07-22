/**
 * @file bench_water.cpp
 * @brief Adapter for rppt_water - water/ripple distortion with six per-image params.
 *
 * Signature:
 *   rppt_water(src, srcDesc, dst, dstDesc, amplitudeXTensor, amplitudeYTensor,
 *              frequencyXTensor, frequencyYTensor, phaseXTensor, phaseYTensor,
 *              roi, roiType, handle, backend)
 *
 * The six params are each a Rpp32f* of size batchSize (one value per image),
 * ALWAYS float regardless of the tensor dtype. No dtype/layout/backend
 * restrictions: U8/F16/F32/I8, PKD3/PLN3/PLN1, HOST + HIP are all supported.
 */
#include "harness/bench_simple_adapter.hpp"
#include <rpp/rppt_tensor_effects_augmentations.h>

namespace rppbench {

class WaterAdapter : public SimpleOpAdapter {
public:
    void setup(const BenchContext &ctx, TensorBuffer &, TensorBuffer &) override {
        const auto amplitudeX = ctx.param<float>("amplitude_x", 2.0F);
        const auto amplitudeY = ctx.param<float>("amplitude_y", 5.0F);
        const auto frequencyX = ctx.param<float>("frequency_x", 5.8F);
        const auto frequencyY = ctx.param<float>("frequency_y", 1.2F);
        const auto phaseX = ctx.param<float>("phase_x", 10.0F);
        const auto phaseY = ctx.param<float>("phase_y", 15.0F);
        isHip_ = ctx.isHip;
        amplitudeX_ =
            static_cast<Rpp32f *>(bench_pinned_alloc(sizeof(Rpp32f) * ctx.batch, ctx.isHip));
        amplitudeY_ =
            static_cast<Rpp32f *>(bench_pinned_alloc(sizeof(Rpp32f) * ctx.batch, ctx.isHip));
        frequencyX_ =
            static_cast<Rpp32f *>(bench_pinned_alloc(sizeof(Rpp32f) * ctx.batch, ctx.isHip));
        frequencyY_ =
            static_cast<Rpp32f *>(bench_pinned_alloc(sizeof(Rpp32f) * ctx.batch, ctx.isHip));
        phaseX_ = static_cast<Rpp32f *>(bench_pinned_alloc(sizeof(Rpp32f) * ctx.batch, ctx.isHip));
        phaseY_ = static_cast<Rpp32f *>(bench_pinned_alloc(sizeof(Rpp32f) * ctx.batch, ctx.isHip));
        for (int i = 0; i < ctx.batch; ++i) {
            amplitudeX_[i] = amplitudeX;
            amplitudeY_[i] = amplitudeY;
            frequencyX_[i] = frequencyX;
            frequencyY_[i] = frequencyY;
            phaseX_[i] = phaseX;
            phaseY_[i] = phaseY;
        }
    }

    RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                  rppHandle_t handle) override {
        return rppt_water(src.data, src.descPtr, dst.data, dst.descPtr, amplitudeX_, amplitudeY_,
                          frequencyX_, frequencyY_, phaseX_, phaseY_, src.roi, src.roiType, handle,
                          ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(amplitudeX_, isHip_);
        amplitudeX_ = nullptr;
        bench_pinned_free(amplitudeY_, isHip_);
        amplitudeY_ = nullptr;
        bench_pinned_free(frequencyX_, isHip_);
        frequencyX_ = nullptr;
        bench_pinned_free(frequencyY_, isHip_);
        frequencyY_ = nullptr;
        bench_pinned_free(phaseX_, isHip_);
        phaseX_ = nullptr;
        bench_pinned_free(phaseY_, isHip_);
        phaseY_ = nullptr;
    }

private:
    Rpp32f *amplitudeX_ = nullptr;
    Rpp32f *amplitudeY_ = nullptr;
    Rpp32f *frequencyX_ = nullptr;
    Rpp32f *frequencyY_ = nullptr;
    Rpp32f *phaseX_ = nullptr;
    Rpp32f *phaseY_ = nullptr;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("water", WaterAdapter)

} // namespace rppbench
