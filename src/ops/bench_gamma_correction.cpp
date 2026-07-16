/**
 * @file bench_gamma_correction.cpp
 * @brief Adapter for rppt_gamma_correction - one scalar (gamma) param per image.
 *
 * Signature:
 *   rppt_gamma_correction(src, srcDesc, dst, dstDesc, gammaTensor,
 *                         roi, roiType, handle, backend)
 */
#include "harness/bench_registry.hpp"
#include <rpp/rppt_tensor_color_augmentations.h>

namespace rppbench {

class GammaCorrectionAdapter : public SimpleOpAdapter {
public:
    void setup(const BenchContext &ctx, TensorBuffer &, TensorBuffer &) override {
        isHip_ = ctx.isHip;
        const auto gamma = ctx.param<float>("gamma", 0.5F);
        gamma_ = static_cast<float *>(bench_pinned_alloc(sizeof(float) * ctx.batch, ctx.isHip));
        for (int i = 0; i < ctx.batch; ++i)
            gamma_[i] = gamma;
    }

    RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                  rppHandle_t handle) override {
        return rppt_gamma_correction(src.data, src.descPtr, dst.data, dst.descPtr, gamma_, src.roi,
                                     src.roiType, handle, ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(gamma_, isHip_);
        gamma_ = nullptr;
    }

private:
    float *gamma_ = nullptr;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("gamma_correction", GammaCorrectionAdapter)

} // namespace rppbench
