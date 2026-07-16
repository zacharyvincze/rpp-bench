/**
 * @file bench_exposure.cpp
 * @brief Adapter for rppt_exposure - one scalar (exposure factor) param per image.
 *
 * Signature:
 *   rppt_exposure(src, srcDesc, dst, dstDesc, exposureFactorTensor,
 *                 roi, roiType, handle, backend)
 */
#include "harness/bench_registry.hpp"
#include <rpp/rppt_tensor_color_augmentations.h>

namespace rppbench {

class ExposureAdapter : public SimpleOpAdapter {
public:
    void setup(const BenchContext &ctx, TensorBuffer &, TensorBuffer &) override {
        isHip_ = ctx.isHip;
        const auto factor = ctx.param<float>("factor", 1.5F);
        factor_ = static_cast<float *>(bench_pinned_alloc(sizeof(float) * ctx.batch, ctx.isHip));
        for (int i = 0; i < ctx.batch; ++i)
            factor_[i] = factor;
    }

    RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                  rppHandle_t handle) override {
        return rppt_exposure(src.data, src.descPtr, dst.data, dst.descPtr, factor_, src.roi,
                             src.roiType, handle, ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(factor_, isHip_);
        factor_ = nullptr;
    }

private:
    float *factor_ = nullptr;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("exposure", ExposureAdapter)

} // namespace rppbench
