/**
 * @file bench_contrast.cpp
 * @brief Adapter for rppt_contrast - per-image contrast factor + contrast center.
 *
 * Signature:
 *   rppt_contrast(src, srcDesc, dst, dstDesc, contrastFactorTensor,
 *                 contrastCenterTensor, roi, roiType, handle, backend)
 */
#include "rpp/bench_simple_adapter.hpp"
#include <rpp/rppt_tensor_color_augmentations.h>

namespace rppbench {

class ContrastAdapter : public SimpleOpAdapter {
public:
    void setup(const BenchContext &ctx, TensorBuffer &, TensorBuffer &) override {
        isHip_ = ctx.isHip;
        const auto factor = ctx.param<float>("factor", 2.96F);
        const auto center = ctx.param<float>("center", 128.0F);
        factor_ = static_cast<float *>(bench_pinned_alloc(sizeof(float) * ctx.batch, ctx.isHip));
        center_ = static_cast<float *>(bench_pinned_alloc(sizeof(float) * ctx.batch, ctx.isHip));
        for (int i = 0; i < ctx.batch; ++i) {
            factor_[i] = factor;
            center_[i] = center;
        }
    }

    RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                  rppHandle_t handle) override {
        return rppt_contrast(src.data, src.descPtr, dst.data, dst.descPtr, factor_, center_,
                             src.roi, src.roiType, handle, ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(factor_, isHip_);
        factor_ = nullptr;
        bench_pinned_free(center_, isHip_);
        center_ = nullptr;
    }

private:
    float *factor_ = nullptr;
    float *center_ = nullptr;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("contrast", ContrastAdapter)

} // namespace rppbench
