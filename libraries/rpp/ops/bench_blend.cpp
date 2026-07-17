/**
 * @file bench_blend.cpp
 * @brief Adapter for rppt_blend - alpha-blend of two sources (src1*alpha + src2*(1-alpha)).
 *
 * A two-source op with a per-image alpha tensor: both sources share one
 * srcDesc/ROI, so it overrides numSrc() to 2 and inherits OpAdapter directly.
 * Supports all dtypes/layouts (c = 1/3). Param: `alpha` (0 <= alpha <= 1).
 *
 * Signature:
 *   rppt_blend(src1, src2, srcDesc, dst, dstDesc, alphaTensor, roi, roiType, handle, backend)
 */
#include "rpp/bench_registry.hpp"
#include <rpp/rppt_tensor_color_augmentations.h>

namespace rppbench {

class BlendAdapter : public OpAdapter {
public:
    int numSrc() const override { return 2; }

    void setup(const BenchContext &ctx, std::vector<TensorBuffer> &,
               std::vector<TensorBuffer> &) override {
        isHip_ = ctx.isHip;
        const auto alpha = ctx.param<float>("alpha", 0.5F);
        alpha_ = static_cast<float *>(bench_pinned_alloc(sizeof(float) * ctx.batch, ctx.isHip));
        for (int i = 0; i < ctx.batch; ++i)
            alpha_[i] = alpha;
    }

    RppStatus run(const BenchContext &ctx, std::vector<TensorBuffer> &src,
                  std::vector<TensorBuffer> &dst, rppHandle_t handle) override {
        return rppt_blend(src[0].data, src[1].data, src[0].descPtr, dst[0].data, dst[0].descPtr,
                          alpha_, src[0].roi, src[0].roiType, handle, ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(alpha_, isHip_);
        alpha_ = nullptr;
    }

private:
    float *alpha_ = nullptr;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("blend", BlendAdapter)

} // namespace rppbench
