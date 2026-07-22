/**
 * @file bench_non_linear_blend.cpp
 * @brief Adapter for rppt_non_linear_blend - Gaussian-mask blend of two sources.
 *
 * A two-source op with a per-image stdDev tensor driving a centred Gaussian
 * blend mask: both sources share one srcDesc/ROI, so it overrides numSrc() to 2
 * and inherits OpAdapter directly. Supports all dtypes/layouts (c = 1/3). Param:
 * `std_dev` (>= 0, in pixel units).
 *
 * Signature:
 *   rppt_non_linear_blend(src1, src2, srcDesc, dst, dstDesc, stdDevTensor,
 *                         roi, roiType, handle, backend)
 */
#include "harness/bench_registry.hpp"
#include <rpp/rppt_tensor_effects_augmentations.h>

namespace rppbench {

class NonLinearBlendAdapter : public OpAdapter {
public:
    int numSrc() const override { return 2; }

    void setup(const BenchContext &ctx, std::vector<TensorBuffer> &,
               std::vector<TensorBuffer> &) override {
        isHip_ = ctx.isHip;
        const auto stdDev = ctx.param<float>("std_dev", 50.0F);
        stdDev_ = static_cast<float *>(bench_pinned_alloc(sizeof(float) * ctx.batch, ctx.isHip));
        for (int i = 0; i < ctx.batch; ++i)
            stdDev_[i] = stdDev;
    }

    RppStatus run(const BenchContext &ctx, std::vector<TensorBuffer> &src,
                  std::vector<TensorBuffer> &dst, rppHandle_t handle) override {
        return rppt_non_linear_blend(src[0].data, src[1].data, src[0].descPtr, dst[0].data,
                                     dst[0].descPtr, stdDev_, src[0].roi, src[0].roiType, handle,
                                     ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(stdDev_, isHip_);
        stdDev_ = nullptr;
    }

private:
    float *stdDev_ = nullptr;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("non_linear_blend", NonLinearBlendAdapter)

} // namespace rppbench
