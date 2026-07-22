/**
 * @file bench_pixelate.cpp
 * @brief Adapter for rppt_pixelate - block-averaging pixelation effect.
 *
 * Signature:
 *   rppt_pixelate(src, srcDesc, dst, dstDesc, intermediateScratchBufferPtr,
 *                 pixelationPercentage, roi, roiType, handle, backend)
 *
 * Restrictions (from the header): dataType = U8/F16/F32/I8, layout = NCHW/NHWC,
 * c = 1/3 - i.e. all sweep dtypes/layouts are valid, so no overrides are needed.
 *
 * The op needs a per-batch F32 intermediate scratch buffer of minimum size
 * srcDescPtr->n * srcDescPtr->strides.nStride * sizeof(Rpp32f); it is allocated
 * in setup() (buffers are initialised before setup) with bench_pinned_alloc so it
 * lives in HIP memory on a HIP build. pixelationPercentage ranges 0..100.
 */
#include "harness/bench_simple_adapter.hpp"
#include <rpp/rppt_tensor_effects_augmentations.h>

namespace rppbench {

class PixelateAdapter : public SimpleOpAdapter {
public:
    void setup(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &) override {
        pixelationPercentage_ = ctx.param<float>("pixelation_percentage", 87.5F);
        isHip_ = ctx.isHip;
        const size_t scratchBytes =
            static_cast<size_t>(src.descPtr->n) * src.descPtr->strides.nStride * sizeof(Rpp32f);
        scratch_ = bench_pinned_alloc(scratchBytes, ctx.isHip);
    }

    RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                  rppHandle_t handle) override {
        return rppt_pixelate(src.data, src.descPtr, dst.data, dst.descPtr, scratch_,
                             pixelationPercentage_, src.roi, src.roiType, handle, ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(scratch_, isHip_);
        scratch_ = nullptr;
    }

private:
    void *scratch_ = nullptr;
    float pixelationPercentage_ = 87.5F;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("pixelate", PixelateAdapter)

} // namespace rppbench
