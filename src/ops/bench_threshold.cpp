/**
 * @file bench_threshold.cpp
 * @brief Adapter for rppt_threshold - per-pixel black/white binary mask by [min,max] bounds.
 *
 * Signature:
 *   rppt_threshold(src, srcDesc, dst, dstDesc, minTensor, maxTensor,
 *                  roi, roiType, handle, backend)
 *
 * Quirks:
 *  - minTensor/maxTensor are each Rpp32f* of length batch*channels (per-sample,
 *    per-channel cutoffs) and are ALWAYS float regardless of src dtype. Channel
 *    count comes from the layout (3 for PKD3/PLN3, 1 for PLN1).
 *  - Valid value ranges are dtype-dependent (U8 0..255, F16/F32 0..1, I8 -128..127),
 *    but timing is not data-dependent, so representative U8-range defaults suffice.
 */
#include "harness/bench_simple_adapter.hpp"
#include <rpp/rppt_tensor_statistical_operations.h>

namespace rppbench {

class ThresholdAdapter : public SimpleOpAdapter {
public:
    void setup(const BenchContext &ctx, TensorBuffer &, TensorBuffer &) override {
        const auto minVal = ctx.param<float>("min", 50.0F);
        const auto maxVal = ctx.param<float>("max", 200.0F);
        const int channels = layout_info(ctx.layout).channels;
        const int count = ctx.batch * channels;
        isHip_ = ctx.isHip;
        min_ = static_cast<Rpp32f *>(bench_pinned_alloc(sizeof(Rpp32f) * count, ctx.isHip));
        max_ = static_cast<Rpp32f *>(bench_pinned_alloc(sizeof(Rpp32f) * count, ctx.isHip));
        for (int i = 0; i < count; ++i) {
            min_[i] = minVal;
            max_[i] = maxVal;
        }
    }

    RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                  rppHandle_t handle) override {
        return rppt_threshold(src.data, src.descPtr, dst.data, dst.descPtr, min_, max_, src.roi,
                              src.roiType, handle, ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(min_, isHip_);
        min_ = nullptr;
        bench_pinned_free(max_, isHip_);
        max_ = nullptr;
    }

private:
    Rpp32f *min_ = nullptr;
    Rpp32f *max_ = nullptr;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("threshold", ThresholdAdapter)

} // namespace rppbench
