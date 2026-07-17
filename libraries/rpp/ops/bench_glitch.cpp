/**
 * @file bench_glitch.cpp
 * @brief Adapter for rppt_glitch - per-channel spatial offset (chromatic shift).
 *
 * A single RpptChannelOffsets (x/y shift for each of R/G/B) is shared across the
 * batch, per the API. RGB augmentation, so 3-channel layouts only.
 *
 * Signature:
 *   rppt_glitch(src, srcDesc, dst, dstDesc, rgbOffsets, roi, roiType, handle, backend)
 */
#include "rpp/bench_simple_adapter.hpp"
#include <rpp/rppt_tensor_effects_augmentations.h>

namespace rppbench {

class GlitchAdapter : public SimpleOpAdapter {
public:
    std::vector<Layout> supportedLayouts() const override { return {Layout::PKD3, Layout::PLN3}; }

    void setup(const BenchContext &ctx, TensorBuffer &, TensorBuffer &) override {
        isHip_ = ctx.isHip;
        offsets_ = static_cast<RpptChannelOffsets *>(
            bench_pinned_alloc(sizeof(RpptChannelOffsets), ctx.isHip));
        offsets_->r = {ctx.param<int>("r_x", 8), ctx.param<int>("r_y", 8)};
        offsets_->g = {ctx.param<int>("g_x", 0), ctx.param<int>("g_y", 0)};
        offsets_->b = {ctx.param<int>("b_x", -8), ctx.param<int>("b_y", -8)};
    }

    RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                  rppHandle_t handle) override {
        return rppt_glitch(src.data, src.descPtr, dst.data, dst.descPtr, offsets_, src.roi,
                           src.roiType, handle, ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(offsets_, isHip_);
        offsets_ = nullptr;
    }

private:
    RpptChannelOffsets *offsets_ = nullptr;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("glitch", GlitchAdapter)

} // namespace rppbench
