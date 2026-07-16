/**
 * @file bench_erase.cpp
 * @brief Adapter for rppt_erase - fill rectangular regions with a solid colour.
 *
 * Uses one erase box per image (a centred quarter-area rectangle) so the param
 * tensors stay simple: an RpptRoiLtrb anchor-box tensor (one per box), an RGB
 * colours tensor (3 values per box, in the dtype's range - zero-filled = black),
 * and a per-image box-count tensor (all 1). 3-channel layouts only (RGB colour).
 *
 * Signature:
 *   rppt_erase(src, srcDesc, dst, dstDesc, anchorBoxInfoTensor, colorsTensor,
 *              numBoxesTensor, roi, roiType, handle, backend)
 */
#include "harness/bench_registry.hpp"
#include <rpp/rppt_tensor_effects_augmentations.h>

#include <cstring>

namespace rppbench {

class EraseAdapter : public SimpleOpAdapter {
public:
    std::vector<Layout> supportedLayouts() const override { return {Layout::PKD3, Layout::PLN3}; }

    void setup(const BenchContext &ctx, TensorBuffer &, TensorBuffer &) override {
        isHip_ = ctx.isHip;
        constexpr int kChannels = 3; // one RGB colour per box

        boxes_ = static_cast<RpptRoiLtrb *>(
            bench_pinned_alloc(sizeof(RpptRoiLtrb) * ctx.batch, ctx.isHip));
        numBoxes_ =
            static_cast<Rpp32u *>(bench_pinned_alloc(sizeof(Rpp32u) * ctx.batch, ctx.isHip));
        // Colours: 3 values per box (1 box/image), in the source dtype; black.
        colors_ = bench_pinned_alloc(
            static_cast<size_t>(ctx.batch) * kChannels * dtype_size(ctx.dtype), ctx.isHip);
        std::memset(colors_, 0, static_cast<size_t>(ctx.batch) * kChannels * dtype_size(ctx.dtype));

        for (int i = 0; i < ctx.batch; ++i) {
            boxes_[i].lt = {ctx.width / 4, ctx.height / 4};
            boxes_[i].rb = {ctx.width * 3 / 4, ctx.height * 3 / 4};
            numBoxes_[i] = 1;
        }
    }

    RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                  rppHandle_t handle) override {
        return rppt_erase(src.data, src.descPtr, dst.data, dst.descPtr, boxes_, colors_, numBoxes_,
                          src.roi, src.roiType, handle, ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(boxes_, isHip_);
        boxes_ = nullptr;
        bench_pinned_free(colors_, isHip_);
        colors_ = nullptr;
        bench_pinned_free(numBoxes_, isHip_);
        numBoxes_ = nullptr;
    }

private:
    RpptRoiLtrb *boxes_ = nullptr;
    void *colors_ = nullptr;
    Rpp32u *numBoxes_ = nullptr;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("erase", EraseAdapter)

} // namespace rppbench
