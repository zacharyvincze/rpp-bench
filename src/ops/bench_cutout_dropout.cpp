/**
 * @file bench_cutout_dropout.cpp
 * @brief Adapter for rppt_cutout_dropout - fill rectangular boxes with a solid colour.
 *
 * Like rppt_erase, but supports multiple boxes per image and any channel count
 * (c=1/3), so it keeps all layouts (PKD3/PLN3/PLN1) rather than RGB-only.
 *
 * Three param tensors, all keyed by a `maxBoxesPerImage` stride (read from the
 * `max_boxes` op param, default 2):
 *   - anchorBoxInfoTensor: RpptRoiLtrb[batch * maxBoxes], box idx = i*maxBoxes + j.
 *     Filled here as `maxBoxes` non-overlapping rectangles laid across the image
 *     (one per horizontal slot), derived from ctx.width/ctx.height.
 *   - colorsTensor: one colour per box, each `channels` values in the SOURCE
 *     dtype (NOT a fixed 3) - size batch*maxBoxes*channels*dtype_size, laid out
 *     as [box0_c0..box0_c{ch-1}, box1_c0..]; zero-filled = black.
 *   - numBoxesTensor: Rpp32u[batch], boxes per image = maxBoxes.
 *
 * Signature:
 *   rppt_cutout_dropout(src, srcDesc, dst, dstDesc, anchorBoxInfoTensor,
 *                       colorsTensor, numBoxesTensor, roi, roiType, handle, backend)
 */
#include "harness/bench_simple_adapter.hpp"
#include <rpp/rppt_tensor_effects_augmentations.h>

#include <cstring>

namespace rppbench {

class CutoutDropoutAdapter : public SimpleOpAdapter {
public:
    void setup(const BenchContext &ctx, TensorBuffer &, TensorBuffer &) override {
        isHip_ = ctx.isHip;
        maxBoxes_ = ctx.param<int>("max_boxes", 2);
        if (maxBoxes_ < 1)
            maxBoxes_ = 1;
        const int channels = layout_info(ctx.layout).channels;
        const size_t totalBoxes = static_cast<size_t>(ctx.batch) * maxBoxes_;

        boxes_ = static_cast<RpptRoiLtrb *>(
            bench_pinned_alloc(sizeof(RpptRoiLtrb) * totalBoxes, ctx.isHip));
        numBoxes_ =
            static_cast<Rpp32u *>(bench_pinned_alloc(sizeof(Rpp32u) * ctx.batch, ctx.isHip));
        // Colours: `channels` values per box, in the source dtype; black.
        const size_t colorBytes = totalBoxes * channels * dtype_size(ctx.dtype);
        colors_ = bench_pinned_alloc(colorBytes, ctx.isHip);
        std::memset(colors_, 0, colorBytes);

        // Lay maxBoxes non-overlapping rectangles across the image width, each a
        // small box centred vertically in its own horizontal slot.
        const int slot = ctx.width / maxBoxes_;
        const int boxW = (slot > 3) ? slot / 2 : 1;
        const int boxH = (ctx.height > 3) ? ctx.height / 2 : 1;
        const int y0 = ctx.height / 4;
        for (int i = 0; i < ctx.batch; ++i) {
            for (int j = 0; j < maxBoxes_; ++j) {
                const int x0 = j * slot + (slot - boxW) / 2;
                RpptRoiLtrb &box = boxes_[static_cast<size_t>(i) * maxBoxes_ + j];
                box.lt = {x0, y0};
                box.rb = {x0 + boxW - 1, y0 + boxH - 1};
            }
            numBoxes_[i] = static_cast<Rpp32u>(maxBoxes_);
        }
    }

    RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                  rppHandle_t handle) override {
        return rppt_cutout_dropout(src.data, src.descPtr, dst.data, dst.descPtr, boxes_, colors_,
                                   numBoxes_, src.roi, src.roiType, handle, ctx.backend);
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
    int maxBoxes_ = 2;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("cutout_dropout", CutoutDropoutAdapter)

} // namespace rppbench
