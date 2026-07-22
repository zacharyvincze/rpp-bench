/**
 * @file bench_coarse_dropout.cpp
 * @brief Adapter for rppt_coarse_dropout - erase several small rectangular
 *        regions per image (regions filled internally, no colours tensor).
 *
 * Quirks vs rppt_erase:
 *   - No colours tensor: erased regions are filled internally by the kernel.
 *   - Takes a scalar `maxBoxesPerImage` that is *also* the per-image stride of
 *     the anchor-box tensor: `anchorBoxInfoTensor` is a flat RpptRoiLtrb array
 *     of `batchSize * maxBoxesPerImage`, and image i's boxes live at offset
 *     `i * maxBoxesPerImage`. `numBoxesTensor[i]` is that image's active count.
 *   - Anchor boxes on a given image must not overlap, so this adapter lays them
 *     out as one small box per column (own column => guaranteed non-overlap).
 *
 * All harness dtypes (U8/F16/F32/I8) and layouts (NHWC/NCHW, c=1/3) and both
 * backends are supported, so no supported* overrides are needed.
 *
 * Signature:
 *   rppt_coarse_dropout(src, srcDesc, dst, dstDesc, anchorBoxInfoTensor,
 *                       numBoxesTensor, maxBoxesPerImage, roi, roiType,
 *                       handle, backend)
 */
#include "harness/bench_simple_adapter.hpp"
#include <rpp/rppt_tensor_effects_augmentations.h>

#include <algorithm>

namespace rppbench {

class CoarseDropoutAdapter : public SimpleOpAdapter {
public:
    void setup(const BenchContext &ctx, TensorBuffer &, TensorBuffer &) override {
        isHip_ = ctx.isHip;
        maxBoxes_ = std::max(1, ctx.param<int>("max_boxes", 2));

        boxes_ = static_cast<RpptRoiLtrb *>(bench_pinned_alloc(
            sizeof(RpptRoiLtrb) * static_cast<size_t>(ctx.batch) * maxBoxes_, ctx.isHip));
        numBoxes_ =
            static_cast<Rpp32u *>(bench_pinned_alloc(sizeof(Rpp32u) * ctx.batch, ctx.isHip));

        // Lay out one small box per column so boxes never overlap within an image.
        const int cellW = std::max(1, ctx.width / maxBoxes_);
        const int boxW = std::max(1, cellW / 2);
        const int boxH = std::max(1, ctx.height / 4);
        const int boxY = ctx.height / 4;
        for (int i = 0; i < ctx.batch; ++i) {
            for (int b = 0; b < maxBoxes_; ++b) {
                const int boxX = b * cellW;
                RpptRoiLtrb &box = boxes_[i * maxBoxes_ + b];
                box.lt = {boxX, boxY};
                box.rb = {std::min(boxX + boxW - 1, ctx.width - 1),
                          std::min(boxY + boxH - 1, ctx.height - 1)};
            }
            numBoxes_[i] = static_cast<Rpp32u>(maxBoxes_);
        }
    }

    RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                  rppHandle_t handle) override {
        return rppt_coarse_dropout(src.data, src.descPtr, dst.data, dst.descPtr, boxes_, numBoxes_,
                                   static_cast<Rpp32u>(maxBoxes_), src.roi, src.roiType, handle,
                                   ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(boxes_, isHip_);
        boxes_ = nullptr;
        bench_pinned_free(numBoxes_, isHip_);
        numBoxes_ = nullptr;
    }

private:
    RpptRoiLtrb *boxes_ = nullptr;
    Rpp32u *numBoxes_ = nullptr;
    int maxBoxes_ = 2;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("coarse_dropout", CoarseDropoutAdapter)

} // namespace rppbench
