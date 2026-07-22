/**
 * @file bench_grid_dropout.cpp
 * @brief Adapter for rppt_grid_dropout - erase a regular grid of holes per image.
 *
 * grid_dropout blanks out a grid of small rectangular holes across each image
 * (regions filled internally by RPP - no colour tensor, unlike erase). The op
 * takes a flat RpptRoiLtrb anchor-box tensor (batch * boxesInEachImage) plus the
 * scalars boxesInEachImage, maxHoleW and maxHoleH.
 *
 * We build a near-square grid from the `boxes` param (default 4 -> 2x2): gridW =
 * ceil(sqrt(boxes)), gridH = ceil(boxes / gridW), and boxesInEachImage is the
 * actual grid cell count (gridW*gridH). Each cell gets one fixed-size hole
 * (hole_w x hole_h params) anchored at the cell's top-left, clamped to the cell
 * so it stays fully inside the image (mirrors the test suite's fixed-seed layout
 * in init_grid_dropout, which uses cellW=roiW/gridW and a hole per cell). The
 * scalars maxHoleW/maxHoleH are the (uniform) clamped hole dimensions.
 *
 * All dtypes (U8/F16/F32/I8), all layouts, both backends - no restrictions.
 *
 * Signature:
 *   rppt_grid_dropout(src, srcDesc, dst, dstDesc, anchorBoxInfoTensor,
 *                     boxesInEachImage, maxHoleW, maxHoleH, roi, roiType,
 *                     handle, backend)
 */
#include "harness/bench_simple_adapter.hpp"
#include <rpp/rppt_tensor_effects_augmentations.h>

#include <algorithm>
#include <cmath>

namespace rppbench {

class GridDropoutAdapter : public SimpleOpAdapter {
public:
    void setup(const BenchContext &ctx, TensorBuffer &, TensorBuffer &) override {
        isHip_ = ctx.isHip;

        const int boxes = std::max(1, ctx.param<int>("boxes", 4));
        const int holeW = std::max(1, ctx.param<int>("hole_w", 20));
        const int holeH = std::max(1, ctx.param<int>("hole_h", 20));

        // Near-square grid whose cell count covers the requested box count.
        const int gridW =
            std::max(1, static_cast<int>(std::ceil(std::sqrt(static_cast<double>(boxes)))));
        const int gridH = std::max(1, (boxes + gridW - 1) / gridW);
        boxesInEachImage_ = static_cast<Rpp32u>(gridW) * static_cast<Rpp32u>(gridH);

        const int cellW = std::max(1, ctx.width / gridW);
        const int cellH = std::max(1, ctx.height / gridH);

        // Keep each hole inside its cell (and thus inside the image).
        const int actualHoleW = std::max(1, std::min(holeW, cellW));
        const int actualHoleH = std::max(1, std::min(holeH, cellH));
        maxHoleW_ = static_cast<Rpp32u>(actualHoleW);
        maxHoleH_ = static_cast<Rpp32u>(actualHoleH);

        const size_t totalBoxes = static_cast<size_t>(ctx.batch) * boxesInEachImage_;
        boxes_ = static_cast<RpptRoiLtrb *>(
            bench_pinned_alloc(sizeof(RpptRoiLtrb) * totalBoxes, ctx.isHip));

        for (int i = 0; i < ctx.batch; ++i) {
            const size_t boxOffset = static_cast<size_t>(i) * boxesInEachImage_;
            for (int row = 0; row < gridH; ++row) {
                for (int col = 0; col < gridW; ++col) {
                    int x1 = col * cellW;
                    int y1 = row * cellH;
                    int x2 = std::min(x1 + actualHoleW - 1, ctx.width - 1);
                    int y2 = std::min(y1 + actualHoleH - 1, ctx.height - 1);
                    x1 = std::min(x1, ctx.width - 1);
                    y1 = std::min(y1, ctx.height - 1);

                    RpptRoiLtrb &box = boxes_[boxOffset + static_cast<size_t>(row) * gridW + col];
                    box.lt.x = x1;
                    box.lt.y = y1;
                    box.rb.x = x2;
                    box.rb.y = y2;
                }
            }
        }
    }

    RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                  rppHandle_t handle) override {
        return rppt_grid_dropout(src.data, src.descPtr, dst.data, dst.descPtr, boxes_,
                                 boxesInEachImage_, maxHoleW_, maxHoleH_, src.roi, src.roiType,
                                 handle, ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(boxes_, isHip_);
        boxes_ = nullptr;
    }

private:
    RpptRoiLtrb *boxes_ = nullptr;
    Rpp32u boxesInEachImage_ = 0;
    Rpp32u maxHoleW_ = 0;
    Rpp32u maxHoleH_ = 0;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("grid_dropout", GridDropoutAdapter)

} // namespace rppbench
