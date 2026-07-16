/**
 * @file bench_crop.cpp
 * @brief Adapter for rppt_crop - copy a sub-region of the source into the dst.
 *
 * No param tensors: the crop window is the source ROI, and the destination is
 * sized to it. The runner sizes dst from the op's "dst_sizes" config; here we
 * build a private ROI tensor once (that WxH, anchored top-left) and pass it in
 * place of the harness's src.roi. Crop only converts the ROI in place for the
 * LTRB roiType; with XYWH it reads the ROI read-only, so a set-once tensor is
 * safe and avoids per-call work in the timed loop. (We can't reuse src.roi:
 * run_one resets it to the full image before every call.)
 *
 * Signature:
 *   rppt_crop(src, srcDesc, dst, dstDesc, roi, roiType, handle, backend)
 */
#include "harness/bench_registry.hpp"
#include <rpp/rppt_tensor_geometric_augmentations.h>

namespace rppbench {

class CropAdapter : public SimpleOpAdapter {
public:
    void setup(const BenchContext &ctx, TensorBuffer &, TensorBuffer &) override {
        isHip_ = ctx.isHip;
        roi_ = static_cast<RpptROI *>(bench_pinned_alloc(sizeof(RpptROI) * ctx.batch, ctx.isHip));
        for (int i = 0; i < ctx.batch; ++i) {
            roi_[i].xywhROI.xy.x = 0;
            roi_[i].xywhROI.xy.y = 0;
            roi_[i].xywhROI.roiWidth = ctx.dstWidth;
            roi_[i].xywhROI.roiHeight = ctx.dstHeight;
        }
    }

    RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                  rppHandle_t handle) override {
        return rppt_crop(src.data, src.descPtr, dst.data, dst.descPtr, roi_, RpptRoiType::XYWH,
                         handle, ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(roi_, isHip_);
        roi_ = nullptr;
    }

private:
    RpptROI *roi_ = nullptr;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("crop", CropAdapter)

} // namespace rppbench
