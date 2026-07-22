/**
 * @file bench_crop_and_patch.cpp
 * @brief Adapter for rppt_crop_and_patch - crop a region from src1, patch it into src2.
 *
 * A two-source op: both sources share one srcDesc/ROI (numSrc() == 2, inherits
 * OpAdapter directly). Beyond the shared source ROI it needs three extra ROI
 * tensors - the destination ROI, the crop window (in src1) and the patch window
 * (in dst) - which it allocates itself, like other adapters allocate param
 * tensors. Supports all dtypes/layouts (c = 1/3).
 *
 * The crop/patch geometry is fixed to a centred half-size window (the op has no
 * scalar knob that affects throughput; the ROI placement is what varies). ROIs
 * are refreshed each call because some RPP kernels convert XYWH->LTRB in place.
 *
 * Signature:
 *   rppt_crop_and_patch(src1, src2, srcDesc, dst, dstDesc, roiDst, cropRoi,
 *                       patchRoi, roiType, handle, backend)
 */
#include "harness/bench_registry.hpp"
#include <rpp/rppt_tensor_geometric_augmentations.h>

namespace rppbench {

class CropAndPatchAdapter : public OpAdapter {
public:
    int numSrc() const override { return 2; }

    void setup(const BenchContext &ctx, std::vector<TensorBuffer> &,
               std::vector<TensorBuffer> &) override {
        isHip_ = ctx.isHip;
        width_ = ctx.width;
        height_ = ctx.height;
        batch_ = ctx.batch;
        dstRoi_ = static_cast<RpptROI *>(bench_pinned_alloc(sizeof(RpptROI) * batch_, ctx.isHip));
        cropRoi_ = static_cast<RpptROI *>(bench_pinned_alloc(sizeof(RpptROI) * batch_, ctx.isHip));
        patchRoi_ = static_cast<RpptROI *>(bench_pinned_alloc(sizeof(RpptROI) * batch_, ctx.isHip));
        resetRois();
    }

    RppStatus run(const BenchContext &ctx, std::vector<TensorBuffer> &src,
                  std::vector<TensorBuffer> &dst, rppHandle_t handle) override {
        resetRois();
        return rppt_crop_and_patch(src[0].data, src[1].data, src[0].descPtr, dst[0].data,
                                   dst[0].descPtr, dstRoi_, cropRoi_, patchRoi_, src[0].roiType,
                                   handle, ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(dstRoi_, isHip_);
        dstRoi_ = nullptr;
        bench_pinned_free(cropRoi_, isHip_);
        cropRoi_ = nullptr;
        bench_pinned_free(patchRoi_, isHip_);
        patchRoi_ = nullptr;
    }

private:
    // Full-image destination; crop the top-left quarter of src1 and patch it into
    // the centre of src2 (a half-size window that stays fully in bounds).
    void resetRois() const {
        const int cw = width_ / 2;
        const int ch = height_ / 2;
        for (int i = 0; i < batch_; ++i) {
            dstRoi_[i].xywhROI = {{0, 0}, width_, height_};
            cropRoi_[i].xywhROI = {{0, 0}, cw, ch};
            patchRoi_[i].xywhROI = {{width_ / 4, height_ / 4}, cw, ch};
        }
    }

    RpptROI *dstRoi_ = nullptr;
    RpptROI *cropRoi_ = nullptr;
    RpptROI *patchRoi_ = nullptr;
    int width_ = 0, height_ = 0, batch_ = 0;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("crop_and_patch", CropAndPatchAdapter)

} // namespace rppbench
