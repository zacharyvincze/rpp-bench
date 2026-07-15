/**
 * @file bench_crop_mirror_normalize.cpp
 * @brief Adapter for rppt_crop_mirror_normalize - crop, optional mirror, normalize.
 *
 * Output = (crop(src) [mirrored]) * multiplier + offset, per-image scalar
 * offset/multiplier plus a mirror flag. The crop window is a private ROI tensor
 * built once (sized to the runner's "dst_sizes", anchored top-left); like crop,
 * the XYWH ROI is read-only here, so there's no per-call ROI work. Works on 1-
 * or 3-channel layouts and all dtypes.
 *
 * Signature:
 *   rppt_crop_mirror_normalize(src, srcDesc, dst, dstDesc, offsetTensor,
 *                              multiplierTensor, mirrorTensor, roi, roiType,
 *                              handle, backend)
 */
#include "harness/bench_registry.hpp"
#include <rpp/rppt_tensor_geometric_augmentations.h>

namespace rppbench {

class CropMirrorNormalizeAdapter : public OpAdapter {
public:
    void setup(const BenchContext &ctx, TensorBuffer &, TensorBuffer &) override {
        isHip_ = ctx.isHip;
        const auto offset = ctx.param<float>("offset", 0.0F);         // <= 0
        const auto multiplier = ctx.param<float>("multiplier", 1.0F); // > 0
        const auto mirror = static_cast<Rpp32u>(ctx.param<int>("mirror", 1));
        offset_ = static_cast<float *>(bench_pinned_alloc(sizeof(float) * ctx.batch, ctx.isHip));
        multiplier_ =
            static_cast<float *>(bench_pinned_alloc(sizeof(float) * ctx.batch, ctx.isHip));
        mirror_ = static_cast<Rpp32u *>(bench_pinned_alloc(sizeof(Rpp32u) * ctx.batch, ctx.isHip));
        roi_ = static_cast<RpptROI *>(bench_pinned_alloc(sizeof(RpptROI) * ctx.batch, ctx.isHip));
        for (int i = 0; i < ctx.batch; ++i) {
            offset_[i] = offset;
            multiplier_[i] = multiplier;
            mirror_[i] = mirror;
            roi_[i].xywhROI.xy.x = 0;
            roi_[i].xywhROI.xy.y = 0;
            roi_[i].xywhROI.roiWidth = ctx.dstWidth;
            roi_[i].xywhROI.roiHeight = ctx.dstHeight;
        }
    }

    RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                  rppHandle_t handle) override {
        return rppt_crop_mirror_normalize(src.data, src.descPtr, dst.data, dst.descPtr, offset_,
                                          multiplier_, mirror_, roi_, RpptRoiType::XYWH, handle,
                                          ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(offset_, isHip_);
        offset_ = nullptr;
        bench_pinned_free(multiplier_, isHip_);
        multiplier_ = nullptr;
        bench_pinned_free(mirror_, isHip_);
        mirror_ = nullptr;
        bench_pinned_free(roi_, isHip_);
        roi_ = nullptr;
    }

private:
    float *offset_ = nullptr;
    float *multiplier_ = nullptr;
    Rpp32u *mirror_ = nullptr;
    RpptROI *roi_ = nullptr;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("crop_mirror_normalize", CropMirrorNormalizeAdapter)

} // namespace rppbench
