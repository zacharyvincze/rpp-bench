/**
 * @file bench_resize_crop_mirror.cpp
 * @brief Adapter for rppt_resize_crop_mirror - fused resize + crop + mirror.
 *
 * Resizes the source ROI (full image here) to the runner's "dst_sizes" and
 * optionally mirrors it, reusing the shared interpolation enum. Builds the
 * per-image RpptImagePatch target-size tensor (as resize does) plus a mirror
 * flag tensor.
 *
 * Signature:
 *   rppt_resize_crop_mirror(src, srcDesc, dst, dstDesc, dstImgSizes,
 *                           interpolationType, mirrorTensor, roi, roiType,
 *                           handle, backend)
 */
#include "harness/bench_registry.hpp"
#include <rpp/rppt_tensor_geometric_augmentations.h>

namespace rppbench {

class ResizeCropMirrorAdapter : public SimpleOpAdapter {
public:
    void setup(const BenchContext &ctx, TensorBuffer &, TensorBuffer &) override {
        isHip_ = ctx.isHip;
        interp_ = parse_interpolation(ctx.param<std::string>("interpolation", "BILINEAR"));
        const auto mirror = static_cast<Rpp32u>(ctx.param<int>("mirror", 1));
        dstSizes_ = static_cast<RpptImagePatch *>(
            bench_pinned_alloc(sizeof(RpptImagePatch) * ctx.batch, ctx.isHip));
        mirror_ = static_cast<Rpp32u *>(bench_pinned_alloc(sizeof(Rpp32u) * ctx.batch, ctx.isHip));
        for (int i = 0; i < ctx.batch; ++i) {
            dstSizes_[i].width = static_cast<Rpp32u>(ctx.dstWidth);
            dstSizes_[i].height = static_cast<Rpp32u>(ctx.dstHeight);
            mirror_[i] = mirror;
        }
    }

    RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                  rppHandle_t handle) override {
        return rppt_resize_crop_mirror(src.data, src.descPtr, dst.data, dst.descPtr, dstSizes_,
                                       interp_, mirror_, src.roi, src.roiType, handle, ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(dstSizes_, isHip_);
        dstSizes_ = nullptr;
        bench_pinned_free(mirror_, isHip_);
        mirror_ = nullptr;
    }

private:
    RpptImagePatch *dstSizes_ = nullptr;
    Rpp32u *mirror_ = nullptr;
    RpptInterpolationType interp_ = RpptInterpolationType::BILINEAR;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("resize_crop_mirror", ResizeCropMirrorAdapter)

} // namespace rppbench
