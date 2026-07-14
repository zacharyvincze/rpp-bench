// Adapter for rppt_resize - destination sizes + interpolation type.
//
// Signature:
//   rppt_resize(src, srcDesc, dst, dstDesc, dstImgSizes, interpolationType,
//               roi, roiType, handle, backend)
//
// The runner varies the destination dimensions via the op's "dst_sizes" config
// list; here we build the per-image RpptImagePatch tensor and pick interpolation.
#include "bench_registry.hpp"
#include <rpp/rppt_tensor_geometric_augmentations.h>

namespace rppbench {

class ResizeAdapter : public OpAdapter {
public:
    void setup(const BenchContext &ctx, TensorBuffer &, TensorBuffer &) override {
        isHip_ = ctx.isHip;
        interp_ = parse_interpolation(ctx.param<std::string>("interpolation", "BILINEAR"));
        dstSizes_ = static_cast<RpptImagePatch *>(
            bench_pinned_alloc(sizeof(RpptImagePatch) * ctx.batch, ctx.isHip));
        for (int i = 0; i < ctx.batch; ++i) {
            dstSizes_[i].width = static_cast<Rpp32u>(ctx.dstWidth);
            dstSizes_[i].height = static_cast<Rpp32u>(ctx.dstHeight);
        }
    }

    RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                  rppHandle_t handle) override {
        // The resize kernel derives the scale as srcRoi / dstImgSize, so the ROI
        // must describe the *source* region (full source here) and dstSizes_ the
        // target. This yields a true resize (src.roi=WxH -> dstSizes_).
        return rppt_resize(src.data, src.descPtr, dst.data, dst.descPtr, dstSizes_, interp_,
                           src.roi, src.roiType, handle, ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(dstSizes_, isHip_);
        dstSizes_ = nullptr;
    }

private:
    RpptImagePatch *dstSizes_ = nullptr;
    RpptInterpolationType interp_ = RpptInterpolationType::BILINEAR;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("resize", ResizeAdapter)

} // namespace rppbench
