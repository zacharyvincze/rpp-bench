/**
 * @file bench_resize_mirror_normalize.cpp
 * @brief Adapter for rppt_resize_mirror_normalize - fused resize + mirror + normalize.
 *
 * Resizes the source ROI (full image here) to the runner's "dst_sizes",
 * optionally mirrors, then normalizes with per-channel mean/stddev. Builds the
 * RpptImagePatch target-size tensor plus mean/stddev tensors (sized batch *
 * channels) and a mirror flag tensor.
 *
 * Signature:
 *   rppt_resize_mirror_normalize(src, srcDesc, dst, dstDesc, dstImgSizes,
 *                                interpolationType, meanTensor, stdDevTensor,
 *                                mirrorTensor, roi, roiType, handle, backend)
 */
#include "harness/bench_registry.hpp"
#include <rpp/rppt_tensor_geometric_augmentations.h>

namespace rppbench {

class ResizeMirrorNormalizeAdapter : public OpAdapter {
public:
    void setup(const BenchContext &ctx, TensorBuffer &, TensorBuffer &) override {
        isHip_ = ctx.isHip;
        interp_ = parse_interpolation(ctx.param<std::string>("interpolation", "BILINEAR"));
        const auto mean = ctx.param<float>("mean", 0.0F);
        const auto stddev = ctx.param<float>("std_dev", 1.0F);
        const auto mirror = static_cast<Rpp32u>(ctx.param<int>("mirror", 1));
        const int channels = layout_info(ctx.layout).channels;
        const int scalarCount = ctx.batch * channels;

        dstSizes_ = static_cast<RpptImagePatch *>(
            bench_pinned_alloc(sizeof(RpptImagePatch) * ctx.batch, ctx.isHip));
        mean_ = static_cast<float *>(bench_pinned_alloc(sizeof(float) * scalarCount, ctx.isHip));
        stddev_ = static_cast<float *>(bench_pinned_alloc(sizeof(float) * scalarCount, ctx.isHip));
        mirror_ = static_cast<Rpp32u *>(bench_pinned_alloc(sizeof(Rpp32u) * ctx.batch, ctx.isHip));

        for (int i = 0; i < ctx.batch; ++i) {
            dstSizes_[i].width = static_cast<Rpp32u>(ctx.dstWidth);
            dstSizes_[i].height = static_cast<Rpp32u>(ctx.dstHeight);
            mirror_[i] = mirror;
        }
        for (int i = 0; i < scalarCount; ++i) {
            mean_[i] = mean;
            stddev_[i] = stddev;
        }
    }

    RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                  rppHandle_t handle) override {
        return rppt_resize_mirror_normalize(src.data, src.descPtr, dst.data, dst.descPtr, dstSizes_,
                                            interp_, mean_, stddev_, mirror_, src.roi, src.roiType,
                                            handle, ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(dstSizes_, isHip_);
        dstSizes_ = nullptr;
        bench_pinned_free(mean_, isHip_);
        mean_ = nullptr;
        bench_pinned_free(stddev_, isHip_);
        stddev_ = nullptr;
        bench_pinned_free(mirror_, isHip_);
        mirror_ = nullptr;
    }

private:
    RpptImagePatch *dstSizes_ = nullptr;
    float *mean_ = nullptr;
    float *stddev_ = nullptr;
    Rpp32u *mirror_ = nullptr;
    RpptInterpolationType interp_ = RpptInterpolationType::BILINEAR;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("resize_mirror_normalize", ResizeMirrorNormalizeAdapter)

} // namespace rppbench
