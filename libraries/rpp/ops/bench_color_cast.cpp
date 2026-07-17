/**
 * @file bench_color_cast.cpp
 * @brief Adapter for rppt_color_cast - blend each image toward an RGB colour.
 *
 * Two per-image param tensors: an RpptRGB cast colour and an f32 alpha (blend
 * weight, 0..20 per the API). RGB augmentation, so 3-channel layouts only.
 *
 * Signature:
 *   rppt_color_cast(src, srcDesc, dst, dstDesc, rgbTensor, alphaTensor,
 *                   roi, roiType, handle, backend)
 */
#include "rpp/bench_simple_adapter.hpp"
#include <rpp/rppt_tensor_color_augmentations.h>

namespace rppbench {

class ColorCastAdapter : public SimpleOpAdapter {
public:
    std::vector<Layout> supportedLayouts() const override { return {Layout::PKD3, Layout::PLN3}; }

    void setup(const BenchContext &ctx, TensorBuffer &, TensorBuffer &) override {
        isHip_ = ctx.isHip;
        const auto r = static_cast<Rpp8u>(ctx.param<int>("r", 128));
        const auto g = static_cast<Rpp8u>(ctx.param<int>("g", 64));
        const auto b = static_cast<Rpp8u>(ctx.param<int>("b", 32));
        const auto alpha = ctx.param<float>("alpha", 0.5F);
        rgb_ = static_cast<RpptRGB *>(bench_pinned_alloc(sizeof(RpptRGB) * ctx.batch, ctx.isHip));
        alpha_ = static_cast<float *>(bench_pinned_alloc(sizeof(float) * ctx.batch, ctx.isHip));
        for (int i = 0; i < ctx.batch; ++i) {
            rgb_[i].R = r;
            rgb_[i].G = g;
            rgb_[i].B = b;
            alpha_[i] = alpha;
        }
    }

    RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                  rppHandle_t handle) override {
        return rppt_color_cast(src.data, src.descPtr, dst.data, dst.descPtr, rgb_, alpha_, src.roi,
                               src.roiType, handle, ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(rgb_, isHip_);
        rgb_ = nullptr;
        bench_pinned_free(alpha_, isHip_);
        alpha_ = nullptr;
    }

private:
    RpptRGB *rgb_ = nullptr;
    float *alpha_ = nullptr;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("color_cast", ColorCastAdapter)

} // namespace rppbench
