// Adapter for rppt_saturation - one scalar (saturation factor) param per image.
//
// RGB-only: srcDescPtr/dstDescPtr require c = 3, so the sweep is restricted to
// the 3-channel layouts (PKD3/PLN3).
//
// Signature:
//   rppt_saturation(src, srcDesc, dst, dstDesc, saturationTensor,
//                   roi, roiType, handle, backend)
#include "harness/bench_registry.hpp"
#include <rpp/rppt_tensor_color_augmentations.h>

namespace rppbench {

class SaturationAdapter : public OpAdapter {
public:
    std::vector<Layout> supportedLayouts() const override { return {Layout::PKD3, Layout::PLN3}; }

    void setup(const BenchContext &ctx, TensorBuffer &, TensorBuffer &) override {
        isHip_ = ctx.isHip;
        const auto saturation = ctx.param<float>("saturation", 1.5F); // >= 0
        saturation_ =
            static_cast<float *>(bench_pinned_alloc(sizeof(float) * ctx.batch, ctx.isHip));
        for (int i = 0; i < ctx.batch; ++i)
            saturation_[i] = saturation;
    }

    RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                  rppHandle_t handle) override {
        return rppt_saturation(src.data, src.descPtr, dst.data, dst.descPtr, saturation_, src.roi,
                               src.roiType, handle, ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(saturation_, isHip_);
        saturation_ = nullptr;
    }

private:
    float *saturation_ = nullptr;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("saturation", SaturationAdapter)

} // namespace rppbench
