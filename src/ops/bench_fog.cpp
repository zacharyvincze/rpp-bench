// Adapter for rppt_fog - per-image intensity + grey factor tensors.
//
// Signature:
//   rppt_fog(src, srcDesc, dst, dstDesc, intensityFactor, greyFactor,
//            roi, roiType, handle, backend)
#include "harness/bench_registry.hpp"
#include <rpp/rppt_tensor_effects_augmentations.h>

namespace rppbench {

class FogAdapter : public OpAdapter {
public:
    void setup(const BenchContext &ctx, TensorBuffer &, TensorBuffer &) override {
        isHip_ = ctx.isHip;
        const auto intensity = ctx.param<float>("intensity", 0.4F); // 0..0.5
        const auto grey = ctx.param<float>("grey", 0.5F);           // 0..1
        intensity_ = static_cast<float *>(bench_pinned_alloc(sizeof(float) * ctx.batch, ctx.isHip));
        grey_ = static_cast<float *>(bench_pinned_alloc(sizeof(float) * ctx.batch, ctx.isHip));
        for (int i = 0; i < ctx.batch; ++i) {
            intensity_[i] = intensity;
            grey_[i] = grey;
        }
    }

    RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                  rppHandle_t handle) override {
        return rppt_fog(src.data, src.descPtr, dst.data, dst.descPtr, intensity_, grey_, src.roi,
                        src.roiType, handle, ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(intensity_, isHip_);
        intensity_ = nullptr;
        bench_pinned_free(grey_, isHip_);
        grey_ = nullptr;
    }

private:
    float *intensity_ = nullptr;
    float *grey_ = nullptr;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("fog", FogAdapter)

} // namespace rppbench
