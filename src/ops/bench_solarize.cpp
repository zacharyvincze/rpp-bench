// Adapter for rppt_solarize - one scalar (threshold) param per image.
//
// Threshold is normalized (0..1) regardless of dtype, so a single value works
// across the whole dtype sweep.
//
// Signature:
//   rppt_solarize(src, srcDesc, dst, dstDesc, thresholdTensor,
//                 roi, roiType, handle, backend)
#include "harness/bench_registry.hpp"
#include <rpp/rppt_tensor_effects_augmentations.h>

namespace rppbench {

class SolarizeAdapter : public OpAdapter {
public:
    void setup(const BenchContext &ctx, TensorBuffer &, TensorBuffer &) override {
        isHip_ = ctx.isHip;
        const auto threshold = ctx.param<float>("threshold", 0.5F); // 0..1
        threshold_ = static_cast<float *>(bench_pinned_alloc(sizeof(float) * ctx.batch, ctx.isHip));
        for (int i = 0; i < ctx.batch; ++i)
            threshold_[i] = threshold;
    }

    RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                  rppHandle_t handle) override {
        return rppt_solarize(src.data, src.descPtr, dst.data, dst.descPtr, threshold_, src.roi,
                             src.roiType, handle, ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(threshold_, isHip_);
        threshold_ = nullptr;
    }

private:
    float *threshold_ = nullptr;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("solarize", SolarizeAdapter)

} // namespace rppbench
