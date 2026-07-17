/**
 * @file bench_gaussian_noise.cpp
 * @brief Adapter for rppt_gaussian_noise - additive Gaussian noise per image.
 *
 * Two f32 param tensors (mean, stddev) plus a fixed RNG seed for a deterministic
 * sweep.
 *
 * Signature:
 *   rppt_gaussian_noise(src, srcDesc, dst, dstDesc, meanTensor, stdDevTensor,
 *                       seed, roi, roiType, handle, backend)
 */
#include "rpp/bench_simple_adapter.hpp"
#include <rpp/rppt_tensor_effects_augmentations.h>

namespace rppbench {

class GaussianNoiseAdapter : public SimpleOpAdapter {
public:
    void setup(const BenchContext &ctx, TensorBuffer &, TensorBuffer &) override {
        isHip_ = ctx.isHip;
        seed_ = static_cast<Rpp32u>(ctx.param<int>("seed", 0));
        const auto mean = ctx.param<float>("mean", 0.0F);
        const auto stddev = ctx.param<float>("std_dev", 0.2F);
        mean_ = static_cast<float *>(bench_pinned_alloc(sizeof(float) * ctx.batch, ctx.isHip));
        stddev_ = static_cast<float *>(bench_pinned_alloc(sizeof(float) * ctx.batch, ctx.isHip));
        for (int i = 0; i < ctx.batch; ++i) {
            mean_[i] = mean;
            stddev_[i] = stddev;
        }
    }

    RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                  rppHandle_t handle) override {
        return rppt_gaussian_noise(src.data, src.descPtr, dst.data, dst.descPtr, mean_, stddev_,
                                   seed_, src.roi, src.roiType, handle, ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(mean_, isHip_);
        mean_ = nullptr;
        bench_pinned_free(stddev_, isHip_);
        stddev_ = nullptr;
    }

private:
    float *mean_ = nullptr;
    float *stddev_ = nullptr;
    Rpp32u seed_ = 0;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("gaussian_noise", GaussianNoiseAdapter)

} // namespace rppbench
