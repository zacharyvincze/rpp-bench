/**
 * @file bench_jitter.cpp
 * @brief Adapter for rppt_jitter - random per-pixel spatial displacement.
 *
 * One u32 kernel-size tensor (3/5/7 for optimal use) sets the jitter window;
 * a fixed RNG seed keeps the sweep deterministic. flip-style uint tensor.
 *
 * Signature:
 *   rppt_jitter(src, srcDesc, dst, dstDesc, kernelSizeTensor, seed,
 *               roi, roiType, handle, backend)
 */
#include "harness/bench_registry.hpp"
#include <rpp/rppt_tensor_effects_augmentations.h>

namespace rppbench {

class JitterAdapter : public SimpleOpAdapter {
public:
    void setup(const BenchContext &ctx, TensorBuffer &, TensorBuffer &) override {
        isHip_ = ctx.isHip;
        seed_ = static_cast<Rpp32u>(ctx.param<int>("seed", 0));
        const auto kernelSize = static_cast<Rpp32u>(ctx.param<int>("kernel_size", 3));
        kernelSize_ =
            static_cast<Rpp32u *>(bench_pinned_alloc(sizeof(Rpp32u) * ctx.batch, ctx.isHip));
        for (int i = 0; i < ctx.batch; ++i)
            kernelSize_[i] = kernelSize;
    }

    RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                  rppHandle_t handle) override {
        return rppt_jitter(src.data, src.descPtr, dst.data, dst.descPtr, kernelSize_, seed_,
                           src.roi, src.roiType, handle, ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(kernelSize_, isHip_);
        kernelSize_ = nullptr;
    }

private:
    Rpp32u *kernelSize_ = nullptr;
    Rpp32u seed_ = 0;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("jitter", JitterAdapter)

} // namespace rppbench
