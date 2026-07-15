/**
 * @file bench_posterize.cpp
 * @brief Adapter for rppt_posterize - one 8-bit (level-bits) param per image.
 *
 * posterizeLevelBits is an Rpp8u tensor, range 1..8 (bits kept per channel).
 *
 * Signature:
 *   rppt_posterize(src, srcDesc, dst, dstDesc, posterizeLevelBits,
 *                  roi, roiType, handle, backend)
 */
#include "harness/bench_registry.hpp"
#include <rpp/rppt_tensor_effects_augmentations.h>

namespace rppbench {

class PosterizeAdapter : public OpAdapter {
public:
    void setup(const BenchContext &ctx, TensorBuffer &, TensorBuffer &) override {
        isHip_ = ctx.isHip;
        const auto bits = static_cast<Rpp8u>(ctx.param<int>("level_bits", 4)); // 1..8
        bits_ = static_cast<Rpp8u *>(bench_pinned_alloc(sizeof(Rpp8u) * ctx.batch, ctx.isHip));
        for (int i = 0; i < ctx.batch; ++i)
            bits_[i] = bits;
    }

    RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                  rppHandle_t handle) override {
        return rppt_posterize(src.data, src.descPtr, dst.data, dst.descPtr, bits_, src.roi,
                              src.roiType, handle, ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(bits_, isHip_);
        bits_ = nullptr;
    }

private:
    Rpp8u *bits_ = nullptr;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("posterize", PosterizeAdapter)

} // namespace rppbench
