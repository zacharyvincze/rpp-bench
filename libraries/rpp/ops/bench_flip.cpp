/**
 * @file bench_flip.cpp
 * @brief Adapter for rppt_flip - per-image horizontal/vertical flip flags.
 *
 * Signature:
 *   rppt_flip(src, srcDesc, dst, dstDesc, horizontalTensor, verticalTensor,
 *             roi, roiType, handle, backend)
 */
#include "rpp/bench_simple_adapter.hpp"
#include <rpp/rppt_tensor_geometric_augmentations.h>

namespace rppbench {

class FlipAdapter : public SimpleOpAdapter {
public:
    void setup(const BenchContext &ctx, TensorBuffer &, TensorBuffer &) override {
        isHip_ = ctx.isHip;
        const auto horizontal = static_cast<Rpp32u>(ctx.param<int>("horizontal", 1));
        const auto vertical = static_cast<Rpp32u>(ctx.param<int>("vertical", 0));
        horizontal_ =
            static_cast<Rpp32u *>(bench_pinned_alloc(sizeof(Rpp32u) * ctx.batch, ctx.isHip));
        vertical_ =
            static_cast<Rpp32u *>(bench_pinned_alloc(sizeof(Rpp32u) * ctx.batch, ctx.isHip));
        for (int i = 0; i < ctx.batch; ++i) {
            horizontal_[i] = horizontal;
            vertical_[i] = vertical;
        }
    }

    RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                  rppHandle_t handle) override {
        return rppt_flip(src.data, src.descPtr, dst.data, dst.descPtr, horizontal_, vertical_,
                         src.roi, src.roiType, handle, ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(horizontal_, isHip_);
        horizontal_ = nullptr;
        bench_pinned_free(vertical_, isHip_);
        vertical_ = nullptr;
    }

private:
    Rpp32u *horizontal_ = nullptr;
    Rpp32u *vertical_ = nullptr;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("flip", FlipAdapter)

} // namespace rppbench
