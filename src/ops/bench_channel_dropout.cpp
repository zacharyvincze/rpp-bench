/**
 * @file bench_channel_dropout.cpp
 * @brief Adapter for rppt_channel_dropout - a per-channel keep/drop mask per image.
 *
 * dropoutTensor is an Rpp8u tensor of size batchSize * channels, each value
 * 0 (drop) or 1 (keep). We keep all channels (all 1s) so the benchmark measures
 * the op's steady-state cost rather than a degenerate all-dropped path.
 *
 * Signature:
 *   rppt_channel_dropout(src, srcDesc, dst, dstDesc, dropoutTensor,
 *                        roi, roiType, handle, backend)
 */
#include "harness/bench_registry.hpp"
#include <rpp/rppt_tensor_effects_augmentations.h>

namespace rppbench {

class ChannelDropoutAdapter : public OpAdapter {
public:
    void setup(const BenchContext &ctx, TensorBuffer &, TensorBuffer &) override {
        isHip_ = ctx.isHip;
        const int channels = layout_info(ctx.layout).channels;
        const size_t count = static_cast<size_t>(ctx.batch) * channels;
        dropout_ = static_cast<Rpp8u *>(bench_pinned_alloc(sizeof(Rpp8u) * count, ctx.isHip));
        for (size_t i = 0; i < count; ++i)
            dropout_[i] = 1; // keep every channel
    }

    RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                  rppHandle_t handle) override {
        return rppt_channel_dropout(src.data, src.descPtr, dst.data, dst.descPtr, dropout_, src.roi,
                                    src.roiType, handle, ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(dropout_, isHip_);
        dropout_ = nullptr;
    }

private:
    Rpp8u *dropout_ = nullptr;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("channel_dropout", ChannelDropoutAdapter)

} // namespace rppbench
