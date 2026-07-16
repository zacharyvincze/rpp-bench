/**
 * @file bench_channel_permute.cpp
 * @brief Adapter for rppt_channel_permute - reorder the colour channels.
 *
 * The permutation tensor holds 3 indices (each 0..2) per image; the default
 * {2,1,0} swaps R and B. 3-channel only, so PLN1 is excluded. No ROI in the
 * signature - it permutes the whole tensor.
 *
 * Signature:
 *   rppt_channel_permute(src, srcDesc, dst, dstDesc, permutationTensor, handle, backend)
 */
#include "harness/bench_registry.hpp"
#include <rpp/rppt_tensor_data_exchange_operations.h>

namespace rppbench {

class ChannelPermuteAdapter : public SimpleOpAdapter {
public:
    std::vector<Layout> supportedLayouts() const override { return {Layout::PKD3, Layout::PLN3}; }

    void setup(const BenchContext &ctx, TensorBuffer &, TensorBuffer &) override {
        isHip_ = ctx.isHip;
        const auto p0 = static_cast<Rpp32u>(ctx.param<int>("perm0", 2));
        const auto p1 = static_cast<Rpp32u>(ctx.param<int>("perm1", 1));
        const auto p2 = static_cast<Rpp32u>(ctx.param<int>("perm2", 0));
        perm_ =
            static_cast<Rpp32u *>(bench_pinned_alloc(sizeof(Rpp32u) * 3 * ctx.batch, ctx.isHip));
        for (int i = 0; i < ctx.batch; ++i) {
            perm_[3 * i + 0] = p0;
            perm_[3 * i + 1] = p1;
            perm_[3 * i + 2] = p2;
        }
    }

    RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                  rppHandle_t handle) override {
        return rppt_channel_permute(src.data, src.descPtr, dst.data, dst.descPtr, perm_, handle,
                                    ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(perm_, isHip_);
        perm_ = nullptr;
    }

private:
    Rpp32u *perm_ = nullptr;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("channel_permute", ChannelPermuteAdapter)

} // namespace rppbench
