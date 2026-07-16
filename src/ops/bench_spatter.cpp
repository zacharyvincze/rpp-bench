/**
 * @file bench_spatter.cpp
 * @brief Adapter for rppt_spatter - overlays a spatter (e.g. mud/blood) pattern.
 *
 * The spatter colour is a single RpptRGB passed by value (not a per-image
 * tensor), so there are no param buffers to allocate. RGB colour -> 3-channel
 * layouts only (PKD3/PLN3).
 *
 * Signature:
 *   rppt_spatter(src, srcDesc, dst, dstDesc, spatterColor, roi, roiType, handle, backend)
 */
#include "harness/bench_registry.hpp"
#include <rpp/rppt_tensor_effects_augmentations.h>

namespace rppbench {

class SpatterAdapter : public SimpleOpAdapter {
public:
    std::vector<Layout> supportedLayouts() const override { return {Layout::PKD3, Layout::PLN3}; }

    void setup(const BenchContext &ctx, TensorBuffer &, TensorBuffer &) override {
        color_.R = static_cast<Rpp8u>(ctx.param<int>("r", 65));
        color_.G = static_cast<Rpp8u>(ctx.param<int>("g", 42));
        color_.B = static_cast<Rpp8u>(ctx.param<int>("b", 20));
    }

    RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                  rppHandle_t handle) override {
        return rppt_spatter(src.data, src.descPtr, dst.data, dst.descPtr, color_, src.roi,
                            src.roiType, handle, ctx.backend);
    }

private:
    RpptRGB color_{};
};

REGISTER_RPP_BENCH("spatter", SpatterAdapter)

} // namespace rppbench
