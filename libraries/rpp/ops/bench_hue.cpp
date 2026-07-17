/**
 * @file bench_hue.cpp
 * @brief Adapter for rppt_hue - one scalar (hue degrees) param per image.
 *
 * RGB-only: srcDescPtr/dstDescPtr require c = 3, so the sweep is restricted to
 * the 3-channel layouts (PKD3/PLN3).
 *
 * Signature:
 *   rppt_hue(src, srcDesc, dst, dstDesc, hueTensor, roi, roiType, handle, backend)
 */
#include "rpp/bench_simple_adapter.hpp"
#include <rpp/rppt_tensor_color_augmentations.h>

namespace rppbench {

class HueAdapter : public SimpleOpAdapter {
public:
    std::vector<Layout> supportedLayouts() const override { return {Layout::PKD3, Layout::PLN3}; }

    void setup(const BenchContext &ctx, TensorBuffer &, TensorBuffer &) override {
        isHip_ = ctx.isHip;
        const auto hue = ctx.param<float>("hue", 90.0F); // 0..359
        hue_ = static_cast<float *>(bench_pinned_alloc(sizeof(float) * ctx.batch, ctx.isHip));
        for (int i = 0; i < ctx.batch; ++i)
            hue_[i] = hue;
    }

    RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                  rppHandle_t handle) override {
        return rppt_hue(src.data, src.descPtr, dst.data, dst.descPtr, hue_, src.roi, src.roiType,
                        handle, ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(hue_, isHip_);
        hue_ = nullptr;
    }

private:
    float *hue_ = nullptr;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("hue", HueAdapter)

} // namespace rppbench
