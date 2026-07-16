/**
 * @file bench_color_temperature.cpp
 * @brief Adapter for rppt_color_temperature - one integer (adjustment) param per image.
 *
 * The adjustment tensor is Rpp32s (signed), range -100..100.
 *
 * Signature:
 *   rppt_color_temperature(src, srcDesc, dst, dstDesc, adjustmentValueTensor,
 *                          roi, roiType, handle, backend)
 */
#include "harness/bench_registry.hpp"
#include <rpp/rppt_tensor_color_augmentations.h>

namespace rppbench {

class ColorTemperatureAdapter : public SimpleOpAdapter {
public:
    void setup(const BenchContext &ctx, TensorBuffer &, TensorBuffer &) override {
        isHip_ = ctx.isHip;
        const auto adjustment = static_cast<Rpp32s>(ctx.param<int>("adjustment", 40)); // -100..100
        adjustment_ =
            static_cast<Rpp32s *>(bench_pinned_alloc(sizeof(Rpp32s) * ctx.batch, ctx.isHip));
        for (int i = 0; i < ctx.batch; ++i)
            adjustment_[i] = adjustment;
    }

    RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                  rppHandle_t handle) override {
        return rppt_color_temperature(src.data, src.descPtr, dst.data, dst.descPtr, adjustment_,
                                      src.roi, src.roiType, handle, ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(adjustment_, isHip_);
        adjustment_ = nullptr;
    }

private:
    Rpp32s *adjustment_ = nullptr;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("color_temperature", ColorTemperatureAdapter)

} // namespace rppbench
