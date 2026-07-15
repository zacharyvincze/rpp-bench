/**
 * @file bench_gridmask.cpp
 * @brief Adapter for rppt_gridmask - masks a regular grid of squares to zero.
 *
 * All knobs are scalars passed by value: tile width (u32), grid ratio and angle
 * (f32), and a translate vector (RpptUintVector2D). No per-image param tensors.
 *
 * Signature:
 *   rppt_gridmask(src, srcDesc, dst, dstDesc, tileWidth, gridRatio, gridAngle,
 *                 translateVector, roi, roiType, handle, backend)
 */
#include "harness/bench_registry.hpp"
#include <rpp/rppt_tensor_effects_augmentations.h>

namespace rppbench {

class GridmaskAdapter : public OpAdapter {
public:
    void setup(const BenchContext &ctx, TensorBuffer &, TensorBuffer &) override {
        tileWidth_ = static_cast<Rpp32u>(ctx.param<int>("tile_width", 40));
        gridRatio_ = ctx.param<float>("grid_ratio", 0.6F);
        gridAngle_ = ctx.param<float>("grid_angle", 0.0F);
        translate_.x = static_cast<Rpp32u>(ctx.param<int>("translate_x", 0));
        translate_.y = static_cast<Rpp32u>(ctx.param<int>("translate_y", 0));
    }

    RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                  rppHandle_t handle) override {
        return rppt_gridmask(src.data, src.descPtr, dst.data, dst.descPtr, tileWidth_, gridRatio_,
                             gridAngle_, translate_, src.roi, src.roiType, handle, ctx.backend);
    }

private:
    Rpp32u tileWidth_ = 0;
    Rpp32f gridRatio_ = 0.0F;
    Rpp32f gridAngle_ = 0.0F;
    RpptUintVector2D translate_{};
};

REGISTER_RPP_BENCH("gridmask", GridmaskAdapter)

} // namespace rppbench
