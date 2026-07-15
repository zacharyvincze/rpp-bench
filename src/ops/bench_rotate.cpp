/**
 * @file bench_rotate.cpp
 * @brief Adapter for rppt_rotate - per-image rotation about the centre.
 *
 * One f32 angle tensor (degrees, +anticlockwise) plus an interpolation type,
 * reusing the shared interpolation enum from resize. Output keeps the source
 * dimensions, so no dst_sizes are needed.
 *
 * Signature:
 *   rppt_rotate(src, srcDesc, dst, dstDesc, angleTensor, interpolationType,
 *               roi, roiType, handle, backend)
 */
#include "harness/bench_registry.hpp"
#include <rpp/rppt_tensor_geometric_augmentations.h>

namespace rppbench {

class RotateAdapter : public OpAdapter {
public:
    void setup(const BenchContext &ctx, TensorBuffer &, TensorBuffer &) override {
        isHip_ = ctx.isHip;
        interp_ = parse_interpolation(ctx.param<std::string>("interpolation", "BILINEAR"));
        const auto angle = ctx.param<float>("angle", 30.0F);
        angle_ = static_cast<float *>(bench_pinned_alloc(sizeof(float) * ctx.batch, ctx.isHip));
        for (int i = 0; i < ctx.batch; ++i)
            angle_[i] = angle;
    }

    RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                  rppHandle_t handle) override {
        return rppt_rotate(src.data, src.descPtr, dst.data, dst.descPtr, angle_, interp_, src.roi,
                           src.roiType, handle, ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(angle_, isHip_);
        angle_ = nullptr;
    }

private:
    float *angle_ = nullptr;
    RpptInterpolationType interp_ = RpptInterpolationType::BILINEAR;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("rotate", RotateAdapter)

} // namespace rppbench
