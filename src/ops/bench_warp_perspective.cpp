/**
 * @file bench_warp_perspective.cpp
 * @brief Adapter for rppt_warp_perspective - per-image 3x3 perspective warp.
 *
 * Builds a rotation homography from an "angle" param (degrees) - the nine matrix
 * values per image are the rotation in the top-left 2x2 with a {0,0,1} bottom
 * row - and reuses the shared interpolation enum. Matrix values do not affect
 * kernel cost. Output keeps source dims.
 *
 * Signature:
 *   rppt_warp_perspective(src, srcDesc, dst, dstDesc, perspectiveTensor,
 *                         interpolationType, roi, roiType, handle, backend)
 */
#include "harness/bench_simple_adapter.hpp"
#include <rpp/rppt_tensor_geometric_augmentations.h>
#include <cmath>

namespace rppbench {

class WarpPerspectiveAdapter : public SimpleOpAdapter {
public:
    void setup(const BenchContext &ctx, TensorBuffer &, TensorBuffer &) override {
        isHip_ = ctx.isHip;
        interp_ = parse_interpolation(ctx.param<std::string>("interpolation", "BILINEAR"));
        const float radians = ctx.param<float>("angle", 30.0F) * 3.14159265358979F / 180.0F;
        const float c = std::cos(radians), s = std::sin(radians);
        perspective_ =
            static_cast<float *>(bench_pinned_alloc(sizeof(float) * 9 * ctx.batch, ctx.isHip));
        for (int i = 0; i < ctx.batch; ++i) {
            float *m = perspective_ + 9 * i;
            m[0] = c;
            m[1] = -s;
            m[2] = 0.0F;
            m[3] = s;
            m[4] = c;
            m[5] = 0.0F;
            m[6] = 0.0F;
            m[7] = 0.0F;
            m[8] = 1.0F;
        }
    }

    RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                  rppHandle_t handle) override {
        return rppt_warp_perspective(src.data, src.descPtr, dst.data, dst.descPtr, perspective_,
                                     interp_, src.roi, src.roiType, handle, ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(perspective_, isHip_);
        perspective_ = nullptr;
    }

private:
    float *perspective_ = nullptr;
    RpptInterpolationType interp_ = RpptInterpolationType::BILINEAR;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("warp_perspective", WarpPerspectiveAdapter)

} // namespace rppbench
