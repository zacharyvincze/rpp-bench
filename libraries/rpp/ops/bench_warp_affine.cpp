/**
 * @file bench_warp_affine.cpp
 * @brief Adapter for rppt_warp_affine - per-image 2x3 affine warp.
 *
 * Builds a rotation affine from an "angle" param (degrees) - the six matrix
 * values per image are {cos, -sin, 0, sin, cos, 0} - and reuses the shared
 * interpolation enum. Matrix values do not affect kernel cost, so a rotation is
 * just a representative, non-degenerate transform. Output keeps source dims.
 *
 * Signature:
 *   rppt_warp_affine(src, srcDesc, dst, dstDesc, affineTensor, interpolationType,
 *                    roi, roiType, handle, backend)
 */
#include "rpp/bench_simple_adapter.hpp"
#include <rpp/rppt_tensor_geometric_augmentations.h>
#include <cmath>

namespace rppbench {

class WarpAffineAdapter : public SimpleOpAdapter {
public:
    void setup(const BenchContext &ctx, TensorBuffer &, TensorBuffer &) override {
        isHip_ = ctx.isHip;
        interp_ = parse_interpolation(ctx.param<std::string>("interpolation", "BILINEAR"));
        const float radians = ctx.param<float>("angle", 30.0F) * 3.14159265358979F / 180.0F;
        const float c = std::cos(radians), s = std::sin(radians);
        affine_ =
            static_cast<float *>(bench_pinned_alloc(sizeof(float) * 6 * ctx.batch, ctx.isHip));
        for (int i = 0; i < ctx.batch; ++i) {
            float *m = affine_ + 6 * i;
            m[0] = c;
            m[1] = -s;
            m[2] = 0.0F;
            m[3] = s;
            m[4] = c;
            m[5] = 0.0F;
        }
    }

    RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                  rppHandle_t handle) override {
        return rppt_warp_affine(src.data, src.descPtr, dst.data, dst.descPtr, affine_, interp_,
                                src.roi, src.roiType, handle, ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(affine_, isHip_);
        affine_ = nullptr;
    }

private:
    float *affine_ = nullptr;
    RpptInterpolationType interp_ = RpptInterpolationType::BILINEAR;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("warp_affine", WarpAffineAdapter)

} // namespace rppbench
