/**
 * @file bench_fused_multiply_add_scalar.cpp
 * @brief Adapter for rppt_fused_multiply_add_scalar - per-sample (src*mul)+add on a voxel tensor.
 *
 * Voxel op (RpptGenericDesc 5D + RpptROI3D) via VoxelOpAdapter; see
 * bench_add_scalar.cpp. Two per-sample scalar tensors (multiplier, addend). F32.
 *
 * Signature:
 *   rppt_fused_multiply_add_scalar(src, srcGenericDesc, dst, dstGenericDesc,
 *                                  mulTensor, addTensor, roiGenericSrc, roiType,
 *                                  handle, backend)
 */
#include "harness/bench_generic_adapters.hpp"
#include <rpp/rppt_tensor_arithmetic_operations.h>

#include <vector>

namespace rppbench {

class FusedMultiplyAddScalarAdapter : public VoxelOpAdapter {
public:
    std::vector<RpptDataType> supportedDtypes() const override { return {RpptDataType::F32}; }

    void setup(const BenchContext &ctx, std::vector<TensorBuffer> &,
               std::vector<TensorBuffer> &) override {
        isHip_ = ctx.isHip;
        const auto mul = ctx.param<float>("multiply", 80.0F);
        const auto add = ctx.param<float>("add", 5.0F);
        mul_ = static_cast<float *>(bench_pinned_alloc(sizeof(float) * ctx.batch, ctx.isHip));
        add_ = static_cast<float *>(bench_pinned_alloc(sizeof(float) * ctx.batch, ctx.isHip));
        for (int i = 0; i < ctx.batch; ++i) {
            mul_[i] = mul;
            add_[i] = add;
        }
    }

    RppStatus run(const BenchContext &ctx, std::vector<TensorBuffer> &src,
                  std::vector<TensorBuffer> &dst, rppHandle_t handle) override {
        return rppt_fused_multiply_add_scalar(src[0].data, src[0].gdescPtr, dst[0].data,
                                              dst[0].gdescPtr, mul_, add_, src[0].roi3d,
                                              src[0].roi3dType, handle, ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(mul_, isHip_);
        mul_ = nullptr;
        bench_pinned_free(add_, isHip_);
        add_ = nullptr;
    }

private:
    float *mul_ = nullptr;
    float *add_ = nullptr;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("fused_multiply_add_scalar", FusedMultiplyAddScalarAdapter)

} // namespace rppbench
