/**
 * @file bench_multiply_scalar.cpp
 * @brief Adapter for rppt_multiply_scalar - scale a voxel tensor by a per-sample scalar.
 *
 * Voxel op (RpptGenericDesc 5D + RpptROI3D) via VoxelOpAdapter; see
 * bench_add_scalar.cpp for the pattern. F32 only.
 *
 * Signature:
 *   rppt_multiply_scalar(src, srcGenericDesc, dst, dstGenericDesc, mulTensor,
 *                        roiGenericSrc, roiType, handle, backend)
 */
#include "rpp/bench_generic_adapters.hpp"
#include <rpp/rppt_tensor_arithmetic_operations.h>

#include <vector>

namespace rppbench {

class MultiplyScalarAdapter : public VoxelOpAdapter {
public:
    std::vector<RpptDataType> supportedDtypes() const override { return {RpptDataType::F32}; }

    void setup(const BenchContext &ctx, std::vector<TensorBuffer> &,
               std::vector<TensorBuffer> &) override {
        isHip_ = ctx.isHip;
        const auto value = ctx.param<float>("multiply", 80.0F);
        tensor_ = static_cast<float *>(bench_pinned_alloc(sizeof(float) * ctx.batch, ctx.isHip));
        for (int i = 0; i < ctx.batch; ++i)
            tensor_[i] = value;
    }

    RppStatus run(const BenchContext &ctx, std::vector<TensorBuffer> &src,
                  std::vector<TensorBuffer> &dst, rppHandle_t handle) override {
        return rppt_multiply_scalar(src[0].data, src[0].gdescPtr, dst[0].data, dst[0].gdescPtr,
                                    tensor_, src[0].roi3d, src[0].roi3dType, handle, ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(tensor_, isHip_);
        tensor_ = nullptr;
    }

private:
    float *tensor_ = nullptr;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("multiply_scalar", MultiplyScalarAdapter)

} // namespace rppbench
