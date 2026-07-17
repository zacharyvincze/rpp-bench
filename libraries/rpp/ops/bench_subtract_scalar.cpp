/**
 * @file bench_subtract_scalar.cpp
 * @brief Adapter for rppt_subtract_scalar - subtract a per-sample scalar from a voxel tensor.
 *
 * Voxel op (RpptGenericDesc 5D + RpptROI3D) via VoxelOpAdapter; see
 * bench_add_scalar.cpp for the pattern. Only op-specific tensor is the per-sample
 * subtrahend. F32 only.
 *
 * Signature:
 *   rppt_subtract_scalar(src, srcGenericDesc, dst, dstGenericDesc, subtractTensor,
 *                        roiGenericSrc, roiType, handle, backend)
 */
#include "rpp/bench_generic_adapters.hpp"
#include <rpp/rppt_tensor_arithmetic_operations.h>

#include <vector>

namespace rppbench {

class SubtractScalarAdapter : public VoxelOpAdapter {
public:
    std::vector<RpptDataType> supportedDtypes() const override { return {RpptDataType::F32}; }

    void setup(const BenchContext &ctx, std::vector<TensorBuffer> &,
               std::vector<TensorBuffer> &) override {
        isHip_ = ctx.isHip;
        const auto value = ctx.param<float>("subtract", 40.0F);
        tensor_ = static_cast<float *>(bench_pinned_alloc(sizeof(float) * ctx.batch, ctx.isHip));
        for (int i = 0; i < ctx.batch; ++i)
            tensor_[i] = value;
    }

    RppStatus run(const BenchContext &ctx, std::vector<TensorBuffer> &src,
                  std::vector<TensorBuffer> &dst, rppHandle_t handle) override {
        return rppt_subtract_scalar(src[0].data, src[0].gdescPtr, dst[0].data, dst[0].gdescPtr,
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

REGISTER_RPP_BENCH("subtract_scalar", SubtractScalarAdapter)

} // namespace rppbench
