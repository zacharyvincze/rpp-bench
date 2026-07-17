/**
 * @file bench_add_scalar.cpp
 * @brief Adapter for rppt_add_scalar - add a per-sample scalar to a 3D voxel tensor.
 *
 * First voxel op: it uses RpptGenericDesc (5D NCDHW/NDHWC) plus an RpptROI3D, so
 * it subclasses VoxelOpAdapter, which builds that descriptor and full-volume ROI
 * from the batch/size sweep axes plus a "depth" param (Z has no matrix axis). The
 * only op-specific tensor is the per-sample addend. F32 only.
 *
 * Signature:
 *   rppt_add_scalar(src, srcGenericDesc, dst, dstGenericDesc, addTensor,
 *                   roiGenericSrc, roiType, handle, backend)
 */
#include "rpp/bench_generic_adapters.hpp"
#include <rpp/rppt_tensor_arithmetic_operations.h>

#include <vector>

namespace rppbench {

class AddScalarAdapter : public VoxelOpAdapter {
public:
    std::vector<RpptDataType> supportedDtypes() const override { return {RpptDataType::F32}; }

    void setup(const BenchContext &ctx, std::vector<TensorBuffer> &,
               std::vector<TensorBuffer> &) override {
        isHip_ = ctx.isHip;
        const auto value = ctx.param<float>("add", 40.0F);
        addTensor_ = static_cast<float *>(bench_pinned_alloc(sizeof(float) * ctx.batch, ctx.isHip));
        for (int i = 0; i < ctx.batch; ++i)
            addTensor_[i] = value;
    }

    RppStatus run(const BenchContext &ctx, std::vector<TensorBuffer> &src,
                  std::vector<TensorBuffer> &dst, rppHandle_t handle) override {
        return rppt_add_scalar(src[0].data, src[0].gdescPtr, dst[0].data, dst[0].gdescPtr,
                               addTensor_, src[0].roi3d, src[0].roi3dType, handle, ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(addTensor_, isHip_);
        addTensor_ = nullptr;
    }

private:
    float *addTensor_ = nullptr;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("add_scalar", AddScalarAdapter)

} // namespace rppbench
