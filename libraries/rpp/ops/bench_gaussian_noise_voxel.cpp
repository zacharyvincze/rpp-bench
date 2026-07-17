/**
 * @file bench_gaussian_noise_voxel.cpp
 * @brief Adapter for rppt_gaussian_noise_voxel - add Gaussian noise to a 3D voxel tensor.
 *
 * Voxel op (RpptGenericDesc 5D + RpptROI3D) via VoxelOpAdapter. Per-sample mean
 * and stddev tensors plus a fixed seed (kept constant so runs are reproducible).
 * F32 only.
 *
 * Signature:
 *   rppt_gaussian_noise_voxel(src, srcGenericDesc, dst, dstGenericDesc, meanTensor,
 *                             stdDevTensor, seed, roiGenericSrc, roiType, handle,
 *                             backend)
 */
#include "rpp/bench_generic_adapters.hpp"
#include <rpp/rppt_tensor_effects_augmentations.h>

#include <vector>

namespace rppbench {

class GaussianNoiseVoxelAdapter : public VoxelOpAdapter {
public:
    std::vector<RpptDataType> supportedDtypes() const override { return {RpptDataType::F32}; }

    void setup(const BenchContext &ctx, std::vector<TensorBuffer> &,
               std::vector<TensorBuffer> &) override {
        isHip_ = ctx.isHip;
        const auto mean = ctx.param<float>("mean", 1.4F);
        const auto stddev = ctx.param<float>("stddev", 0.6F);
        mean_ = static_cast<float *>(bench_pinned_alloc(sizeof(float) * ctx.batch, ctx.isHip));
        stddev_ = static_cast<float *>(bench_pinned_alloc(sizeof(float) * ctx.batch, ctx.isHip));
        for (int i = 0; i < ctx.batch; ++i) {
            mean_[i] = mean;
            stddev_[i] = stddev;
        }
    }

    RppStatus run(const BenchContext &ctx, std::vector<TensorBuffer> &src,
                  std::vector<TensorBuffer> &dst, rppHandle_t handle) override {
        const auto seed = static_cast<Rpp32u>(ctx.param<int>("seed", 1255459));
        return rppt_gaussian_noise_voxel(src[0].data, src[0].gdescPtr, dst[0].data, dst[0].gdescPtr,
                                         mean_, stddev_, seed, src[0].roi3d, src[0].roi3dType,
                                         handle, ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(mean_, isHip_);
        mean_ = nullptr;
        bench_pinned_free(stddev_, isHip_);
        stddev_ = nullptr;
    }

private:
    float *mean_ = nullptr;
    float *stddev_ = nullptr;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("gaussian_noise_voxel", GaussianNoiseVoxelAdapter)

} // namespace rppbench
