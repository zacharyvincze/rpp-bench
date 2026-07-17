/**
 * @file bench_flip_voxel.cpp
 * @brief Adapter for rppt_flip_voxel - mirror a 3D voxel tensor along chosen axes.
 *
 * Voxel op (RpptGenericDesc 5D + RpptROI3D) via VoxelOpAdapter. Three per-sample
 * Rpp32u flags select horizontal/vertical/depth mirroring; defaults mirror
 * horizontally. Supports U8 and F32.
 *
 * Signature:
 *   rppt_flip_voxel(src, srcGenericDesc, dst, dstGenericDesc, horizontalTensor,
 *                   verticalTensor, depthTensor, roiGenericSrc, roiType, handle,
 *                   backend)
 */
#include "rpp/bench_generic_adapters.hpp"
#include <rpp/rppt_tensor_geometric_augmentations.h>

#include <vector>

namespace rppbench {

class FlipVoxelAdapter : public VoxelOpAdapter {
public:
    std::vector<RpptDataType> supportedDtypes() const override {
        return {RpptDataType::U8, RpptDataType::F32};
    }

    void setup(const BenchContext &ctx, std::vector<TensorBuffer> &,
               std::vector<TensorBuffer> &) override {
        isHip_ = ctx.isHip;
        const auto h = static_cast<Rpp32u>(ctx.param<int>("horizontal", 1));
        const auto v = static_cast<Rpp32u>(ctx.param<int>("vertical", 0));
        const auto d = static_cast<Rpp32u>(ctx.param<int>("depth_flip", 0));
        horizontal_ =
            static_cast<Rpp32u *>(bench_pinned_alloc(sizeof(Rpp32u) * ctx.batch, ctx.isHip));
        vertical_ =
            static_cast<Rpp32u *>(bench_pinned_alloc(sizeof(Rpp32u) * ctx.batch, ctx.isHip));
        depth_ = static_cast<Rpp32u *>(bench_pinned_alloc(sizeof(Rpp32u) * ctx.batch, ctx.isHip));
        for (int i = 0; i < ctx.batch; ++i) {
            horizontal_[i] = h;
            vertical_[i] = v;
            depth_[i] = d;
        }
    }

    RppStatus run(const BenchContext &ctx, std::vector<TensorBuffer> &src,
                  std::vector<TensorBuffer> &dst, rppHandle_t handle) override {
        return rppt_flip_voxel(src[0].data, src[0].gdescPtr, dst[0].data, dst[0].gdescPtr,
                               horizontal_, vertical_, depth_, src[0].roi3d, src[0].roi3dType,
                               handle, ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(horizontal_, isHip_);
        horizontal_ = nullptr;
        bench_pinned_free(vertical_, isHip_);
        vertical_ = nullptr;
        bench_pinned_free(depth_, isHip_);
        depth_ = nullptr;
    }

private:
    Rpp32u *horizontal_ = nullptr;
    Rpp32u *vertical_ = nullptr;
    Rpp32u *depth_ = nullptr;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("flip_voxel", FlipVoxelAdapter)

} // namespace rppbench
