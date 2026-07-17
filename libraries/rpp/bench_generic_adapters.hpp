/**
 * @file bench_generic_adapters.hpp
 * @brief Adapter bases for ops that use RPP's generic tensor descriptor.
 *
 * A large family of `rppt_*` ops takes `RpptGenericDescPtr` (an arbitrary-rank
 * descriptor) instead of the 4D image `RpptDescPtr`. These three bases play the
 * same role for that dialect that `SimpleOpAdapter` plays for the image dialect:
 * they declare the tensor shapes (via `srcSpecs()`/`dstSpecs()`) so the runner
 * allocates the generic view, leaving subclasses to only issue the timed call.
 * Pick a base by the op's ROI shape:
 *   - GenericOpAdapter   - ND tensor + flat roiTensor (transpose, slice, ...)
 *   - VoxelOpAdapter     - 5D NCDHW/NDHWC + RpptROI3D (the *_scalar/voxel ops)
 *   - BroadcastOpAdapter - two ND sources + broadcast mode (tensor_*_tensor)
 * See CLAUDE.md ("Adding a generic-descriptor op") for the full recipe.
 */
#ifndef RPP_BENCH_GENERIC_ADAPTERS_HPP
#define RPP_BENCH_GENERIC_ADAPTERS_HPP

#include "rpp/bench_registry.hpp"

#include <vector>

namespace rppbench {

/**
 * @brief Base for ops that speak the generic (RpptGenericDesc) descriptor dialect.
 *
 * Mirrors SimpleOpAdapter's role for the image path: it wires up the common
 * shape so subclasses need only issue the rppt_* call. By default it declares a
 * single shape-preserving 1-in/1-out generic-ND tensor derived from the image
 * sweep point (ctx.layout picks channel-last NHWC vs channel-first NCHW), which
 * fits the elementwise ops (log) and the shape-preserving ones (normalize).
 * Override tensorSpec() to change rank/shape, or dstSpecs() when the destination
 * differs from the source (e.g. transpose permutes the dims).
 */
class GenericOpAdapter : public OpAdapter {
public:
    /**
     * @brief The per-op ND tensor shape for this sweep point.
     *
     * Default: a 2D (height x width) tensor with the channel folded in per
     * ctx.layout - dims {n,h,w,c} (NHWC) or {n,c,h,w} (NCHW).
     * @param ctx The fully-resolved sweep point.
     * @return The source/destination tensor spec.
     */
    virtual TensorSpec tensorSpec(const BenchContext &ctx) const;

    std::vector<TensorSpec> srcSpecs(const BenchContext &ctx) const override {
        return {tensorSpec(ctx)};
    }
    std::vector<TensorSpec> dstSpecs(const BenchContext &ctx) const override {
        return {tensorSpec(ctx)};
    }
};

/**
 * @brief Base for the 3D voxel ops (RpptGenericDesc + RpptROI3D).
 *
 * Emits a 5D NCDHW/NDHWC spec from the image sweep point plus a `depth` param
 * (the config's batch/size axes give n/w/h; depth has no matrix axis, so it is
 * read as an op param). The runner builds a per-sample RpptROI3D covering the
 * full volume; subclasses reach it via src[0].roi3d / src[0].roi3dType.
 */
class VoxelOpAdapter : public GenericOpAdapter {
public:
    /**
     * @brief Voxel depth (Z extent). No matrix axis; read from op params.
     * @param ctx The fully-resolved sweep point.
     * @return Depth in voxels (default 4).
     */
    virtual int depth(const BenchContext &ctx) const { return ctx.param<int>("depth", 4); }

    TensorSpec tensorSpec(const BenchContext &ctx) const override;
};

/**
 * @brief Base for the two-source broadcast ops (tensor_add_tensor, ...).
 *
 * Both sources use the same generic-ND spec (identical dims => broadcast disabled,
 * matching how the image two-source ops share one shape); the broadcast mode is
 * exposed via broadcastMode() for subclasses that pass it to the rppt_* call.
 */
class BroadcastOpAdapter : public GenericOpAdapter {
public:
    int numSrc() const override { return 2; }

    std::vector<TensorSpec> srcSpecs(const BenchContext &ctx) const override {
        return {tensorSpec(ctx), tensorSpec(ctx)};
    }

    /**
     * @brief Broadcast mode for the two-source call.
     * @return RPP_BROADCAST_DISABLE (both sources share dims). Override to sweep it.
     */
    virtual RpptBroadcastMode broadcastMode(const BenchContext &) const {
        return RpptBroadcastMode::RPP_BROADCAST_DISABLE;
    }
};

} // namespace rppbench

#endif // RPP_BENCH_GENERIC_ADAPTERS_HPP
