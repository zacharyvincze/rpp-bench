/**
 * @file bench_generic_adapters.cpp
 * @brief Out-of-line spec builders for the generic-descriptor adapter bases.
 *
 * The shape math (mapping the image sweep point + ctx.layout onto an ND or 5D
 * voxel tensor) lives here to keep bench_generic_adapters.hpp lean; the trivial
 * one-liners stay inline in the header.
 */
#include "rpp/bench_generic_adapters.hpp"

#include "rpp/rpp_enums.hpp"

namespace rppbench {

TensorSpec GenericOpAdapter::tensorSpec(const BenchContext &ctx) const {
    LayoutInfo li = layout_info(ctx.layout);
    TensorSpec spec;
    spec.kind = TensorKind::GENERIC_ND;
    spec.layout = li.rpptLayout;
    spec.dtype = ctx.dtype;
    const auto n = static_cast<Rpp32u>(ctx.batch);
    const auto h = static_cast<Rpp32u>(ctx.height);
    const auto w = static_cast<Rpp32u>(ctx.width);
    const auto c = static_cast<Rpp32u>(li.channels);
    if (li.rpptLayout == RpptLayout::NHWC)
        spec.dims = {n, h, w, c};
    else // NCHW
        spec.dims = {n, c, h, w};
    return spec;
}

TensorSpec VoxelOpAdapter::tensorSpec(const BenchContext &ctx) const {
    LayoutInfo li = layout_info(ctx.layout);
    TensorSpec spec;
    spec.kind = TensorKind::VOXEL;
    spec.dtype = ctx.dtype;
    const auto n = static_cast<Rpp32u>(ctx.batch);
    const auto d = static_cast<Rpp32u>(depth(ctx));
    const auto h = static_cast<Rpp32u>(ctx.height);
    const auto w = static_cast<Rpp32u>(ctx.width);
    const auto c = static_cast<Rpp32u>(li.channels);
    if (li.rpptLayout == RpptLayout::NHWC) { // packed -> NDHWC
        spec.layout = RpptLayout::NDHWC;
        spec.dims = {n, d, h, w, c};
    } else { // planar -> NCDHW
        spec.layout = RpptLayout::NCDHW;
        spec.dims = {n, c, d, h, w};
    }
    return spec;
}

} // namespace rppbench
