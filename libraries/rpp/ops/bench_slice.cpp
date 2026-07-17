/**
 * @file bench_slice.cpp
 * @brief Adapter for rppt_slice - extract a sub-tensor from a generic ND tensor.
 *
 * Generic-descriptor op (RpptGenericDesc + flat roiTensor) via GenericOpAdapter.
 * anchorTensor/shapeTensor hold the start and length per spatial axis (batch
 * excluded), sized batch*nSpatial. The slice covers a `fraction` (default 1.0 =
 * the full tensor, which keeps the case valid without padding) of each spatial
 * axis from anchor 0; the destination shape is the slice shape.
 *
 * Signature:
 *   rppt_slice(src, srcGenericDesc, dst, dstGenericDesc, anchorTensor, shapeTensor,
 *              fillValue, enablePadding, roiTensor, handle, backend)
 */
#include "rpp/bench_generic_adapters.hpp"
#include <rpp/rppt_tensor_geometric_augmentations.h>

#include <algorithm>
#include <vector>

namespace rppbench {

class SliceAdapter : public GenericOpAdapter {
public:
    std::vector<RpptDataType> supportedDtypes() const override {
        return {RpptDataType::U8, RpptDataType::F32};
    }

    std::vector<TensorSpec> dstSpecs(const BenchContext &ctx) const override {
        TensorSpec spec = tensorSpec(ctx);
        const int nSpatial = static_cast<int>(spec.dims.size()) - 1;
        const std::vector<Rpp32u> shape = sliceShape(ctx, spec.dims);
        for (int i = 0; i < nSpatial; ++i)
            spec.dims[1 + i] = shape[i];
        return {spec};
    }

    void setup(const BenchContext &ctx, std::vector<TensorBuffer> &src,
               std::vector<TensorBuffer> &) override {
        isHip_ = ctx.isHip;
        const int nSpatial = src[0].numDims - 1;
        const std::vector<Rpp32u> shape = sliceShape(ctx, src[0].dims);

        const size_t count = static_cast<size_t>(ctx.batch) * nSpatial;
        anchor_ = static_cast<Rpp32s *>(bench_pinned_alloc(sizeof(Rpp32s) * count, ctx.isHip));
        shape_ = static_cast<Rpp32s *>(bench_pinned_alloc(sizeof(Rpp32s) * count, ctx.isHip));
        for (int b = 0; b < ctx.batch; ++b)
            for (int j = 0; j < nSpatial; ++j) {
                anchor_[b * nSpatial + j] = 0;
                shape_[b * nSpatial + j] = static_cast<Rpp32s>(shape[j]);
            }
        // Single fill value used when padding is enabled; here padding is off, but
        // the API still dereferences the pointer, so keep a valid one.
        fillValue_ = bench_pinned_alloc(sizeof(float), ctx.isHip);
        *static_cast<float *>(fillValue_) = 0.0F;
    }

    RppStatus run(const BenchContext &ctx, std::vector<TensorBuffer> &src,
                  std::vector<TensorBuffer> &dst, rppHandle_t handle) override {
        const bool enablePadding = false;
        return rppt_slice(src[0].data, src[0].gdescPtr, dst[0].data, dst[0].gdescPtr, anchor_,
                          shape_, fillValue_, enablePadding, src[0].roiTensor, handle, ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(anchor_, isHip_);
        anchor_ = nullptr;
        bench_pinned_free(shape_, isHip_);
        shape_ = nullptr;
        bench_pinned_free(fillValue_, isHip_);
        fillValue_ = nullptr;
    }

private:
    // Slice length per spatial axis: fraction of the source extent, at least 1.
    std::vector<Rpp32u> sliceShape(const BenchContext &ctx, const std::vector<Rpp32u> &dims) const {
        const float fraction = std::clamp(ctx.param<float>("fraction", 1.0F), 0.0F, 1.0F);
        const int nSpatial = static_cast<int>(dims.size()) - 1;
        std::vector<Rpp32u> shape(nSpatial);
        for (int i = 0; i < nSpatial; ++i) {
            auto len = static_cast<Rpp32u>(static_cast<float>(dims[1 + i]) * fraction);
            shape[i] = std::max<Rpp32u>(1, len);
        }
        return shape;
    }

    Rpp32s *anchor_ = nullptr;
    Rpp32s *shape_ = nullptr;
    void *fillValue_ = nullptr;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("slice", SliceAdapter)

} // namespace rppbench
