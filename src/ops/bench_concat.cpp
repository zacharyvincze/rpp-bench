/**
 * @file bench_concat.cpp
 * @brief Adapter for rppt_concat - concatenate two generic ND tensors along an axis.
 *
 * Two-source generic op. Both sources share one generic-ND shape (via
 * BroadcastOpAdapter); the destination is that shape with the concat axis
 * doubled. `axis` is a spatial-axis index (0-based, batch excluded) and defaults
 * to 0. Concat is a pure data move, so all four dtypes are allowed.
 *
 * Signature:
 *   rppt_concat(src1, src2, srcGenericDesc1, srcGenericDesc2, dst, dstGenericDesc,
 *               axisMask, roiTensor1, roiTensor2, handle, backend)
 */
#include "harness/bench_generic_adapters.hpp"
#include <rpp/rppt_tensor_geometric_augmentations.h>

#include <vector>

namespace rppbench {

class ConcatAdapter : public BroadcastOpAdapter {
public:
    std::vector<TensorSpec> dstSpecs(const BenchContext &ctx) const override {
        TensorSpec spec = tensorSpec(ctx);
        const Rpp32u axis = concatAxis(ctx, static_cast<int>(spec.dims.size()) - 1);
        spec.dims[1 + axis] *= 2; // concatenated extent along the chosen spatial axis
        return {spec};
    }

    void setup(const BenchContext &, std::vector<TensorBuffer> &,
               std::vector<TensorBuffer> &) override {}

    RppStatus run(const BenchContext &ctx, std::vector<TensorBuffer> &src,
                  std::vector<TensorBuffer> &dst, rppHandle_t handle) override {
        const Rpp32u axis = concatAxis(ctx, src[0].numDims - 1);
        return rppt_concat(src[0].data, src[1].data, src[0].gdescPtr, src[1].gdescPtr, dst[0].data,
                           dst[0].gdescPtr, axis, src[0].roiTensor, src[1].roiTensor, handle,
                           ctx.backend);
    }

private:
    static Rpp32u concatAxis(const BenchContext &ctx, int nSpatial) {
        const int axis = ctx.param<int>("axis", 0);
        if (axis < 0 || axis >= nSpatial)
            return 0;
        return static_cast<Rpp32u>(axis);
    }
};

REGISTER_RPP_BENCH("concat", ConcatAdapter)

} // namespace rppbench
