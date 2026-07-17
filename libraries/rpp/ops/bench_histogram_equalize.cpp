/**
 * @file bench_histogram_equalize.cpp
 * @brief Adapter for rppt_histogram_equalize - contrast stretch via CDF, no params.
 *
 * U8 only (the kernel builds a 256-bin histogram); works on 1- or 3-channel
 * layouts. Single-source, ROI, no param tensors - a copy-style adapter with a
 * dtype constraint.
 *
 * HOST only: the HIP kernel hipMalloc's an internal scratch buffer on every call
 * (its signature has no scratch param, so the caller cannot manage it) and never
 * frees it. Across a sweep's warmup + timed iterations this leaks device memory
 * until hipMalloc OOMs, which then also takes down the next case's rocRAND input
 * fill. Restricting to HOST sidesteps the leak; revisit if RPP fixes the kernel.
 *
 * Signature:
 *   rppt_histogram_equalize(src, srcDesc, dst, dstDesc, roi, roiType, handle, backend)
 */
#include "rpp/bench_simple_adapter.hpp"
#include <rpp/rppt_tensor_color_augmentations.h>

namespace rppbench {

class HistogramEqualizeAdapter : public SimpleOpAdapter {
public:
    std::vector<RpptDataType> supportedDtypes() const override { return {RpptDataType::U8}; }

    std::vector<RppBackend> supportedBackends() const override {
        return {RppBackend::RPP_HOST_BACKEND};
    }

    void setup(const BenchContext &, TensorBuffer &, TensorBuffer &) override {}

    RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                  rppHandle_t handle) override {
        return rppt_histogram_equalize(src.data, src.descPtr, dst.data, dst.descPtr, src.roi,
                                       src.roiType, handle, ctx.backend);
    }
};

REGISTER_RPP_BENCH("histogram_equalize", HistogramEqualizeAdapter)

} // namespace rppbench
