/**
 * @file bench_dilate.cpp
 * @brief Adapter for rppt_dilate - kernelSize-scalar morphological dilation.
 *
 * Same source halo requirement as erode (offsetInBytes >= 12*(kernelSize/2) plus
 * a kernelSize/2 column margin, enforced on both HOST and HIP). kernelSize = 3/5/7/9.
 *
 * Signature:
 *   rppt_dilate(src, srcDesc, dst, dstDesc, kernelSize, roi, roiType, handle, backend)
 */
#include "harness/bench_registry.hpp"
#include <rpp/rppt_tensor_morphological_operations.h>

namespace rppbench {

class DilateAdapter : public OpAdapter {
public:
    // No HOST kernel in the installed RPP (the API is gated behind GPU_SUPPORT);
    // HOST calls return RPP_ERROR_NOT_IMPLEMENTED, so keep this HIP-only.
    std::vector<RppBackend> supportedBackends() const override {
        return {RppBackend::RPP_HIP_BACKEND};
    }

    void setup(const BenchContext &ctx, TensorBuffer &, TensorBuffer &) override {
        kernelSize_ = static_cast<Rpp32u>(ctx.param<int>("kernel_size", 3));
    }

    RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                  rppHandle_t handle) override {
        return rppt_dilate(src.data, src.descPtr, dst.data, dst.descPtr, kernelSize_, src.roi,
                           src.roiType, handle, ctx.backend);
    }

    int srcOffsetBytes(const BenchContext &ctx) const override {
        return 12 * (ctx.param<int>("kernel_size", 3) / 2);
    }
    int srcAdditionalStride(const BenchContext &ctx) const override {
        return ctx.param<int>("kernel_size", 3) / 2;
    }

private:
    Rpp32u kernelSize_ = 3;
};

REGISTER_RPP_BENCH("dilate", DilateAdapter)

} // namespace rppbench
