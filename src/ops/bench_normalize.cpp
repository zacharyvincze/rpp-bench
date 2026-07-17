/**
 * @file bench_normalize.cpp
 * @brief Adapter for rppt_normalize - mean/stddev normalization of a generic ND tensor.
 *
 * Generic-descriptor op (RpptGenericDesc + flat roiTensor) via GenericOpAdapter.
 * `axisMask` selects which spatial axes are reduced (bit i => reduce spatial dim
 * i); mean/stddev are computed internally (computeMeanStddev=3), so the harness
 * only allocates the scratch tensors RPP writes into. Their length per sample is
 * the product of the *non-reduced* spatial dims (mirrors the test suite). F32.
 *
 * Signature:
 *   rppt_normalize(src, srcGenericDesc, dst, dstGenericDesc, axisMask, meanTensor,
 *                  stdDevTensor, computeMeanStddev, scale, shift, roiTensor,
 *                  handle, backend)
 */
#include "harness/bench_generic_adapters.hpp"
#include <rpp/rppt_tensor_statistical_operations.h>

#include <vector>

namespace rppbench {

class NormalizeAdapter : public GenericOpAdapter {
public:
    std::vector<RpptDataType> supportedDtypes() const override { return {RpptDataType::F32}; }

    void setup(const BenchContext &ctx, std::vector<TensorBuffer> &src,
               std::vector<TensorBuffer> &) override {
        isHip_ = ctx.isHip;
        axisMask_ = static_cast<Rpp32u>(ctx.param<int>("axisMask", 1));
        scale_ = ctx.param<float>("scale", 1.0F);
        shift_ = ctx.param<float>("shift", 0.0F);

        // Param length = product of the spatial dims NOT reduced by axisMask.
        const int nSpatial = src[0].numDims - 1;
        Rpp32u perSample = 1;
        for (int i = 0; i < nSpatial; ++i)
            if (!(axisMask_ & (1U << i)))
                perSample *= src[0].dims[1 + i];
        const size_t count = static_cast<size_t>(perSample) * ctx.batch;
        mean_ = static_cast<float *>(bench_pinned_alloc(sizeof(float) * count, ctx.isHip));
        stddev_ = static_cast<float *>(bench_pinned_alloc(sizeof(float) * count, ctx.isHip));
    }

    RppStatus run(const BenchContext &ctx, std::vector<TensorBuffer> &src,
                  std::vector<TensorBuffer> &dst, rppHandle_t handle) override {
        // 0th bit => compute mean internally, 1st bit => compute stddev internally.
        const Rpp8u computeMeanStddev = 3;
        return rppt_normalize(src[0].data, src[0].gdescPtr, dst[0].data, dst[0].gdescPtr, axisMask_,
                              mean_, stddev_, computeMeanStddev, scale_, shift_, src[0].roiTensor,
                              handle, ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(mean_, isHip_);
        mean_ = nullptr;
        bench_pinned_free(stddev_, isHip_);
        stddev_ = nullptr;
    }

private:
    Rpp32u axisMask_ = 1;
    float scale_ = 1.0F;
    float shift_ = 0.0F;
    float *mean_ = nullptr;
    float *stddev_ = nullptr;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("normalize", NormalizeAdapter)

} // namespace rppbench
