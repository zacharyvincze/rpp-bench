/**
 * @file bench_tensor_min.cpp
 * @brief Adapter for rppt_tensor_min - channel-wise + overall min reduction of a tensor.
 *
 * tensor_min is a reduction, not an image-to-image op: it reads one source and
 * writes a plain results array (there is no destination image). The array holds
 * the channel-wise mins plus an overall min per sample - length n for 1-channel
 * input and n*4 (R/G/B/overall) for 3-channel. Unlike tensor_sum the result
 * element type matches the source depth (Rpp8u/Rpp8s/Rpp16f/Rpp32f), so the
 * widest element is Rpp32f; allocating sizeof(Rpp32f) per element covers every
 * supported dtype (see the RPP test suite's reductionFuncResultArr typing).
 *
 * The runner always hands us a destination image buffer (its minimum is one),
 * but a reduction has no image output, so we ignore dst entirely.
 *
 * Signature:
 *   rppt_tensor_min(src, srcDesc, minArr, minArrLength, roi, roiType, handle, backend)
 */
#include "harness/bench_registry.hpp"
#include <rpp/rppt_tensor_statistical_operations.h>

namespace rppbench {

class TensorMinAdapter : public OpAdapter {
public:
    void setup(const BenchContext &ctx, std::vector<TensorBuffer> &src,
               std::vector<TensorBuffer> &) override {
        isHip_ = ctx.isHip;
        // Length contract (from the RPP header): n mins for 1-channel input,
        // n*4 (R/G/B/overall) for 3-channel.
        const Rpp32u channels = src[0].descPtr->c;
        minArrLength_ =
            (channels == 1) ? static_cast<Rpp32u>(ctx.batch) : static_cast<Rpp32u>(ctx.batch) * 4;
        // Result element matches source depth; Rpp32f (4 bytes) is the widest.
        minArr_ = bench_pinned_alloc(sizeof(Rpp32f) * minArrLength_, ctx.isHip);
    }

    RppStatus run(const BenchContext &ctx, std::vector<TensorBuffer> &src,
                  std::vector<TensorBuffer> &, rppHandle_t handle) override {
        return rppt_tensor_min(src[0].data, src[0].descPtr, minArr_, minArrLength_, src[0].roi,
                               src[0].roiType, handle, ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(minArr_, isHip_);
        minArr_ = nullptr;
    }

private:
    void *minArr_ = nullptr;
    Rpp32u minArrLength_ = 0;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("tensor_min", TensorMinAdapter)

} // namespace rppbench
