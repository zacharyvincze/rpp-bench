/**
 * @file bench_tensor_max.cpp
 * @brief Adapter for rppt_tensor_max - channel-wise + overall max reduction of a tensor.
 *
 * tensor_max is a reduction, not an image-to-image op: it reads one source and
 * writes a plain results array (there is no destination image). The array holds
 * the channel-wise maxes plus an overall max per sample - length n for 1-channel
 * input and n*4 (R/G/B/overall) for 3-channel. Like tensor_min the result element
 * type matches the source depth (Rpp8u/Rpp8s/Rpp16f/Rpp32f), so the widest element
 * is Rpp32f; allocating sizeof(Rpp32f) per element covers every supported dtype
 * (see the RPP test suite's reductionFuncResultArr typing).
 *
 * The runner always hands us a destination image buffer (its minimum is one),
 * but a reduction has no image output, so we ignore dst entirely.
 *
 * Signature:
 *   rppt_tensor_max(src, srcDesc, maxArr, maxArrLength, roi, roiType, handle, backend)
 */
#include "harness/bench_registry.hpp"
#include <rpp/rppt_tensor_statistical_operations.h>

namespace rppbench {

class TensorMaxAdapter : public OpAdapter {
public:
    void setup(const BenchContext &ctx, std::vector<TensorBuffer> &src,
               std::vector<TensorBuffer> &) override {
        isHip_ = ctx.isHip;
        // Length contract (from the RPP header): n maxes for 1-channel input,
        // n*4 (R/G/B/overall) for 3-channel.
        const Rpp32u channels = src[0].descPtr->c;
        maxArrLength_ =
            (channels == 1) ? static_cast<Rpp32u>(ctx.batch) : static_cast<Rpp32u>(ctx.batch) * 4;
        // Result element matches source depth; Rpp32f (4 bytes) is the widest.
        maxArr_ = bench_pinned_alloc(sizeof(Rpp32f) * maxArrLength_, ctx.isHip);
    }

    RppStatus run(const BenchContext &ctx, std::vector<TensorBuffer> &src,
                  std::vector<TensorBuffer> &, rppHandle_t handle) override {
        return rppt_tensor_max(src[0].data, src[0].descPtr, maxArr_, maxArrLength_, src[0].roi,
                               src[0].roiType, handle, ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(maxArr_, isHip_);
        maxArr_ = nullptr;
    }

private:
    void *maxArr_ = nullptr;
    Rpp32u maxArrLength_ = 0;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("tensor_max", TensorMaxAdapter)

} // namespace rppbench
