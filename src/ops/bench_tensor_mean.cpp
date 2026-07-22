/**
 * @file bench_tensor_mean.cpp
 * @brief Adapter for rppt_tensor_mean - channel-wise + total mean reduction of a tensor.
 *
 * tensor_mean is a reduction, not an image-to-image op: it reads one source and
 * writes a plain results array (there is no destination image). The array holds
 * the channel-wise means plus a total mean per sample - length n for 1-channel
 * input and n*4 (R/G/B/total) for 3-channel. The mean is always emitted as
 * Rpp32f regardless of source depth (see the RPP test suite's
 * reductionFuncResultArr typing), so we allocate sizeof(Rpp32f) per element.
 *
 * The runner always hands us a destination image buffer (its minimum is one),
 * but a reduction has no image output, so we ignore dst entirely.
 *
 * Signature:
 *   rppt_tensor_mean(src, srcDesc, meanArr, meanArrLength, roi, roiType, handle, backend)
 */
#include "harness/bench_registry.hpp"
#include <rpp/rppt_tensor_statistical_operations.h>

namespace rppbench {

class TensorMeanAdapter : public OpAdapter {
public:
    void setup(const BenchContext &ctx, std::vector<TensorBuffer> &src,
               std::vector<TensorBuffer> &) override {
        isHip_ = ctx.isHip;
        // Length contract (from the RPP header): n means for 1-channel input,
        // n*4 (R/G/B/total) for 3-channel.
        const Rpp32u channels = src[0].descPtr->c;
        meanArrLength_ =
            (channels == 1) ? static_cast<Rpp32u>(ctx.batch) : static_cast<Rpp32u>(ctx.batch) * 4;
        // Mean is always a 32-bit float output.
        meanArr_ = bench_pinned_alloc(sizeof(Rpp32f) * meanArrLength_, ctx.isHip);
    }

    RppStatus run(const BenchContext &ctx, std::vector<TensorBuffer> &src,
                  std::vector<TensorBuffer> &, rppHandle_t handle) override {
        return rppt_tensor_mean(src[0].data, src[0].descPtr, meanArr_, meanArrLength_, src[0].roi,
                                src[0].roiType, handle, ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(meanArr_, isHip_);
        meanArr_ = nullptr;
    }

private:
    void *meanArr_ = nullptr;
    Rpp32u meanArrLength_ = 0;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("tensor_mean", TensorMeanAdapter)

} // namespace rppbench
