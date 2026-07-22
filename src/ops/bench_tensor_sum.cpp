/**
 * @file bench_tensor_sum.cpp
 * @brief Adapter for rppt_tensor_sum - channel-wise + total reduction of a tensor.
 *
 * tensor_sum is a reduction, not an image-to-image op: it reads one source and
 * writes a plain results array (there is no destination image). The array holds
 * the channel-wise sums plus a total per sample - length n for 1-channel input
 * and n*4 for 3-channel (R/G/B/total). Because an 8-bit sum overflows, the result
 * element type is wider than the source: Rpp64u for U8/I8 inputs and Rpp32f for
 * F16/F32 (see the RPP test suite's reductionFuncResultArr typing). We allocate
 * the array at sizeof(Rpp64u) per element, which covers both cases.
 *
 * The runner always hands us a destination image buffer (its minimum is one),
 * but a reduction has no image output, so we ignore dst entirely.
 *
 * Signature:
 *   rppt_tensor_sum(src, srcDesc, tensorSumArr, tensorSumArrLength, roi, roiType,
 *                   handle, backend)
 */
#include "harness/bench_registry.hpp"
#include <rpp/rppt_tensor_statistical_operations.h>

namespace rppbench {

class TensorSumAdapter : public OpAdapter {
public:
    void setup(const BenchContext &ctx, std::vector<TensorBuffer> &src,
               std::vector<TensorBuffer> &) override {
        isHip_ = ctx.isHip;
        // Length contract (from the RPP header): n sums for 1-channel input,
        // n*4 (R/G/B/total) for 3-channel.
        const Rpp32u channels = src[0].descPtr->c;
        sumArrLength_ =
            (channels == 1) ? static_cast<Rpp32u>(ctx.batch) : static_cast<Rpp32u>(ctx.batch) * 4;
        // Rpp64u per element is wide enough for both the 64-bit integer sums
        // (U8/I8) and the 32-bit float sums (F16/F32).
        sumArr_ = bench_pinned_alloc(sizeof(Rpp64u) * sumArrLength_, ctx.isHip);
    }

    RppStatus run(const BenchContext &ctx, std::vector<TensorBuffer> &src,
                  std::vector<TensorBuffer> &, rppHandle_t handle) override {
        return rppt_tensor_sum(src[0].data, src[0].descPtr, sumArr_, sumArrLength_, src[0].roi,
                               src[0].roiType, handle, ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(sumArr_, isHip_);
        sumArr_ = nullptr;
    }

private:
    void *sumArr_ = nullptr;
    Rpp32u sumArrLength_ = 0;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("tensor_sum", TensorSumAdapter)

} // namespace rppbench
