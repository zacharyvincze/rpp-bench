/**
 * @file bench_tensor_stddev.cpp
 * @brief Adapter for rppt_tensor_stddev - channel-wise + total standard-deviation reduction.
 *
 * tensor_stddev is a reduction, not an image-to-image op: it reads one source and
 * writes a plain results array (there is no destination image). The array holds
 * the channel-wise stddevs plus a total per sample - length n for 1-channel input
 * and n*4 (R/G/B/total) for 3-channel. The stddev is always emitted as Rpp32f
 * regardless of source depth (see the RPP test suite's reductionFuncResultArr
 * typing), so we allocate sizeof(Rpp32f) per element.
 *
 * Unlike the other statistical reductions, stddev takes an extra input meanTensor:
 * an Rpp32f array of size batch*4 (MeanR, MeanG, MeanB, MeanImage per sample) that
 * the stddev is computed against. We allocate it as a valid pinned/HIP buffer; for
 * a timing benchmark its contents need not be the true per-image means.
 *
 * The runner always hands us a destination image buffer (its minimum is one),
 * but a reduction has no image output, so we ignore dst entirely.
 *
 * Signature:
 *   rppt_tensor_stddev(src, srcDesc, stddevArr, stddevArrLength, meanTensor, roi, roiType,
 *                      handle, backend)
 */
#include "harness/bench_registry.hpp"
#include <rpp/rppt_tensor_statistical_operations.h>

namespace rppbench {

class TensorStddevAdapter : public OpAdapter {
public:
    void setup(const BenchContext &ctx, std::vector<TensorBuffer> &src,
               std::vector<TensorBuffer> &) override {
        isHip_ = ctx.isHip;
        // Length contract (from the RPP header): n stddevs for 1-channel input,
        // n*4 (R/G/B/total) for 3-channel.
        const Rpp32u channels = src[0].descPtr->c;
        stddevArrLength_ =
            (channels == 1) ? static_cast<Rpp32u>(ctx.batch) : static_cast<Rpp32u>(ctx.batch) * 4;
        // Stddev is always a 32-bit float output.
        stddevArr_ = bench_pinned_alloc(sizeof(Rpp32f) * stddevArrLength_, ctx.isHip);
        // meanTensor is a batch*4 float array the stddev is computed against.
        meanTensor_ = static_cast<Rpp32f *>(
            bench_pinned_alloc(sizeof(Rpp32f) * static_cast<Rpp32u>(ctx.batch) * 4, ctx.isHip));
    }

    RppStatus run(const BenchContext &ctx, std::vector<TensorBuffer> &src,
                  std::vector<TensorBuffer> &, rppHandle_t handle) override {
        return rppt_tensor_stddev(src[0].data, src[0].descPtr, stddevArr_, stddevArrLength_,
                                  meanTensor_, src[0].roi, src[0].roiType, handle, ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(stddevArr_, isHip_);
        bench_pinned_free(meanTensor_, isHip_);
        stddevArr_ = nullptr;
        meanTensor_ = nullptr;
    }

private:
    void *stddevArr_ = nullptr;
    Rpp32f *meanTensor_ = nullptr;
    Rpp32u stddevArrLength_ = 0;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("tensor_stddev", TensorStddevAdapter)

} // namespace rppbench
