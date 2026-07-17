/**
 * @file rpp_case.cpp
 * @brief RppCase implementation (see rpp_case.hpp).
 *
 * The RPP-specific per-case lifecycle: allocate buffers (image or generic path),
 * create the handle + HIP stream, run the adapter, and on each timed call refresh
 * the ROI, launch, and synchronize. This is the code the neutral core runner used
 * to carry inline; keeping it here is what makes the core dependency-free.
 */
#include "rpp/rpp_case.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

#if RPP_BENCH_HIP
#include <hip/hip_runtime.h>
#endif

namespace rppbench {

RppCase::RppCase(AdapterFactory factory, const BenchPoint &point)
    : factory_(std::move(factory)), ctx_(BenchContext::from(point)) {}

RppCase::~RppCase() {
    if (adapter_ && ready_)
        adapter_->teardown();
    adapter_.reset();
    srcs_.clear(); // ~TensorBuffer frees each buffer
    dsts_.clear();
    if (handle_) {
        rppDestroy(handle_, ctx_.backend);
        handle_ = nullptr;
    }
#if RPP_BENCH_HIP
    if (hipStream_) {
        (void)hipStreamDestroy(static_cast<hipStream_t>(hipStream_));
        hipStream_ = nullptr;
    }
#endif
}

std::string RppCase::setup() {
    try {
        adapter_ = factory_();
        // Generic-descriptor ops declare their tensor shapes via srcSpecs()/
        // dstSpecs(); an empty list selects the legacy image path. The two paths
        // differ only in how each TensorBuffer is configured - image via
        // init(w,h), generic via initGeneric(spec) - after which they are treated
        // uniformly (fill, resetRoi, free).
        const std::vector<TensorSpec> srcSpecs = adapter_->srcSpecs(ctx_);
        const std::vector<TensorSpec> dstSpecs = adapter_->dstSpecs(ctx_);

        if (srcSpecs.empty()) {
            // Image path: every source shares dims/dtype/layout (the two-source
            // rppt_* calls take a single srcDesc); each is filled independently.
            srcs_.resize(std::max(1, adapter_->numSrc()));
            for (auto &s : srcs_) {
                s.init(ctx_.backend, ctx_.dtype, ctx_.layout, ctx_.batch, ctx_.width, ctx_.height,
                       adapter_->srcOffsetBytes(ctx_), adapter_->srcAdditionalStride(ctx_));
                s.fill();
            }
        } else {
            srcs_.resize(srcSpecs.size());
            for (size_t i = 0; i < srcSpecs.size(); ++i) {
                srcs_[i].initGeneric(ctx_.backend, srcSpecs[i]);
                srcs_[i].fill();
            }
        }

        if (dstSpecs.empty()) {
            // Destinations use the (possibly resized) dst dims.
            dsts_.resize(std::max(1, adapter_->numDst()));
            for (auto &d : dsts_)
                d.init(ctx_.backend, ctx_.dtype, ctx_.layout, ctx_.batch, ctx_.dstWidth,
                       ctx_.dstHeight);
        } else {
            dsts_.resize(dstSpecs.size());
            for (size_t i = 0; i < dstSpecs.size(); ++i)
                dsts_[i].initGeneric(ctx_.backend, dstSpecs[i]);
        }

        void *stream = nullptr;
#if RPP_BENCH_HIP
        if (ctx_.isHip) {
            hipStream_t s = nullptr;
            if (hipStreamCreate(&s) != hipSuccess)
                return "hipStreamCreate failed";
            hipStream_ = s;
            stream = s;
        }
#endif
        if (rppCreate(&handle_, ctx_.batch, 0, stream, ctx_.backend) != rppStatusSuccess)
            return "rppCreate failed";
        adapter_->setup(ctx_, srcs_, dsts_);
        ready_ = true;
        return {};
    } catch (const std::exception &e) {
        return std::string("case setup failed: ") + e.what();
    }
}

std::string RppCase::runOnce() {
    // Some ops convert the ROI in place; refresh it each call so repeated
    // invocations don't accumulate and drive indices out of bounds.
    for (auto &s : srcs_)
        s.resetRoi();
    for (auto &d : dsts_)
        d.resetRoi();
    RppStatus st = adapter_->run(ctx_, srcs_, dsts_, handle_);
#if RPP_BENCH_HIP
    if (ctx_.isHip) {
        // The rppt_* launch is async; a kernel fault surfaces here. Capture it so
        // a single failure doesn't spam thousands of launches.
        hipError_t he = hipStreamSynchronize(static_cast<hipStream_t>(hipStream_));
        if (he != hipSuccess)
            return std::string("HIP kernel fault: ") + hipGetErrorString(he);
    }
#endif
    if (st != RPP_SUCCESS)
        return "rppt call returned status " + std::to_string(st);
    return {};
}

size_t RppCase::srcDataBytes() const {
    return srcs_.empty() ? 0 : srcs_[0].dataBytes;
}

} // namespace rppbench
