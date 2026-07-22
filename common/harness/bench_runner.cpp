#include "harness/bench_runner.hpp"

#include <benchmark/benchmark.h>

#include "harness/bench_name.hpp"
#include "harness/bench_registry.hpp"

#include <algorithm>
#include <cstdio>
#include <memory>
#include <string>

#if RPP_BENCH_HIP
#include <hip/hip_runtime.h>
#endif

namespace rppbench {

namespace {

bool supports(const std::vector<RpptDataType> &v, RpptDataType t) {
    return std::find(v.begin(), v.end(), t) != v.end();
}
bool supports(const std::vector<Layout> &v, Layout l) {
    return std::find(v.begin(), v.end(), l) != v.end();
}
bool supports(const std::vector<RppBackend> &v, RppBackend b) {
    return std::find(v.begin(), v.end(), b) != v.end();
}

// Expensive per-case resources (handle, stream, buffers, adapter). Google
// Benchmark determines a case's iteration count by re-invoking its function
// several times (calibration ramp), then again per repetition. Building these
// inside the function meant paying rppCreate + hipMalloc + H2D ~5x per case.
// Instead we build once, keyed by caseId, and reuse across re-invocations.
struct CaseResources {
    int caseId = -1;     // which registered case currently owns these
    bool ready = false;  // successfully built
    bool failed = false; // build failed; skip without retrying every re-invoke
    bool warmed = false; // warmup already run for this case
    std::string error;

    BenchContext ctx;
    std::unique_ptr<OpAdapter> adapter;
    std::vector<TensorBuffer> srcs, dsts;
    rppHandle_t handle = nullptr;
#if RPP_BENCH_HIP
    hipStream_t hipStream = nullptr;
#endif

    void release() {
        if (adapter && ready)
            adapter->teardown();
        adapter.reset();
        for (auto &s : srcs)
            s.free();
        for (auto &d : dsts)
            d.free();
        srcs.clear();
        dsts.clear();
        if (handle) {
            rppDestroy(handle, ctx.backend);
            handle = nullptr;
        }
#if RPP_BENCH_HIP
        if (hipStream) {
            (void)hipStreamDestroy(hipStream);
            hipStream = nullptr;
        }
#endif
        ready = failed = warmed = false;
        caseId = -1;
    }

    void build(int id, const AdapterFactory &factory, const BenchContext &c) {
        release();
        caseId = id;
        ctx = c;
        try {
            adapter = factory();
            // Generic-descriptor ops declare their tensor shapes via srcSpecs()/
            // dstSpecs(); an empty list selects the legacy image path. The two
            // paths differ only in how each TensorBuffer is configured - image via
            // init(w,h), generic via initGeneric(spec) - after which the runner
            // treats them uniformly (fill, resetRoi, free).
            const std::vector<TensorSpec> srcSpecs = adapter->srcSpecs(ctx);
            const std::vector<TensorSpec> dstSpecs = adapter->dstSpecs(ctx);

            if (srcSpecs.empty()) {
                // Image path: every source shares dims/dtype/layout (the two-source
                // rppt_* calls take a single srcDesc); each is filled independently.
                srcs.resize(std::max(1, adapter->numSrc()));
                for (auto &s : srcs) {
                    s.init(ctx.backend, ctx.dtype, ctx.layout, ctx.batch, ctx.width, ctx.height,
                           adapter->srcOffsetBytes(ctx), adapter->srcAdditionalStride(ctx));
                    s.fill();
                }
            } else {
                srcs.resize(srcSpecs.size());
                for (size_t i = 0; i < srcSpecs.size(); ++i) {
                    srcs[i].initGeneric(ctx.backend, srcSpecs[i]);
                    srcs[i].fill();
                }
            }

            if (dstSpecs.empty()) {
                // Destinations use the (possibly resized) dst dims.
                dsts.resize(std::max(1, adapter->numDst()));
                for (auto &d : dsts)
                    d.init(ctx.backend, ctx.dtype, ctx.layout, ctx.batch, ctx.dstWidth,
                           ctx.dstHeight);
            } else {
                dsts.resize(dstSpecs.size());
                for (size_t i = 0; i < dstSpecs.size(); ++i)
                    dsts[i].initGeneric(ctx.backend, dstSpecs[i]);
            }

            void *stream = nullptr;
#if RPP_BENCH_HIP
            if (ctx.isHip) {
                if (hipStreamCreate(&hipStream) != hipSuccess) {
                    failed = true;
                    error = "hipStreamCreate failed";
                    return;
                }
                stream = hipStream;
            }
#endif
            if (rppCreate(&handle, ctx.batch, 0, stream, ctx.backend) != rppStatusSuccess) {
                failed = true;
                error = "rppCreate failed";
                return;
            }
            adapter->setup(ctx, srcs, dsts);
            ready = true;
        } catch (const std::exception &e) {
            failed = true;
            error = std::string("case setup failed: ") + e.what();
        }
    }
};

// The single live case. Benchmarks run sequentially, so building a new case
// releases the previous one - at most one case's GPU memory is resident.
// (Assumes single-threaded benchmarks, which is how they are registered.)
CaseResources g_case;

/**
 * @brief One invocation of the op: refresh ROI, launch, and (on HIP) synchronize.
 *
 * On HIP it synchronizes so the kernel has actually finished. Shared verbatim by
 * the warmup and timed loops so warmup exercises exactly what is measured.
 * @param ctx The fully-resolved sweep point.
 * @param srcs Source buffers.
 * @param dsts Destination buffers.
 * @return An error string on failure, empty on success.
 */
std::string run_one(const BenchContext &ctx, std::vector<TensorBuffer> &srcs,
                    std::vector<TensorBuffer> &dsts) {
    // Some ops convert the ROI in place; refresh it each call so repeated
    // invocations don't accumulate and drive indices out of bounds.
    for (auto &s : srcs)
        s.resetRoi();
    for (auto &d : dsts)
        d.resetRoi();
    RppStatus st = g_case.adapter->run(ctx, srcs, dsts, g_case.handle);
#if RPP_BENCH_HIP
    if (ctx.isHip) {
        // The rppt_* launch is async; a kernel fault surfaces here. Capture it
        // so a single failure doesn't spam thousands of launches.
        hipError_t he = hipStreamSynchronize(g_case.hipStream);
        if (he != hipSuccess)
            return std::string("HIP kernel fault: ") + hipGetErrorString(he);
    }
#endif
    if (st != RPP_SUCCESS)
        return "rppt call returned status " + std::to_string(st);
    return {};
}

/**
 * @brief The timed function body for one fully-resolved combo.
 * @param state Google Benchmark state driving the timing loop.
 * @param caseId Index of the registered case (keys the reused resources).
 * @param factory Factory that builds the op adapter.
 * @param ctx The fully-resolved sweep point.
 * @param warmupIters Untimed iterations run once per case before measuring.
 */
void run_case(benchmark::State &state, int caseId, const AdapterFactory &factory,
              const BenchContext &ctx, int warmupIters) {
    if (g_case.caseId != caseId)
        g_case.build(caseId, factory, ctx); // first invocation for this case
    if (g_case.failed) {
        state.SkipWithError(g_case.error);
        return;
    }

    std::vector<TensorBuffer> &srcs = g_case.srcs;
    std::vector<TensorBuffer> &dsts = g_case.dsts;

    // Untimed warmup, run once per case (before the calibration ramp's first
    // measured invocation) to reach steady clocks / paged-in buffers.
    if (!g_case.warmed) {
        for (int i = 0; i < warmupIters; ++i) {
            std::string err = run_one(ctx, srcs, dsts);
            if (!err.empty()) {
                state.SkipWithError(err);
                return;
            }
        }
        g_case.warmed = true;
    }

    for (auto _ : state) {
        (void)_; // benchmark loop sentinel; body work is what's timed
        std::string err = run_one(ctx, srcs, dsts);
        if (!err.empty()) {
            state.SkipWithError(err);
            break;
        }
    }

    // Rate counters (kIsIterationInvariantRate multiplies by iterations / time).
    const double pixels = static_cast<double>(ctx.batch) * ctx.width * ctx.height;
    state.counters["images_per_sec"] =
        benchmark::Counter(ctx.batch, benchmark::Counter::kIsIterationInvariantRate);
    state.counters["pixels_per_sec"] =
        benchmark::Counter(pixels, benchmark::Counter::kIsIterationInvariantRate);
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(srcs[0].dataBytes));
}

} // namespace

void release_active_resources() {
    g_case.release();
}

int register_benchmarks(const BenchConfig &cfg, std::vector<std::string> *names) {
    auto &registry = OpRegistry::instance();
    int registered = 0;

    for (const auto &op : cfg.ops) {
        if (!registry.has(op.name)) {
            std::fprintf(stderr, "warning: no adapter for op '%s' - skipping\n", op.name.c_str());
            continue;
        }
        AdapterFactory factory = registry.find(op.name);
        auto probe = factory(); // query supported dtypes/layouts
        const auto dtypesOk = probe->supportedDtypes();
        const auto layoutsOk = probe->supportedLayouts();
        const auto backendsOk = probe->supportedBackends();

        // dst sizes: use configured list, else a single "same as source" sentinel.
        std::vector<std::pair<int, int>> dsts = op.dstSizes;
        const bool sameDst = dsts.empty();

        for (const auto &backend : op.matrix.backends) {
            RppBackend rppBackend = parse_backend(backend);
            if (!supports(backendsOk, rppBackend)) {
                std::fprintf(stderr, "note: op '%s' has no %s implementation - skipping\n",
                             op.name.c_str(), backend.c_str());
                continue;
            }
#if !RPP_BENCH_HIP
            if (rppBackend == RppBackend::RPP_HIP_BACKEND) {
                std::fprintf(stderr,
                             "note: HIP backend requested for '%s' but rpp_bench was "
                             "built HOST-only - skipping\n",
                             op.name.c_str());
                continue;
            }
#endif
            for (RpptDataType dt : op.matrix.dtypes) {
                if (!supports(dtypesOk, dt))
                    continue;
                for (Layout layout : op.matrix.layouts) {
                    if (!supports(layoutsOk, layout))
                        continue;
                    for (int batch : op.matrix.batchSizes) {
                        for (const auto &sz : op.matrix.imageSizes) {
                            const auto dstList =
                                sameDst ? std::vector<std::pair<int, int>>{sz} : dsts;
                            for (const auto &dstSz : dstList) {
                                for (const auto &paramSet : op.paramSets) {
                                    BenchContext ctx;
                                    ctx.opName = op.name;
                                    ctx.backendName = backend;
                                    ctx.backend = rppBackend;
                                    ctx.dtype = dt;
                                    ctx.layout = layout;
                                    ctx.batch = batch;
                                    ctx.width = sz.first;
                                    ctx.height = sz.second;
                                    ctx.dstWidth = dstSz.first;
                                    ctx.dstHeight = dstSz.second;
                                    ctx.isHip = (rppBackend == RppBackend::RPP_HIP_BACKEND);
                                    ctx.params = &paramSet;

                                    const int caseId = registered;
                                    std::string name = encode_name(ctx);
                                    if (names)
                                        names->push_back(name);
                                    const int warmupIters = cfg.warmupIterations;
                                    auto *b = benchmark::RegisterBenchmark(
                                        name,
                                        [caseId, factory, ctx, warmupIters](benchmark::State &st) {
                                            run_case(st, caseId, factory, ctx, warmupIters);
                                        });
                                    b->UseRealTime();
                                    // Iterations and MinTime are mutually exclusive in
                                    // Google Benchmark; the config parser rejects setting
                                    // both, so at most one branch fires here.
                                    if (cfg.iterations > 0)
                                        b->Iterations(cfg.iterations);
                                    else if (cfg.minTimeSec > 0.0)
                                        b->MinTime(cfg.minTimeSec);
                                    if (cfg.repetitions > 0)
                                        b->Repetitions(cfg.repetitions);
                                    ++registered;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    return registered;
}

} // namespace rppbench
