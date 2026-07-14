#include "bench_runner.hpp"

#include <benchmark/benchmark.h>

#include "bench_registry.hpp"

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

// A JSON scalar as a bare string ("BILINEAR", "5", "1.75") - no quotes for
// strings, compact repr for numbers/bools.
std::string json_scalar(const nlohmann::json &v) {
    return v.is_string() ? v.get<std::string>() : v.dump();
}

// Compact, deterministic "k1=v1,k2=v2" encoding of a param set (empty if none).
std::string encode_params(const nlohmann::json &p) {
    if (!p.is_object() || p.empty())
        return "";
    std::vector<std::string> kv;
    for (auto it = p.begin(); it != p.end(); ++it)
        kv.push_back(it.key() + "=" + json_scalar(it.value()));
    std::sort(kv.begin(), kv.end());
    std::string s;
    for (size_t i = 0; i < kv.size(); ++i) {
        if (i)
            s += ",";
        s += kv[i];
    }
    return s;
}

// Parseable, filter-friendly benchmark name. json2csv.py splits this back out.
std::string encode_name(const BenchContext &c) {
    std::string s = "op:" + c.opName + "/backend:" + c.backendName +
                    "/dtype:" + dtype_name(c.dtype) + "/layout:" + layout_name(c.layout) +
                    "/batch:" + std::to_string(c.batch) + "/size:" + std::to_string(c.width) + "x" +
                    std::to_string(c.height);
    if (c.dstWidth != c.width || c.dstHeight != c.height)
        s += "/dst:" + std::to_string(c.dstWidth) + "x" + std::to_string(c.dstHeight);
    // Encode params so swept sets get unique names and appear in the output.
    if (c.params) {
        std::string ps = encode_params(*c.params);
        if (!ps.empty())
            s += "/params:" + ps;
    }
    return s;
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
    std::string error;

    BenchContext ctx;
    std::unique_ptr<OpAdapter> adapter;
    TensorBuffer src, dst;
    rppHandle_t handle = nullptr;
#if RPP_BENCH_HIP
    hipStream_t hipStream = nullptr;
#endif

    void release() {
        if (adapter && ready)
            adapter->teardown();
        adapter.reset();
        src.free();
        dst.free();
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
        ready = failed = false;
        caseId = -1;
    }

    void build(int id, const AdapterFactory &factory, const BenchContext &c) {
        release();
        caseId = id;
        ctx = c;
        adapter = factory();
        src.init(ctx.backend, ctx.dtype, ctx.layout, ctx.batch, ctx.width, ctx.height,
                 adapter->srcOffsetBytes(ctx), adapter->srcAdditionalStride(ctx));
        dst.init(ctx.backend, ctx.dtype, ctx.layout, ctx.batch, ctx.dstWidth, ctx.dstHeight);
        src.fill();

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
        adapter->setup(ctx, src, dst);
        ready = true;
    }
};

// The single live case. Benchmarks run sequentially, so building a new case
// releases the previous one - at most one case's GPU memory is resident.
// (Assumes single-threaded benchmarks, which is how they are registered.)
CaseResources g_case;

// The timed function body for one fully-resolved combo.
void run_case(benchmark::State &state, int caseId, const AdapterFactory &factory,
              const BenchContext &ctx) {
    if (g_case.caseId != caseId)
        g_case.build(caseId, factory, ctx); // first invocation for this case
    if (g_case.failed) {
        state.SkipWithError(g_case.error);
        return;
    }

    TensorBuffer &src = g_case.src;
    TensorBuffer &dst = g_case.dst;

    for (auto _ : state) {
        (void)_; // benchmark loop sentinel; body work is what's timed
        // Some ops convert the ROI in place; refresh it each call so repeated
        // invocations don't accumulate and drive indices out of bounds.
        src.resetRoi();
        dst.resetRoi();
        RppStatus st = g_case.adapter->run(ctx, src, dst, g_case.handle);
#if RPP_BENCH_HIP
        if (ctx.isHip) {
            // The rppt_* launch is async; a kernel fault surfaces here. Capture it
            // so a single failure doesn't spam thousands of launches.
            hipError_t he = hipStreamSynchronize(g_case.hipStream);
            if (he != hipSuccess) {
                state.SkipWithError(std::string("HIP kernel fault: ") + hipGetErrorString(he));
                break;
            }
        }
#endif
        if (st != RPP_SUCCESS) {
            state.SkipWithError("rppt call returned status " + std::to_string(st));
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
                            static_cast<int64_t>(src.dataBytes));
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

        // dst sizes: use configured list, else a single "same as source" sentinel.
        std::vector<std::pair<int, int>> dsts = op.dstSizes;
        const bool sameDst = dsts.empty();

        for (const auto &backend : op.matrix.backends) {
            RppBackend rppBackend = parse_backend(backend);
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
                                    auto *b = benchmark::RegisterBenchmark(
                                        name, [caseId, factory, ctx](benchmark::State &st) {
                                            run_case(st, caseId, factory, ctx);
                                        });
                                    b->UseRealTime();
                                    if (cfg.minTimeSec > 0.0)
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
