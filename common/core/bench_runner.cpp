/**
 * @file bench_runner.cpp
 * @brief Matrix expansion + the timed body, driven through the Library interface.
 *
 * This is the library-neutral heart of the harness: it resolves each op's
 * Library from the registry, filters the matrix by the library's neutral
 * capability report, registers one Google Benchmark per surviving combo, and
 * drives each case through BenchCase. It contains no library or device headers -
 * all of that lives behind Library::makeCase() / BenchCase.
 */
#include "core/bench_runner.hpp"

#include <benchmark/benchmark.h>

#include "core/bench_library.hpp"
#include "core/bench_name.hpp"
#include "core/bench_point.hpp"

#include <algorithm>
#include <cstdio>
#include <memory>
#include <string>

namespace rppbench {

namespace {

template <typename T>
bool supports(const std::vector<T> &v, T t) {
    return std::find(v.begin(), v.end(), t) != v.end();
}

// The single live case's neutral wrapper. Google Benchmark determines a case's
// iteration count by re-invoking its function several times (calibration ramp),
// then again per repetition; building the case each time would pay the full
// allocation + setup cost repeatedly. Instead we build once, keyed by caseId,
// and reuse. Benchmarks run sequentially, so building a new case releases the
// previous one - at most one case's device memory is resident. The library
// resources themselves live inside `impl` (a library-specific BenchCase).
struct ActiveCase {
    int caseId = -1;
    bool warmed = false;
    std::string error; // sticky setup error; skip without rebuilding every re-invoke
    BenchPoint point;
    std::unique_ptr<BenchCase> impl;

    void release() {
        impl.reset(); // BenchCase destructor frees buffers/handle/stream + op teardown
        warmed = false;
        error.clear();
        caseId = -1;
    }
};

// At most one case resident (see above). Assumes single-threaded benchmarks,
// which is how they are registered.
ActiveCase g_case;

/**
 * @brief The timed function body for one fully-resolved combo.
 * @param state Google Benchmark state driving the timing loop.
 * @param caseId Index of the registered case (keys the reused resources).
 * @param lib The library that owns this op.
 * @param point The fully-resolved sweep point.
 * @param warmupIters Untimed iterations run once per case before measuring.
 */
void run_case(benchmark::State &state, int caseId, const Library *lib, const BenchPoint &point,
              int warmupIters) {
    if (g_case.caseId != caseId) {
        // First invocation for this case: release the previous one and build.
        g_case.release();
        g_case.caseId = caseId;
        g_case.point = point;
        g_case.impl = lib->makeCase(point);
        g_case.error = g_case.impl ? g_case.impl->setup() : "library returned no case";
    }
    if (!g_case.error.empty()) {
        state.SkipWithError(g_case.error);
        return;
    }

    // Untimed warmup, run once per case (before the calibration ramp's first
    // measured invocation) to reach steady clocks / paged-in buffers.
    if (!g_case.warmed) {
        for (int i = 0; i < warmupIters; ++i) {
            std::string err = g_case.impl->runOnce();
            if (!err.empty()) {
                state.SkipWithError(err);
                return;
            }
        }
        g_case.warmed = true;
    }

    for (auto _ : state) {
        (void)_; // benchmark loop sentinel; body work is what's timed
        std::string err = g_case.impl->runOnce();
        if (!err.empty()) {
            state.SkipWithError(err);
            break;
        }
    }

    // Rate counters (kIsIterationInvariantRate multiplies by iterations / time).
    const double pixels = static_cast<double>(point.batch) * point.width * point.height;
    state.counters["images_per_sec"] =
        benchmark::Counter(point.batch, benchmark::Counter::kIsIterationInvariantRate);
    state.counters["pixels_per_sec"] =
        benchmark::Counter(pixels, benchmark::Counter::kIsIterationInvariantRate);
    state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                            static_cast<int64_t>(g_case.impl->srcDataBytes()));
}

} // namespace

void release_active_resources() {
    g_case.release();
}

int register_benchmarks(const BenchConfig &cfg, std::vector<std::string> *names) {
    auto &registry = LibraryRegistry::instance();
    int registered = 0;

    for (const auto &op : cfg.ops) {
        Library *lib = registry.find(op.library);
        if (!lib) {
            std::fprintf(stderr, "warning: no library '%s' registered for op '%s' - skipping\n",
                         op.library.c_str(), op.name.c_str());
            continue;
        }
        if (!lib->hasOp(op.name)) {
            std::fprintf(stderr, "warning: library '%s' has no op '%s' - skipping\n",
                         op.library.c_str(), op.name.c_str());
            continue;
        }
        const OpCapabilities caps = lib->capabilities(op.name);
        const std::vector<Backend> avail = lib->availableBackends();

        // dst sizes: use configured list, else a single "same as source" sentinel.
        std::vector<std::pair<int, int>> dsts = op.dstSizes;
        const bool sameDst = dsts.empty();

        for (const auto &backendStr : op.matrix.backends) {
            Backend backend = parse_backend(backendStr);
            if (!supports(avail, backend)) {
                std::fprintf(stderr,
                             "note: library '%s' was not built with the %s backend - "
                             "skipping op '%s'\n",
                             op.library.c_str(), backendStr.c_str(), op.name.c_str());
                continue;
            }
            if (!supports(caps.backends, backend)) {
                std::fprintf(stderr, "note: op '%s' (%s) has no %s implementation - skipping\n",
                             op.name.c_str(), op.library.c_str(), backendStr.c_str());
                continue;
            }
            for (DataType dt : op.matrix.dtypes) {
                if (!supports(caps.dtypes, dt))
                    continue;
                for (Layout layout : op.matrix.layouts) {
                    if (!supports(caps.layouts, layout))
                        continue;
                    for (int batch : op.matrix.batchSizes) {
                        for (const auto &sz : op.matrix.imageSizes) {
                            const auto dstList =
                                sameDst ? std::vector<std::pair<int, int>>{sz} : dsts;
                            for (const auto &dstSz : dstList) {
                                for (const auto &paramSet : op.paramSets) {
                                    BenchPoint point;
                                    point.opName = op.name;
                                    point.library = op.library;
                                    point.backend = backend;
                                    point.dtype = dt;
                                    point.layout = layout;
                                    point.batch = batch;
                                    point.width = sz.first;
                                    point.height = sz.second;
                                    point.dstWidth = dstSz.first;
                                    point.dstHeight = dstSz.second;
                                    point.params = &paramSet;

                                    const int caseId = registered;
                                    std::string name = encode_name(point);
                                    if (names)
                                        names->push_back(name);
                                    const int warmupIters = cfg.warmupIterations;
                                    auto *b = benchmark::RegisterBenchmark(
                                        name,
                                        [caseId, lib, point, warmupIters](benchmark::State &st) {
                                            run_case(st, caseId, lib, point, warmupIters);
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
