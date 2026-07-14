// ============================================================================
// bench_runner.hpp - expand a BenchConfig into registered Google Benchmarks.
// ============================================================================
#ifndef RPP_BENCH_RUNNER_HPP
#define RPP_BENCH_RUNNER_HPP

#include "config/bench_config.hpp"

#include <string>
#include <vector>

namespace rppbench {

// Expand every op's matrix into (backend x dtype x layout x batch x size [x dstSize
// x paramSet]) combinations and register one Google Benchmark per combo. Skips
// combos the adapter doesn't support or backends not compiled in. Returns the
// number of benchmarks registered. If `names` is non-null, each registered
// benchmark's name is appended to it (used to compute filtered counts for the
// progress display).
int register_benchmarks(const BenchConfig &cfg, std::vector<std::string> *names = nullptr);

// Free the resources (handle, stream, buffers) held by the most recently run
// case. Call once after RunSpecifiedBenchmarks(); harmless if nothing is held.
// Per-case resources are built lazily on first use and reused across Google
// Benchmark's calibration/repetition re-invocations, so this releases the final
// case's allocations explicitly rather than relying on static destruction order.
void release_active_resources();

} // namespace rppbench

#endif // RPP_BENCH_RUNNER_HPP
