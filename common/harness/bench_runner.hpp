/**
 * @file bench_runner.hpp
 * @brief Expand a BenchConfig into registered Google Benchmarks.
 */
#ifndef RPP_BENCH_RUNNER_HPP
#define RPP_BENCH_RUNNER_HPP

#include "config/bench_config.hpp"

#include <string>
#include <vector>

namespace rppbench {

/**
 * @brief Expand every op's matrix and register one Google Benchmark per combo.
 *
 * Expands (backend x dtype x layout x batch x size [x dstSize x paramSet])
 * combinations, skipping combos the adapter doesn't support or backends not
 * compiled in.
 * @param cfg The parsed sweep description.
 * @param names If non-null, each registered benchmark's name is appended to it
 *              (used to compute filtered counts for the progress display).
 * @return The number of benchmarks registered.
 */
int register_benchmarks(const BenchConfig &cfg, std::vector<std::string> *names = nullptr);

/**
 * @brief Free the resources (handle, stream, buffers) held by the most recently run case.
 *
 * Call once after RunSpecifiedBenchmarks(); harmless if nothing is held.
 * Per-case resources are built lazily on first use and reused across Google
 * Benchmark's calibration/repetition re-invocations, so this releases the final
 * case's allocations explicitly rather than relying on static destruction order.
 */
void release_active_resources();

} // namespace rppbench

#endif // RPP_BENCH_RUNNER_HPP
