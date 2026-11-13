/**
 * @file bench_metadata.hpp
 * @brief Run metadata embedded in every result file's "context" block.
 *
 * Google Benchmark already records the host, CPU count/caches, load average and
 * date. On their own, those can't tell two result files apart when they came
 * from different RPP builds, ROCm versions, GPUs or configs, so the numbers
 * can't be compared or tracked over time. This adds the missing facts through
 * benchmark::AddCustomContext so each JSON result describes itself: which
 * rpp-bench and RPP produced it, on which GPU/driver, from which config, and
 * with which command line and environment. The config content itself is
 * appended to JSON result files after the run (embed_config_in_results).
 */
#ifndef RPP_BENCH_METADATA_HPP
#define RPP_BENCH_METADATA_HPP

#include <string>
#include <vector>

namespace rppbench {

/**
 * @brief Version of the metadata keys and the benchmark-name format.
 *
 * Bump when either changes incompatibly, so tooling that ingests result files
 * can tell old and new layouts apart.
 */
inline constexpr const char *kResultSchemaVersion = "1";

/**
 * @brief Record the run metadata as Google Benchmark custom context.
 *
 * Must be called before benchmark::RunSpecifiedBenchmarks(). Every probe is
 * best-effort: a fact that can't be determined is recorded as "unknown" (or
 * skipped, for unset environment variables) rather than failing the run.
 * @param configPath Path of the loaded sweep config (recorded with a content hash).
 * @param commandLine The original, unmodified argv, joined into one string.
 */
void add_run_metadata(const std::string &configPath, const std::vector<std::string> &commandLine);

/**
 * @brief Add the sweep config, as structured JSON, to a finished JSON result file.
 *
 * Stored under a top-level "rpp_bench_config" key (Google Benchmark's reader
 * and scripts/ ignore unknown keys), so a result stays reproducible after the
 * config file is later edited. Call after the run, once the file is closed.
 * @param resultsPath The JSON result file written by Google Benchmark.
 * @param configPath The sweep config to embed.
 * @return Whether the file was updated (false if either file is unreadable or
 *         the results aren't a JSON object, e.g. an empty file from a run
 *         whose filter matched nothing).
 */
bool embed_config_in_results(const std::string &resultsPath, const std::string &configPath);

} // namespace rppbench

#endif // RPP_BENCH_METADATA_HPP
