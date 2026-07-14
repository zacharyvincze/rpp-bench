// ============================================================================
// bench_progress.hpp - compact live-progress reporting for --progress.
//
// Swaps Google Benchmark's per-case result table for a single self-updating
// progress line on stderr, and counts how many cases the active filter selects
// so the progress denominator is right under --benchmark_filter.
// ============================================================================
#ifndef RPP_BENCH_PROGRESS_HPP
#define RPP_BENCH_PROGRESS_HPP

#include <benchmark/benchmark.h>

#include <string>
#include <vector>

namespace rppbench {

// How many of `names` Google Benchmark will actually run under `filter`.
// Mirrors its matcher: POSIX-extended regex, substring search, and a leading '-'
// inverts to an exclusion filter. Used so --progress counts filtered cases, not
// all registered ones. Falls back to the full count on an unset/invalid filter.
int count_matching(const std::vector<std::string> &names, std::string filter);

// Compact display reporter: one self-overwriting progress line on stderr instead
// of a full result block per benchmark. Full results still go to --benchmark_out.
// With repetitions > 1, Google Benchmark calls ReportRuns twice per case (once for
// the iteration runs, once for the aggregates), so progress is tracked by the
// underlying case name (`Run::run_name`, shared across a case's runs) rather than
// by counting calls.
class ProgressReporter : public benchmark::BenchmarkReporter {
public:
    explicit ProgressReporter(int total) : total_(total) {}

    bool ReportContext(const Context &) override;
    void ReportRuns(const std::vector<Run> &runs) override;
    void Finalize() override;

private:
    int total_;
    int done_ = 0;
    std::string lastCase_; // run_name of the case counted last (dedupes repeat calls)
};

} // namespace rppbench

#endif // RPP_BENCH_PROGRESS_HPP
