// ============================================================================
// bench_progress_simple.hpp - the plain --progress=simple reporter.
//
// One self-overwriting ASCII progress line on stderr (the default style). Full
// results still go to --benchmark_out. The line rendering itself is the shared
// draw_plain_line() in progress/bench_progress.hpp.
// ============================================================================
#ifndef RPP_BENCH_PROGRESS_SIMPLE_HPP
#define RPP_BENCH_PROGRESS_SIMPLE_HPP

#include "cli/progress/bench_progress.hpp"

namespace rppbench {

class SimpleProgressReporter : public ProgressReporter {
public:
    using ProgressReporter::ProgressReporter;

    bool ReportContext(const Context &) override;
    void ReportRuns(const std::vector<Run> &runs) override;
    void Finalize() override;
};

} // namespace rppbench

#endif // RPP_BENCH_PROGRESS_SIMPLE_HPP
