// ============================================================================
// bench_progress_simple.cpp - the plain --progress=simple reporter.
// ============================================================================
#include "cli/progress/bench_progress_simple.hpp"

#include <cstdio>

namespace rppbench {

bool SimpleProgressReporter::ReportContext(const Context &) {
    std::fprintf(stderr, "Running %d benchmark case(s)...\n", total());
    return true;
}

void SimpleProgressReporter::ReportRuns(const std::vector<Run> &runs) {
    Sample s = sample(runs);
    if (s.name.empty())
        return;
    draw_plain_line(done(), total(), s.name, s.val, s.unit);
}

void SimpleProgressReporter::Finalize() {
    std::fprintf(stderr, "\ndone: %d benchmark case(s).\n", done());
}

} // namespace rppbench
