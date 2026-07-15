/**
 * @file bench_progress.hpp
 * @brief Shared base for the --progress live reporters.
 *
 * --progress swaps Google Benchmark's per-case result table for a self-updating
 * progress display on stderr. The two display styles live in their own files and
 * share this base, which owns the case-counting / run-sampling logic Google
 * Benchmark's callback shape forces on us:
 *   - progress/bench_progress_simple.hpp    one plain self-overwriting line.
 *   - progress/bench_progress_dashboard.hpp a bordered TTY dashboard.
 * The style is chosen at the command line via --progress[=simple|fancy]; see
 * cli/bench_cli.hpp (extract_progress) and src/bench_main.cpp.
 */
#ifndef RPP_BENCH_PROGRESS_HPP
#define RPP_BENCH_PROGRESS_HPP

#include <benchmark/benchmark.h>

#include <string>
#include <vector>

namespace rppbench {

/**
 * @brief How many of @p names Google Benchmark will actually run under @p filter.
 *
 * Mirrors its matcher: POSIX-extended regex, substring search, and a leading '-'
 * inverts to an exclusion filter. Used so --progress counts filtered cases, not
 * all registered ones.
 * @param names All registered benchmark names.
 * @param filter The active --benchmark_filter value.
 * @return The matching count; falls back to the full count on an unset/invalid filter.
 */
int count_matching(const std::vector<std::string> &names, std::string filter);

// Shared base for the progress reporters: owns the total/done counters and the
// per-call sampling that both styles need. Abstract - the display lives in the
// derived classes' ReportRuns.
//
// With repetitions > 1, Google Benchmark calls ReportRuns twice per case (once
// for the iteration runs, once for the aggregates) that share the same run_name,
// so a case is counted once, on first sighting, rather than by counting calls.
class ProgressReporter : public benchmark::BenchmarkReporter {
public:
    explicit ProgressReporter(int total) : total_(total) {}

protected:
    // One digested ReportRuns call: whether this is a case not seen before, the
    // trimmed encoded name, and the representative run's time (native + in ms).
    struct Sample {
        bool isNew = false;
        std::string name;
        double val = 0.0;      // GetAdjustedRealTime() in `unit`
        double ms = 0.0;       // same, normalized to milliseconds
        const char *unit = ""; // display unit string for `val`
    };

    /**
     * @brief Digest one ReportRuns callback.
     *
     * Updates the done counter on a newly-seen case and pulls the representative
     * run's name/time.
     * @param runs The runs passed to ReportRuns.
     * @return The digested sample; an empty (isNew=false) Sample for empty runs.
     */
    Sample sample(const std::vector<Run> &runs);

    int total() const { return total_; }
    int done() const { return done_; }

private:
    int total_;
    int done_ = 0;
    std::string lastCase_; // run_name of the case counted last (dedupes repeat calls)
};

// --- shared display helpers (used by both reporter styles) -------------------

/**
 * @brief Drop the "op:" prefix from a trimmed encoded name for display.
 * @param name The encoded benchmark name.
 * @return The name without a leading "op:".
 */
std::string strip_op_prefix(std::string name);

/**
 * @brief Draw the plain self-overwriting progress line.
 *
 * An ASCII bar, done/total, percent, the current case, and its time. Used by
 * SimpleProgressReporter directly and by the dashboard's non-TTY fallback.
 * @param done Cases completed so far.
 * @param total Total cases to run.
 * @param name The current case's encoded name.
 * @param val The case's time in @p unit.
 * @param unit Display unit string for @p val.
 */
void draw_plain_line(int done, int total, const std::string &name, double val, const char *unit);

} // namespace rppbench

#endif // RPP_BENCH_PROGRESS_HPP
