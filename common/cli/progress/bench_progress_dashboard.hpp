// ============================================================================
// bench_progress_dashboard.hpp - the fancy --progress=fancy reporter.
//
// A bordered TTY dashboard (gradient Unicode bar, elapsed / ETA / throughput,
// the current case, a sparkline of recent timings, and a running backend/dtype
// tally), re-queried against the terminal width each frame so it reflows on
// resize. Degrades to the shared plain line (progress/bench_progress.hpp) when
// stderr is not a TTY, so logs and pipes stay readable.
// ============================================================================
#ifndef RPP_BENCH_PROGRESS_DASHBOARD_HPP
#define RPP_BENCH_PROGRESS_DASHBOARD_HPP

#include "cli/progress/bench_progress.hpp"

#include <chrono>
#include <deque>
#include <map>
#include <string>

namespace rppbench {

class DashboardProgressReporter : public ProgressReporter {
public:
    explicit DashboardProgressReporter(int total);
    ~DashboardProgressReporter() override;

    bool ReportContext(const Context &) override;
    void ReportRuns(const std::vector<Run> &runs) override;
    void Finalize() override;

private:
    void draw(const std::string &caseName, double lastMs);

    using Clock = std::chrono::steady_clock;

    bool tty_ = false;
    bool cursorHidden_ = false;
    int linesDrawn_ = 0; // dashboard height already on screen (for cursor-up redraw)
    Clock::time_point start_;

    std::deque<double> recentMs_;          // sparkline window (normalized to ms)
    double minMs_ = 0.0, maxMs_ = 0.0;     // running spread for sparkline scaling
    std::map<std::string, int> byBackend_; // live tally, e.g. HOST / HIP
    std::map<std::string, int> byDtype_;   // live tally, e.g. u8 / f16 / f32
};

} // namespace rppbench

#endif // RPP_BENCH_PROGRESS_DASHBOARD_HPP
