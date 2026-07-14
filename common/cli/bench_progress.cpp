// ============================================================================
// bench_progress.cpp - compact live-progress reporting for --progress.
// ============================================================================
#include "cli/bench_progress.hpp"

#include <array>
#include <cstdio>
#include <regex>

namespace rppbench {

int count_matching(const std::vector<std::string> &names, std::string filter) {
    if (filter.empty() || filter == "all" || filter == ".")
        return static_cast<int>(names.size());
    bool negate = false;
    if (filter[0] == '-') {
        negate = true;
        filter.erase(0, 1);
    }
    std::regex re;
    try {
        re = std::regex(filter, std::regex_constants::extended);
    } catch (const std::regex_error &) {
        return static_cast<int>(names.size()); // let benchmark report the error
    }
    int count = 0;
    for (const auto &n : names)
        if (std::regex_search(n, re) != negate)
            ++count;
    return count;
}

bool ProgressReporter::ReportContext(const Context &) {
    std::fprintf(stderr, "Running %d benchmark case(s)...\n", total_);
    return true;
}

void ProgressReporter::ReportRuns(const std::vector<Run> &runs) {
    if (runs.empty())
        return;
    // A case with repetitions is reported in two calls (iteration runs, then
    // aggregates) that share the same run_name; count it once, on first sighting.
    const std::string caseName = runs.front().run_name.str();
    if (caseName != lastCase_) {
        ++done_;
        lastCase_ = caseName;
    }
    const Run *pick = nullptr;
    for (const auto &r : runs)
        if (r.run_type == Run::RT_Aggregate && r.aggregate_name == "mean") {
            pick = &r;
            break;
        }
    if (!pick && !runs.empty())
        pick = &runs.front();

    std::string name = pick ? pick->benchmark_name() : std::string();
    auto pos = name.find("/min_time"); // trim boilerplate suffix
    if (pos != std::string::npos)
        name.resize(pos);
    if (name.rfind("op:", 0) == 0)
        name.erase(0, 3);

    std::array<char, 24> tbuf{};
    if (pick)
        std::snprintf(tbuf.data(), tbuf.size(), "%.3g %s", pick->GetAdjustedRealTime(),
                      benchmark::GetTimeUnitString(pick->time_unit));

    constexpr int W = 22;
    double frac = total_ > 0 ? static_cast<double>(done_) / total_ : 0.0;
    int fill = static_cast<int>(frac * W);
    std::array<char, W + 1> bar{};
    for (int i = 0; i < W; ++i)
        bar[i] = i < fill ? '=' : ' ';
    if (fill > 0 && fill < W)
        bar[fill - 1] = '>';
    bar[W] = '\0';

    std::fprintf(stderr, "\r[%s] %4d/%-4d %5.1f%%  %-44.44s %11s", bar.data(), done_, total_,
                 frac * 100.0, name.c_str(), tbuf.data());
    std::fflush(stderr);
}

void ProgressReporter::Finalize() {
    std::fprintf(stderr, "\ndone: %d benchmark case(s).\n", done_);
}

} // namespace rppbench
