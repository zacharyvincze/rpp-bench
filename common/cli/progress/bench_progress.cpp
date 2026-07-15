/**
 * @file bench_progress.cpp
 * @brief Shared base for the --progress live reporters.
 *
 * count_matching() mirrors Google Benchmark's filter so the progress denominator
 * is right. ProgressReporter::sample() digests each ReportRuns callback (case
 * de-duplication + picking the representative run) so both display styles share
 * it. draw_plain_line() is the plain readout used by SimpleProgressReporter and
 * by the dashboard's non-TTY fallback.
 */
#include "cli/progress/bench_progress.hpp"

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

namespace {

/**
 * @brief Normalize a value to milliseconds.
 *
 * So callers can compare across the per-case time units Google Benchmark may pick.
 * @param v The value in unit @p u.
 * @param u The time unit of @p v.
 * @return @p v expressed in milliseconds.
 */
double to_ms(double v, benchmark::TimeUnit u) {
    switch (u) {
    case benchmark::kNanosecond:
        return v / 1e6;
    case benchmark::kMicrosecond:
        return v / 1e3;
    case benchmark::kMillisecond:
        return v;
    case benchmark::kSecond:
        return v * 1e3;
    }
    return v;
}

} // namespace

ProgressReporter::Sample ProgressReporter::sample(const std::vector<Run> &runs) {
    Sample s;
    if (runs.empty())
        return s;

    // A case with repetitions is reported in two calls (iteration runs, then
    // aggregates) that share the same run_name; count it once, on first sighting.
    const std::string caseName = runs.front().run_name.str();
    s.isNew = caseName != lastCase_;
    if (s.isNew) {
        ++done_;
        lastCase_ = caseName;
    }

    const Run *pick = nullptr;
    for (const auto &r : runs)
        if (r.run_type == Run::RT_Aggregate && r.aggregate_name == "mean") {
            pick = &r;
            break;
        }
    if (!pick)
        pick = &runs.front();

    s.name = pick->benchmark_name();
    auto pos = s.name.find("/min_time"); // trim boilerplate suffix
    if (pos != std::string::npos)
        s.name.resize(pos);
    s.val = pick->GetAdjustedRealTime();
    s.unit = benchmark::GetTimeUnitString(pick->time_unit);
    s.ms = to_ms(s.val, pick->time_unit);
    return s;
}

std::string strip_op_prefix(std::string name) {
    if (name.rfind("op:", 0) == 0)
        name.erase(0, 3);
    return name;
}

void draw_plain_line(int done, int total, const std::string &name, double val, const char *unit) {
    std::string shown = strip_op_prefix(name);
    double frac = total > 0 ? static_cast<double>(done) / total : 0.0;
    constexpr int W = 22;
    int fill = static_cast<int>(frac * W);
    std::array<char, W + 1> bar{};
    for (int i = 0; i < W; ++i)
        bar[i] = i < fill ? '=' : ' ';
    if (fill > 0 && fill < W)
        bar[fill - 1] = '>';
    bar[W] = '\0';
    char tbuf[24];
    std::snprintf(tbuf, sizeof(tbuf), "%.3g %s", val, unit);
    std::fprintf(stderr, "\r[%s] %4d/%-4d %5.1f%%  %-44.44s %11s", bar.data(), done, total,
                 frac * 100.0, shown.c_str(), tbuf);
    std::fflush(stderr);
}

} // namespace rppbench
