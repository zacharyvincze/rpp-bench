// ============================================================================
// bench_main.cpp - entry point for the RPP benchmark harness.
//
// We provide our own main (rather than benchmark_main) so the run config can be
// loaded and benchmarks registered *before* RunSpecifiedBenchmarks().
//
// Usage:
//   ./rpp_bench --config=../config/example.json [google-benchmark flags...]
//   ./rpp_bench --list-ops
//   ./rpp_bench --config=cfg.json --benchmark_out=results.json --benchmark_out_format=json
//   ./rpp_bench --config=cfg.json --progress    # compact live progress line
//
// All standard Google Benchmark flags (--benchmark_filter, --benchmark_out,
// --benchmark_repetitions, ...) are forwarded unchanged. --progress swaps the
// per-benchmark result table for a single self-updating progress line on stderr;
// --benchmark_out still writes the full results.
// ============================================================================
#include <benchmark/benchmark.h>

#include "bench_config.hpp"
#include "bench_registry.hpp"
#include "bench_runner.hpp"

#include <array>
#include <cstdio>
#include <cstring>
#include <regex>
#include <string>
#include <vector>

namespace {

// Pull "--config=PATH" (or "--config PATH") out of argv so Google Benchmark's
// flag parser never sees it. Mutates argc/argv in place.
std::string extract_config(int &argc, char **argv) {
    const char *kFlag = "--config";
    std::string path;
    std::vector<char *> kept;
    kept.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], "--config=", 9) == 0) {
            path = argv[i] + 9;
        } else if (std::strcmp(argv[i], kFlag) == 0 && i + 1 < argc) {
            path = argv[++i];
        } else {
            kept.push_back(argv[i]);
        }
    }
    for (size_t i = 0; i < kept.size(); ++i)
        argv[i] = kept[i];
    argc = static_cast<int>(kept.size());
    return path;
}

bool has_flag(int argc, char **argv, const char *flag) {
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], flag) == 0)
            return true;
    return false;
}

// Remove a valueless flag from argv (so benchmark::Initialize won't reject it).
// Returns whether it was present.
bool strip_flag(int &argc, char **argv, const char *flag) {
    bool found = false;
    std::vector<char *> kept{argv[0]};
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], flag) == 0) {
            found = true;
            continue;
        }
        kept.push_back(argv[i]);
    }
    for (size_t i = 0; i < kept.size(); ++i)
        argv[i] = kept[i];
    argc = static_cast<int>(kept.size());
    return found;
}

// How many of `names` Google Benchmark will actually run under `filter`.
// Mirrors its matcher: POSIX-extended regex, substring search, and a leading '-'
// inverts to an exclusion filter. Used so --progress counts filtered cases, not
// all registered ones. Falls back to the full count on an unset/invalid filter.
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

// Compact display reporter: one self-overwriting progress line on stderr instead
// of a full result block per benchmark. Full results still go to --benchmark_out.
// ReportRuns is called once per benchmark case (with its repetition + aggregate
// runs), so counting calls tracks case-level progress.
class ProgressReporter : public benchmark::BenchmarkReporter {
public:
    explicit ProgressReporter(int total) : total_(total) {}

    bool ReportContext(const Context &) override {
        std::fprintf(stderr, "Running %d benchmark case(s)...\n", total_);
        return true;
    }

    void ReportRuns(const std::vector<Run> &runs) override {
        ++done_;
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

    void Finalize() override { std::fprintf(stderr, "\ndone: %d benchmark case(s).\n", done_); }

private:
    int total_;
    int done_ = 0;
};

} // namespace

int main(int argc, char **argv) {
    if (has_flag(argc, argv, "--list-ops")) {
        std::printf("Registered benchmark adapters:\n");
        for (const auto &n : rppbench::OpRegistry::instance().names())
            std::printf("  %s\n", n.c_str());
        return 0;
    }

    const bool progress = strip_flag(argc, argv, "--progress");

    const std::string configPath = extract_config(argc, argv);
    if (configPath.empty()) {
        std::fprintf(stderr, "error: no config given. Use --config=<path.json> "
                             "(or --list-ops).\n");
        return 2;
    }

    rppbench::BenchConfig cfg;
    try {
        cfg = rppbench::load_config(configPath);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "error loading config: %s\n", e.what());
        return 2;
    }

    int n = 0;
    std::vector<std::string> names;
    try {
        n = rppbench::register_benchmarks(cfg, &names);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "error registering benchmarks: %s\n", e.what());
        return 2;
    }
    std::fprintf(stderr, "rpp_bench: registered %d benchmark(s) from %s\n", n, configPath.c_str());
    if (n == 0)
        return 1;

    benchmark::Initialize(&argc, argv);
    if (benchmark::ReportUnrecognizedArguments(argc, argv))
        return 1;
    if (progress) {
        // Count only the cases the active filter will actually run, so the
        // progress denominator is correct under --benchmark_filter.
        int toRun = count_matching(names, benchmark::GetBenchmarkFilter());
        // Custom display reporter; --benchmark_out (if any) still writes full JSON.
        ProgressReporter reporter(toRun);
        benchmark::RunSpecifiedBenchmarks(&reporter);
    } else {
        benchmark::RunSpecifiedBenchmarks();
    }
    rppbench::release_active_resources(); // free the last case's handle/buffers
    benchmark::Shutdown();
    return 0;
}
