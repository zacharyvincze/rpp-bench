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

// Detailed usage. Covers this harness's own flags plus a quick reference to the
// Google Benchmark flags that are forwarded unchanged, so users don't have to
// cross-check `--help` from a stock benchmark binary.
void print_help(const char *prog) {
    std::printf(
        "rpp_bench - config-driven micro-benchmarks for RPP operators.\n"
        "\n"
        "USAGE:\n"
        "  %s --config=<path.json> [harness flags] [google-benchmark flags]\n"
        "  %s --list-ops\n"
        "  %s --help | -h\n"
        "\n"
        "The benchmark matrix (ops x backends x dtypes x layouts x batch x image\n"
        "sizes x per-op params) is described in the JSON config and expanded at\n"
        "runtime - exploring new cases means editing a config, not recompiling.\n"
        "\n"
        "HARNESS FLAGS:\n"
        "  --config=<path>        Sweep config to load (required to run). Two ship\n"
        "                         in config/: smoke.json (small/fast) and\n"
        "                         example.json (fuller sweep).\n"
        "  --list-ops             List the op adapters compiled in, then exit.\n"
        "  --progress             Replace the per-benchmark result table with a\n"
        "                         single self-updating progress line on stderr.\n"
        "                         Honors the active --benchmark_filter; combine with\n"
        "                         --benchmark_out to still capture full results.\n"
        "  --help, -h             Show this help, then exit.\n"
        "\n"
        "The config sets defaults for per-benchmark min time and repetitions\n"
        "(\"min_time_sec\", \"repetitions\"); the Google Benchmark flags below override\n"
        "them at run time. See README.md for the full config format.\n"
        "\n"
        "GOOGLE BENCHMARK FLAGS (forwarded unchanged - quick reference):\n"
        "  --benchmark_filter=<regex>            Run only cases whose name matches\n"
        "                                        (substring). Prefix '-' to exclude.\n"
        "  --benchmark_list_tests={true|false}   Print matching case names, don't run.\n"
        "  --benchmark_min_time=<n>x|<n>s        Iterations (e.g. 50x) or wall time\n"
        "                                        (e.g. 0.2s) per case.\n"
        "  --benchmark_repetitions=<n>           Repeat each case n times\n"
        "                                        (-> mean/median/stddev).\n"
        "  --benchmark_report_aggregates_only={true|false}\n"
        "                                        Emit only aggregates, not each rep.\n"
        "  --benchmark_out=<file>                Write results to a file.\n"
        "  --benchmark_out_format={json|csv|console}   Format for --benchmark_out.\n"
        "  --benchmark_format={json|csv|console} Format for stdout.\n"
        "  --benchmark_time_unit={ns|us|ms|s}    Displayed time unit.\n"
        "  --benchmark_color={auto|true|false}   Colorize console output.\n"
        "  --v=<level>                           Google Benchmark verbosity.\n"
        "  (--help on a stock benchmark binary lists the complete set.)\n"
        "\n"
        "BENCHMARK NAME / FILTERING:\n"
        "  Names encode the combo in a fixed field order, e.g.\n"
        "    op:resize/backend:HIP/dtype:F32/layout:PKD3/batch:8/size:1920x1080/\\\n"
        "      dst:960x540/params:interpolation=BICUBIC\n"
        "  so --benchmark_filter can target any field (op:, backend:, dtype:,\n"
        "  layout:, batch:, size:, dst:, params:). Filtering also trims the\n"
        "  JSON/CSV output to the selected cases.\n"
        "\n"
        "EXAMPLES:\n"
        "  %s --config=config/smoke.json\n"
        "  %s --config=config/example.json --progress \\\n"
        "     --benchmark_out=results.json --benchmark_out_format=json\n"
        "  %s --config=config/example.json --benchmark_filter='op:resize.*dtype:F32'\n"
        "  %s --config=config/example.json --benchmark_filter='-backend:HOST'\n"
        "  %s --config=config/example.json --benchmark_list_tests\n"
        "\n"
        "Post-process results with the scripts in scripts/ (json2csv.py,\n"
        "plot_results.py). See README.md for details.\n",
        prog, prog, prog, prog, prog, prog, prog, prog);
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
    if (has_flag(argc, argv, "--help") || has_flag(argc, argv, "-h")) {
        print_help(argv[0]);
        return 0;
    }

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
