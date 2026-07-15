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
//   ./rpp_bench --help                          # detailed usage
//
// Argument handling lives in bench_cli, the --progress display in bench_progress,
// and the matrix expansion in bench_runner; this file just wires them together.
// ============================================================================
#include <benchmark/benchmark.h>

#include "cli/bench_cli.hpp"
#include "config/bench_config.hpp"
#include "harness/bench_registry.hpp"
#include "harness/bench_runner.hpp"
#include "cli/progress/bench_progress.hpp"
#include "cli/progress/bench_progress_dashboard.hpp"
#include "cli/progress/bench_progress_simple.hpp"

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

int main(int argc, char **argv) {
    using namespace rppbench;

    if (has_flag(argc, argv, "--help") || has_flag(argc, argv, "-h")) {
        print_help(argv[0]);
        return 0;
    }

    if (has_flag(argc, argv, "--list-ops")) {
        std::printf("Registered benchmark adapters:\n");
        for (const auto &n : OpRegistry::instance().names())
            std::printf("  %s\n", n.c_str());
        return 0;
    }

    const std::string progressMode = extract_progress(argc, argv);
    if (!progressMode.empty() && progressMode != "simple" && progressMode != "fancy") {
        std::fprintf(stderr, "error: unknown --progress mode '%s' (use 'simple' or 'fancy').\n",
                     progressMode.c_str());
        return 2;
    }

    const std::string configPath = extract_config(argc, argv);
    if (configPath.empty()) {
        std::fprintf(stderr, "error: no config given. Use --config=<path.json> "
                             "(or --list-ops).\n");
        return 2;
    }

    BenchConfig cfg;
    try {
        cfg = load_config(configPath);
    } catch (const std::exception &e) {
        std::fprintf(stderr, "error loading config: %s\n", e.what());
        return 2;
    }

    // Let a Google Benchmark timing flag on the command line win over the config
    // default. We register benchmarks before benchmark::Initialize() parses these,
    // so we clear the config value here rather than calling MinTime()/Iterations()/
    // Repetitions() - a per-benchmark setting would otherwise shadow the flag (for
    // repetitions) or leave the name's auto-appended suffix lying about what ran.
    if (has_flag_named(argc, argv, "--benchmark_min_time")) {
        cfg.minTimeSec = 0.0;
        cfg.iterations = 0;
    }
    if (has_flag_named(argc, argv, "--benchmark_repetitions"))
        cfg.repetitions = 0;

    int n = 0;
    std::vector<std::string> names;
    try {
        n = register_benchmarks(cfg, &names);
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
    if (!progressMode.empty()) {
        // Count only the cases the active filter will actually run, so the
        // progress denominator is correct under --benchmark_filter.
        int toRun = count_matching(names, benchmark::GetBenchmarkFilter());
        // Custom display reporter; --benchmark_out (if any) still writes full JSON.
        std::unique_ptr<benchmark::BenchmarkReporter> reporter;
        if (progressMode == "fancy")
            reporter = std::make_unique<DashboardProgressReporter>(toRun);
        else
            reporter = std::make_unique<SimpleProgressReporter>(toRun);
        benchmark::RunSpecifiedBenchmarks(reporter.get());
    } else {
        benchmark::RunSpecifiedBenchmarks();
    }
    release_active_resources(); // free the last case's handle/buffers
    benchmark::Shutdown();
    return 0;
}
