// ============================================================================
// bench_cli.cpp - command-line argument handling for the benchmark harness.
// ============================================================================
#include "cli/bench_cli.hpp"

#include <cstdio>
#include <cstring>
#include <vector>

namespace rppbench {

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

bool has_flag_named(int argc, char **argv, const char *name) {
    const size_t n = std::strlen(name);
    for (int i = 1; i < argc; ++i) {
        if (std::strncmp(argv[i], name, n) == 0 && (argv[i][n] == '\0' || argv[i][n] == '='))
            return true;
    }
    return false;
}

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

// Detailed usage. Covers this harness's own flags plus a quick reference to the
// Google Benchmark flags that are forwarded unchanged, so users don't have to
// cross-check `--help` from a stock benchmark binary.
void print_help(const char *prog) {
    std::printf("rpp_bench - config-driven micro-benchmarks for RPP operators.\n"
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
                "The config sets defaults for per-benchmark timing: \"min_time_sec\" (wall\n"
                "time) or \"iterations\" (a fixed count - the two are mutually exclusive), and\n"
                "\"repetitions\". The matching Google Benchmark flags below override them at run\n"
                "time (--benchmark_min_time overrides both timing fields). \"warmup_iterations\"\n"
                "runs that many untimed iterations once per case first. See README.md for\n"
                "the full config format.\n"
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

} // namespace rppbench
