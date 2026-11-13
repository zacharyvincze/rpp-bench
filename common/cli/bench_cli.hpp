/**
 * @file bench_cli.hpp
 * @brief Command-line argument handling for the benchmark harness.
 *
 * Splits the harness's own flags (--config, --list-ops, --progress, --help)
 * out of argv before Google Benchmark's parser sees them, and prints usage.
 */
#ifndef RPP_BENCH_CLI_HPP
#define RPP_BENCH_CLI_HPP

#include <string>

namespace rppbench {

/**
 * @brief Pull "--config=PATH" (or "--config PATH") out of argv.
 *
 * Removes it so Google Benchmark's flag parser never sees it. Mutates argc/argv
 * in place.
 * @param argc Argument count, updated in place.
 * @param argv Argument vector, updated in place.
 * @return The config path, or empty if absent.
 */
std::string extract_config(int &argc, char **argv);

/**
 * @brief Whether @p flag appears verbatim in argv.
 * @param argc Argument count.
 * @param argv Argument vector.
 * @param flag Exact token to search for.
 * @return True if present.
 */
bool has_flag(int argc, char **argv, const char *flag);

/**
 * @brief Whether an argv token is @p name exactly or "name=...".
 *
 * I.e. the caller set the flag, whether in "--flag value" or "--flag=value"
 * form. Used to let a Google Benchmark flag on the command line win over the
 * config's default.
 * @param argc Argument count.
 * @param argv Argument vector.
 * @param name Flag name to match.
 * @return True if the flag was set.
 */
bool has_flag_named(int argc, char **argv, const char *name);

/**
 * @brief Remove a valueless flag from argv (so benchmark::Initialize won't reject it).
 * @param argc Argument count, updated in place.
 * @param argv Argument vector, updated in place.
 * @param flag Exact token to remove.
 * @return Whether it was present.
 */
bool strip_flag(int &argc, char **argv, const char *flag);

/**
 * @brief Pull "--progress[=MODE]" out of argv and return the requested display mode.
 * @param argc Argument count, updated in place.
 * @param argv Argument vector, updated in place.
 * @return "" when the flag is absent, "simple" for a bare --progress (or an
 *         empty MODE), otherwise MODE verbatim (e.g. "fancy"). Validation is
 *         left to the caller.
 */
std::string extract_progress(int &argc, char **argv);

/**
 * @brief Value of a "--name=value" flag in argv (the last occurrence wins, as in Google Benchmark).
 * @param argc Argument count.
 * @param argv Argument vector.
 * @param name Flag name, without the '='.
 * @return The value, or empty if absent.
 */
std::string flag_value(int argc, char **argv, const char *name);

/**
 * @brief Directory default result files are written to (relative to the working directory).
 */
inline constexpr const char *kDefaultResultsDir = "results";

/**
 * @brief Make the run write a result file even when the user didn't ask for one.
 *
 * A run without --benchmark_out leaves nothing to compare or archive, so unless
 * the caller passed --benchmark_out (or --benchmark_list_tests, which runs
 * nothing), this appends
 * `--benchmark_out=results/<config-stem>_<YYYYmmdd-HHMMSS>_<host>.<ext>` to argv,
 * plus `--benchmark_out_format=json` if no format was given. The extension
 * follows the requested format. Creates the results directory. Call before
 * benchmark::Initialize().
 * @param argc Argument count, updated in place.
 * @param argv Argument vector, re-pointed at internally owned storage when an
 *             argument is appended (valid for the rest of the program).
 * @param configPath The sweep config, whose file stem prefixes the file name.
 * @return The injected output path, or empty if none was injected.
 */
std::string inject_default_output(int &argc, char **&argv, const std::string &configPath);

/**
 * @brief Print detailed usage.
 *
 * Covers this harness's own flags plus a quick reference to the forwarded Google
 * Benchmark flags.
 * @param prog The program name (argv[0]).
 */
void print_help(const char *prog);

} // namespace rppbench

#endif // RPP_BENCH_CLI_HPP
