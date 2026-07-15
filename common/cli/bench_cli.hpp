// ============================================================================
// bench_cli.hpp - command-line argument handling for the benchmark harness.
//
// Splits the harness's own flags (--config, --list-ops, --progress, --help)
// out of argv before Google Benchmark's parser sees them, and prints usage.
// ============================================================================
#ifndef RPP_BENCH_CLI_HPP
#define RPP_BENCH_CLI_HPP

#include <string>

namespace rppbench {

// Pull "--config=PATH" (or "--config PATH") out of argv so Google Benchmark's
// flag parser never sees it. Mutates argc/argv in place. Returns the path (empty
// if absent).
std::string extract_config(int &argc, char **argv);

// Whether `flag` appears verbatim in argv.
bool has_flag(int argc, char **argv, const char *flag);

// Whether an argv token is `name` exactly or `name=...` - i.e. the caller set the
// flag, whether in "--flag value" or "--flag=value" form. Used to let a Google
// Benchmark flag on the command line win over the config's default.
bool has_flag_named(int argc, char **argv, const char *name);

// Remove a valueless flag from argv (so benchmark::Initialize won't reject it).
// Returns whether it was present.
bool strip_flag(int &argc, char **argv, const char *flag);

// Pull "--progress[=MODE]" out of argv and return the requested display mode:
// "" when the flag is absent, "simple" for a bare --progress (or an empty MODE),
// otherwise MODE verbatim (e.g. "fancy"). Validation is left to the caller.
std::string extract_progress(int &argc, char **argv);

// Print detailed usage: this harness's own flags plus a quick reference to the
// forwarded Google Benchmark flags. `prog` is argv[0].
void print_help(const char *prog);

} // namespace rppbench

#endif // RPP_BENCH_CLI_HPP
