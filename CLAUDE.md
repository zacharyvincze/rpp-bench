# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Config-driven micro-benchmarks for [RPP](https://github.com/ROCm/rpp) (ROCm Performance Primitives) operators, built on Google Benchmark. The benchmark *matrix* (ops × backends × dtypes × layouts × batch × image sizes × per-op params) lives in a runtime JSON file, so exploring new cases means editing a config, not recompiling. It is a standalone CMake project that consumes an *installed* RPP via `find_package(rpp)` and fetches Google Benchmark + nlohmann/json at configure time.

## Build & run

```shell
cmake -S . -B build            # uses -DROCM_PATH / $ROCM_PATH / /opt/rocm, in that order
cmake --build build -j
./build/rpp_bench --config=config/smoke.json      # small/fast sanity sweep
./build/rpp_bench --config=config/example.json    # fuller sweep
./build/rpp_bench --list-ops                       # list compiled-in adapters
```

- Point at a specific RPP: `-DROCM_PATH=/opt/rocm-x.y.z` or `-Drpp_DIR=/path/to/lib/cmake/rpp`.
- The active backend (HOST vs HIP) is **inherited from the installed RPP**, surfaced to the code as the `RPP_BENCH_HIP` compile definition. A HOST-only build skips `HIP` config entries with a note; a HIP build can still run HOST cases.
- Only host-side HIP calls are issued, so a plain C++17 host compiler builds it — no amdclang toolchain needed.

There is no test suite. Validate changes by running `config/smoke.json` and checking `--list-ops`.

### Selecting cases

`--benchmark_filter=<regex>` (standard Google Benchmark, substring match, leading `-` inverts) runs against the full encoded name. Name fields are fixed-order: `op, backend, dtype, layout, batch, size, dst, params`. Filtering also trims JSON/CSV output. `--progress` swaps the per-case result table for one live progress line (its denominator respects the active filter).

### Analysing results

```shell
./build/rpp_bench --config=config/example.json --benchmark_out=results.json --benchmark_out_format=json
python3 scripts/json2csv.py results.json -o results.csv     # stdlib only
python3 scripts/plot_results.py results.json -o results.png # needs matplotlib + numpy
```

## Architecture

Flow: `bench_main.cpp` loads a `BenchConfig` → `register_benchmarks()` expands the matrix and registers one Google Benchmark per combo → Google Benchmark runs them → `release_active_resources()` frees the last case.

The harness lives in `common/`, grouped by role into four subdirectories. Headers are included **path-qualified from the `common/` root** (e.g. `#include "harness/bench_registry.hpp"`), and `common/` is the only include directory.

`common/config/`
- **`bench_config.*`** — parses the JSON into `BenchConfig` (global `min_time_sec` / `repetitions`, a list of `OpSpec`). Each op inherits `defaults` and may override any matrix axis. `params` (one set) and `param_sets` (a swept list) are mutually exclusive and both normalize into `OpSpec::paramSets` (always ≥1 entry).
- **`bench_enums.hpp`** — the single source of truth for string↔RPP-enum mapping (dtype, layout, backend, interpolation). Config parsing, name encoding, and adapters all go through it.

`common/harness/`
- **`bench_registry.hpp`** — the `OpAdapter` interface, the `BenchContext` (one fully-resolved sweep point), and the global `OpRegistry` singleton. Adapters receive `std::vector<TensorBuffer>` for sources and destinations and declare how many of each they need via `numSrc()`/`numDst()` (default 1 each). `SimpleOpAdapter` is a convenience base for the common one-in/one-out op: it forwards the vector forms to single-buffer `setup(ctx, src, dst)` / `run(ctx, src, dst, handle)` signatures, so single-source adapters never touch the vectors. Multi-source ops (e.g. `bitwise_and`) subclass `OpAdapter` directly and override `numSrc()`.
- **`bench_tensor.*`** — `TensorBuffer`: descriptor/stride/ROI setup and HIP-aware allocation, so adapters get ready-to-use src/dst and only build their own param tensors. HIP inputs are filled directly in device memory via rocRAND (no H2D copy). Multiple sources all share dims/dtype/layout (matching the single `srcDesc` the two-source `rppt_*` calls take) and are each filled independently.
- **`bench_runner.cpp`** — the cartesian expansion (nested loops over backend/dtype/layout/batch/size/dst/paramSet), skipping combos an adapter doesn't support or a backend not compiled in. Also owns `run_case()` (the timed body).
- **`bench_name.*`** — `encode_name()`, the benchmark-name encoding (its own file because it's a contract with `scripts/json2csv.py`).

`common/cli/`
- **`bench_cli.*`** — argument munging (`--config`, `--list-ops`, `--progress[=simple|fancy]`) pulled out of argv before Google Benchmark's parser sees it, plus the `--help` text.

`common/progress/` — the `--progress` live reporters, one Google Benchmark reporter per file.
- **`bench_progress.*`** — the shared `ProgressReporter` base (case-counting + run sampling), `count_matching()` (so the progress denominator respects `--benchmark_filter`), and the shared plain-line renderer.
- **`bench_progress_simple.*`** — `SimpleProgressReporter`: the default one-line style.
- **`bench_progress_dashboard.*`** — `DashboardProgressReporter`: the `--progress=fancy` bordered TTY dashboard (gradient bar, ETA, throughput, sparkline, backend/dtype tally), falling back to the plain line off a TTY.

`src/`
- **`bench_main.cpp`** — the entry point; just wires the above together.
- **`ops/bench_<name>.cpp`** — one adapter per op.

### Two things that are easy to get wrong

- **Case-resource reuse (`CaseResources g_case` in harness/bench_runner.cpp).** Google Benchmark re-invokes a case's function multiple times (calibration ramp + each repetition). Expensive per-case state (rppCreate handle, HIP stream, src/dst buffers, the adapter) is built **once**, keyed by `caseId`, and reused; a new case releases the previous one. So at most one case's GPU memory is resident, and this assumes single-threaded, sequential benchmarks. That's why `release_active_resources()` must be called after `RunSpecifiedBenchmarks()`.
- **The benchmark name is a contract.** `encode_name()` in harness/bench_name.cpp produces `op:.../backend:.../dtype:.../layout:.../batch:.../size:WxH[/dst:WxH][/params:k=v,...]`, and `scripts/json2csv.py` splits it back into columns. Keep the two in sync — changing the format silently breaks CSV conversion. Params are sorted and comma-joined for deterministic, uniquely-named swept cases.

## Adding an op

Each `rppt_*` function has a distinct C signature, so each op needs a small adapter under `src/ops/bench_<name>.cpp`:

1. Subclass `SimpleOpAdapter` for a one-source/one-destination op (see `common/harness/bench_registry.hpp`); subclass `OpAdapter` directly and override `numSrc()`/`numDst()` for a multi-source/destination op (see `src/ops/bench_bitwise_and.cpp` for the two-source pattern).
2. In `setup()` allocate op-specific param tensors with `bench_pinned_alloc` (so they work on HIP); issue the `rppt_*` call in `run()`; free in `teardown()`. Read config params via `ctx.param<T>("key", fallback)`.
3. Optionally override `supportedDtypes()` / `supportedLayouts()` to constrain the sweep, and `srcOffsetBytes()` / `srcAdditionalStride()` for filter ops that need halo padding on the source buffer.
4. `REGISTER_RPP_BENCH("config_name", AdapterClass)` at file scope.

The CMake glob (`CONFIGURE_DEPENDS`) picks up new files under `src/ops/` — just re-run `cmake --build`. Add the op to `config/example.json`, and note any backend/dtype/kernel quirk in a top-of-file comment (see `src/ops/bench_flip.cpp` for the pattern).

## Conventions

- Every source file opens with a banner comment explaining *why* it exists — keep that habit.
- C++ style is enforced by `.clang-format` (4-space indent, attached braces, right-aligned pointers); `.clang-tidy` is also present. Format staged changes: `clang-format -i $(git diff --name-only --diff-filter=ACM '*.cpp' '*.hpp')`. Non-C++ whitespace comes from `.editorconfig`.
- Markdown: write paragraphs as single lines (no hard-wrapping); newlines only between blocks.
