# Contributing to rpp-bench

Thanks for helping improve the RPP benchmark harness. This guide covers the essentials; the [README](README.md) has the deeper detail on the config format and the harness internals.

## Prerequisites

- An **installed RPP** (from ROCm, or a local build) discoverable via `find_package(rpp)` — see the README's *Build* section for pointing CMake at a specific ROCm or `rpp_DIR`.
- CMake ≥ 3.20 and the ROCm clang toolchain (`amdclang++`). Google Benchmark and nlohmann/json are fetched automatically at configure time.
- Python 3 for the helper scripts in [scripts/](scripts/).

## Building

```shell
cmake -S . -B build
cmake --build build -j
```

Confirm your change still links and lists correctly:

```shell
./build/rpp_bench --list-ops
./build/rpp_bench --config=config/smoke.json
```

Run `config/smoke.json` (small, fast) as a sanity check before opening a PR; `config/example.json` is the fuller sweep.

## Code style

C++ style is enforced by [.clang-format](.clang-format) — 4-space indent, attached braces, right-aligned pointers. Format your changes before committing:

```shell
clang-format -i $(git diff --name-only --diff-filter=ACM '*.cpp' '*.hpp')
```

Whitespace conventions for non-C++ files (JSON configs, CMake, Python) come from [.editorconfig](.editorconfig); use an editor that respects it.

Match the surrounding code: the sources use a banner comment at the top of each file explaining *why* it exists — keep that habit when adding files.

## Adding an operator

Op adapters live in `libraries/rpp/ops/bench_<name>.cpp` and self-register. The full recipe is in the README's [*Adding an op*](README.md#adding-an-op) section. In short:

1. Subclass `SimpleOpAdapter` or `OpAdapter` (see [libraries/rpp/bench_registry.hpp](libraries/rpp/bench_registry.hpp)).
2. Allocate op-specific param tensors in `setup()` (use `bench_pinned_alloc` so they work on HIP), issue the `rppt_*` call in `run()`, free in `teardown()`.
3. Override `supportedDtypes()` / `supportedLayouts()` / `supportedBackends()` (and the halo-padding hooks for filter ops) as needed.
4. `REGISTER_RPP_BENCH("config_name", AdapterClass)`.

The CMake glob picks up new files under `libraries/rpp/ops/` — re-run `cmake --build`. Add the op to `config/example.json` so it's exercised, and note any backend or dtype quirks in a comment (as [`bench_flip.cpp`](libraries/rpp/ops/bench_flip.cpp) does). Adding a whole new *library* is documented in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Pull requests

- Keep PRs focused; one op or one concern per PR where practical.
- Describe how you validated: which config you ran, and on which backend (HOST / HIP) and ROCm version.
- Note any new op that a HOST-only build must skip, or any known kernel quirk.
