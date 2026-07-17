# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Config-driven micro-benchmarks for [RPP](https://github.com/ROCm/rpp) (ROCm Performance Primitives) operators, built on Google Benchmark. The benchmark *matrix* (ops × backends × dtypes × layouts × batch × image sizes × per-op params) lives in a runtime JSON file, so exploring new cases means editing a config, not recompiling. It is a standalone CMake project that consumes an *installed* RPP via `find_package(rpp)` and fetches Google Benchmark + nlohmann/json at configure time.

The harness is split into a **dependency-free core** (`common/`) and one **library module** per imaging library (`libraries/<name>/`); today the only module is RPP. A per-op `library` config scalar (default `rpp`) selects the implementation; `backend` keeps RPP's `HOST`/`HIP` meaning. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full design and how to add a library.

## Build & run

```shell
cmake -S . -B build            # uses -DROCM_PATH / $ROCM_PATH / /opt/rocm, in that order
cmake --build build -j
./build/rpp_bench --config=config/smoke.json      # small/fast sanity sweep
./build/rpp_bench --config=config/example.json    # fuller sweep
./build/rpp_bench --list-ops                       # list registered library/op pairs
```

- Point at a specific RPP: `-DROCM_PATH=/opt/rocm-x.y.z` or `-Drpp_DIR=/path/to/lib/cmake/rpp`.
- The active backend (HOST vs HIP) is **inherited from the installed RPP**, surfaced to the code as the `RPP_BENCH_HIP` compile definition. A HOST-only build skips `HIP` config entries with a note; a HIP build can still run HOST cases.
- Only host-side HIP calls are issued, so a plain C++17 host compiler builds it — no amdclang toolchain needed.

There is no test suite. Validate changes by running `config/smoke.json` and checking `--list-ops`.

### Selecting cases

`--benchmark_filter=<regex>` (standard Google Benchmark, substring match, leading `-` inverts) runs against the full encoded name. Name fields are fixed-order: `op, library, backend, dtype, layout, batch, size, dst, params`. Filtering also trims JSON/CSV output. `--progress` swaps the per-case result table for one live progress line (its denominator respects the active filter).

### Analysing results

```shell
./build/rpp_bench --config=config/example.json --benchmark_out=results.json --benchmark_out_format=json
python3 scripts/json2csv.py results.json -o results.csv     # stdlib only
python3 scripts/plot.py results.json -o results.png         # needs matplotlib + numpy
```

## Architecture

Flow: `bench_main.cpp` loads a `BenchConfig` → `register_benchmarks()` resolves each op's `Library`, expands the matrix, and registers one Google Benchmark per combo → Google Benchmark runs them, each driven through a `BenchCase` → `release_active_resources()` frees the last case.

The tree splits into a **dependency-free core** (`common/`) and **library modules** (`libraries/<name>/`). The core includes no imaging-library or HIP headers; every RPP/HIP symbol lives under `libraries/rpp/`. There are two include roots — `common/` (headers as `#include "core/..."`, `"cli/..."`) and `libraries/` (module headers as `#include "rpp/..."`). See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the design rationale.

**`common/core/` — the neutral spine (no RPP, no HIP):**
- **`bench_types.hpp`** — the neutral vocabulary enums (`Backend` HOST/HIP, `DataType`, `Layout`) plus their string↔enum mapping. The dependency-free counterpart of the old `bench_enums.hpp`.
- **`bench_point.hpp`** — `BenchPoint`, one fully-resolved, library-neutral sweep point (neutral enums + the JSON param blob + `param<T>()`). What the core hands to a library and to `encode_name()`.
- **`bench_config.*`** — parses the JSON into `BenchConfig` (global `min_time_sec` / `repetitions`, a list of `OpSpec`). Each op inherits `defaults`, may override any matrix axis, and carries a `library` scalar (default `rpp`). `params` (one set) and `param_sets` (a swept list) are mutually exclusive and both normalize into `OpSpec::paramSets` (always ≥1 entry).
- **`bench_library.*`** — the plugin boundary: the `Library` interface (owns its ops; reports `capabilities`/`availableBackends`; builds cases via `makeCase`), the `BenchCase` interface (`setup`/`runOnce`/`srcDataBytes` — all library work hides behind it), and the `LibraryRegistry` singleton libraries self-register into. There is deliberately **no global op registry** in the core.
- **`bench_runner.*`** — the cartesian expansion (nested loops over backend/dtype/layout/batch/size/dst/paramSet), resolving each op's library and filtering by the library's neutral `OpCapabilities`. Owns `run_case()` (the timed body): it builds a `BenchCase` once per `caseId`, reuses it, and calls `runOnce()` in the timing loop. Contains **no HIP include** — the device sync lives behind `BenchCase::runOnce()`.
- **`bench_name.*`** — `encode_name()`, the benchmark-name encoding (its own file because it's a contract with `scripts/json2csv.py`).

**`common/cli/`** — `bench_cli.*` (argument munging for `--config`, `--list-ops`, `--progress[=simple|fancy]`, plus `--help`) and **`cli/progress/`**, the `--progress` live reporters: `bench_progress.*` (the shared `ProgressReporter` base + `count_matching()`), `bench_progress_simple.*` (default one-line), and `bench_progress_dashboard.*` (`--progress=fancy` TTY dashboard, falling back to the plain line off a TTY). All already library-neutral.

**`libraries/rpp/` — the RPP module (the only place `rpp/*.h` and HIP appear):**
- **`rpp_enums.hpp`** — neutral↔RPP translation (`to_rpp_dtype`/`to_neutral_dtype`, backend, `layout_info`, `dtype_size`, interpolation parsing). The module boundary for enums.
- **`bench_context.hpp`** — `BenchContext`, the **RPP-typed** sweep point adapters receive (`ctx.backend` is an `RppBackend`, `ctx.dtype` an `RpptDataType`), built once per case from a `BenchPoint`. The typed bridge that keeps adapters ergonomic.
- **`bench_registry.hpp`** — the RPP `OpAdapter` interface and the RPP module's own `OpRegistry` singleton. Adapters receive `std::vector<TensorBuffer>` for sources/destinations and declare counts via `numSrc()`/`numDst()` (default 1). Multi-source ops (e.g. `bitwise_and`) subclass `OpAdapter` directly and override `numSrc()`.
- **`bench_simple_adapter.hpp`** — `SimpleOpAdapter`, the convenience base for the common one-in/one-out image op: forwards the vector forms to single-buffer `setup(ctx, src, dst)` / `run(ctx, src, dst, handle)` signatures. The image-dialect counterpart to `bench_generic_adapters.*`.
- **`bench_generic_adapters.*`** — the adapter bases for ops that use RPP's **generic tensor descriptor** (`RpptGenericDesc`): `GenericOpAdapter` (ND tensor + flat `roiTensor`: transpose, slice, normalize, log), `VoxelOpAdapter` (5D NCDHW/NDHWC + `RpptROI3D`: the `*_scalar`, `flip_voxel`, `gaussian_noise_voxel` ops), and `BroadcastOpAdapter` (two ND sources + broadcast mode: `tensor_*_tensor`, `concat`). They build on the `srcSpecs()`/`dstSpecs()` hooks; shape-math bodies live in the `.cpp`.
- **`bench_tensor.*`** — `TensorBuffer`: descriptor/stride/ROI setup and HIP-aware allocation. HIP inputs are filled directly in device memory via rocRAND (no H2D copy). One `TensorBuffer` serves both descriptor dialects off a shared byte core (alloc + fill + guard): `init(...w,h...)` builds the image view (`desc`/`descPtr` + `roi`/`roiType`); `initGeneric(spec)` builds the generic view (`gdesc`/`gdescPtr` + a flat `roiTensor` for `GENERIC_ND`, or an `RpptROI3D` `roi3d` for `VOXEL`). `kind` records which view is live; adapters touch only the members for their dialect.
- **`rpp_library.*`** — `RppLibrary : Library`: owns the RPP `OpRegistry`, translates adapter capabilities to neutral enums, reports `availableBackends()` from `RPP_BENCH_HIP`, and builds an `RppCase` per point. Self-registers into the core `LibraryRegistry`.
- **`rpp_case.*`** — `RppCase : BenchCase`: holds the handle/stream/`TensorBuffer`s/adapter and owns the per-case lifecycle — buffer allocation (image `init` vs generic `initGeneric`, spec-driven from the adapter's `srcSpecs()`/`dstSpecs()`), fill, the timed `rppt_*` call, and the HIP stream sync. The old `CaseResources` logic.
- **`ops/bench_<name>.cpp`** — one adapter per op.

**`src/bench_main.cpp`** — the entry point; wires config → runner → reporters and lists whatever libraries registered themselves.

### Two things that are easy to get wrong

- **Case-resource reuse.** Google Benchmark re-invokes a case's function multiple times (calibration ramp + each repetition). The core (`bench_runner.cpp`) holds one live `BenchCase` (`g_case`), built **once** per `caseId` and reused; a new case releases the previous one. The RPP resources themselves (rppCreate handle, HIP stream, src/dst buffers, adapter) live inside `RppCase`, freed by its destructor. So at most one case's GPU memory is resident; this assumes single-threaded, sequential benchmarks. That's why `release_active_resources()` must be called after `RunSpecifiedBenchmarks()`. A library's adapters may assume their tensors/session came from that same library — the core guarantees it (everything for a case is built by the one `Library` named in its `BenchPoint`).
- **The generic `roiTensor` packing is a contract with RPP.** For `GENERIC_ND` tensors the flat `roiTensor` is `batch * nSpatial * 2` `Rpp32u`s (`nSpatial` = descriptor `numDims` minus the batch dim), laid out **per sample as two blocks**: `[begin_0..begin_{s-1}, length_0..length_{s-1}]` — not interleaved. The generic descriptor's `dims[0]` is the batch and `dims[1..]` are the spatial extents, with packed (contiguous) strides. `TensorBuffer::initGeneric`/`resetRoi` build this to match RPP's own `fill_roi_values`/`compute_strides`; getting the block order, the `nSpatial` (vs `numDims`) count, or the stride packing wrong makes the kernel read out of bounds.
- **The benchmark name is a contract.** `encode_name()` in `common/core/bench_name.cpp` produces `op:.../library:.../backend:.../dtype:.../layout:.../batch:.../size:WxH[/dst:WxH][/params:k=v,...]`, and `scripts/json2csv.py` splits it back into columns. Keep the two in sync — changing the format silently breaks CSV conversion. Params are sorted and comma-joined for deterministic, uniquely-named swept cases.

## Adding an op

Each `rppt_*` function has a distinct C signature, so each op needs a small adapter under `libraries/rpp/ops/bench_<name>.cpp`:

1. Subclass `SimpleOpAdapter` for a one-source/one-destination op (include `rpp/bench_simple_adapter.hpp`); subclass `OpAdapter` directly and override `numSrc()`/`numDst()` for a multi-source/destination op (include `rpp/bench_registry.hpp`; see `libraries/rpp/ops/bench_bitwise_and.cpp` for the two-source pattern).
2. In `setup()` allocate op-specific param tensors with `bench_pinned_alloc` (so they work on HIP); issue the `rppt_*` call in `run()`; free in `teardown()`. Read config params via `ctx.param<T>("key", fallback)`.
3. Optionally override `supportedDtypes()` / `supportedLayouts()` / `supportedBackends()` to constrain the sweep, and `srcOffsetBytes()` / `srcAdditionalStride()` for filter ops that need halo padding on the source buffer.
4. `REGISTER_RPP_BENCH("config_name", AdapterClass)` at file scope.

The CMake glob (`CONFIGURE_DEPENDS`) picks up new files under `libraries/rpp/ops/` — just re-run `cmake --build`. Add the op to `config/example.json`, and note any backend/dtype/kernel quirk in a top-of-file comment (see `libraries/rpp/ops/bench_flip.cpp` for the pattern).

### Adding a generic-descriptor op

Ops whose `rppt_*` signature takes `RpptGenericDescPtr` (not `RpptDescPtr`) subclass a generic base from `rpp/bench_generic_adapters.hpp` (include that header instead of `bench_registry.hpp`), and declare their tensor shapes via `srcSpecs()`/`dstSpecs()` (returning `TensorSpec`s) so `RppCase` allocates the generic view. Pick the base by ROI shape:

1. **`GenericOpAdapter`** — ND tensor with a flat begin/length `roiTensor` (transpose, slice, normalize, log). The default spec is a shape-preserving 1-in/1-out ND tensor derived from the image sweep point (`ctx.layout` picks NHWC vs NCHW); override `tensorSpec()` to change rank/shape, or `dstSpecs()` when the destination differs (transpose permutes dims; concat doubles an axis). Use `src[i].gdescPtr` and `src[i].roiTensor` in `run()`.
2. **`VoxelOpAdapter`** — 5D NCDHW/NDHWC tensor with a per-sample `RpptROI3D` (the `*_scalar` ops, `flip_voxel`, `gaussian_noise_voxel`). Depth has no matrix axis; it is read as a `depth` op param. Use `src[0].roi3d` / `src[0].roi3dType`.
3. **`BroadcastOpAdapter`** — two ND sources (`numSrc()==2`) sharing one shape (broadcasting disabled) plus a broadcast mode (`tensor_*_tensor`, `concat`). Use both `src[0]`/`src[1]` `.gdescPtr` and `.roiTensor`, and `broadcastMode(ctx)`.

The generic axes (depth, permutation, concat axis, axisMask, slice fraction, ...) are **op params**, not matrix axes — read them with `ctx.param<T>(...)`. This keeps the benchmark-name contract and `json2csv.py` untouched. Constrain the sweep with `supportedDtypes()` (most of these ops are F32-only, the bitwise `tensor_*` ops U8-only). `libraries/rpp/ops/bench_transpose.cpp`, `bench_add_scalar.cpp`, and `bench_tensor_add_tensor.cpp` are the reference adapters for the three families.

### Adding a library

A new imaging library is a module under `libraries/<name>/`, built as its own CMake `OBJECT` library and linked into the executable when its dependency is found — no core changes. Implement `Library` (owning your op registry + `capabilities` + `makeCase`) and `BenchCase` (allocation, run, device sync), register the library once with `LibraryRegistrar`, and select it from config with `"library": "<name>"`. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) ("Adding a new library") for the full recipe, including why OBJECT libraries are required for the static-init registration to survive the link.

## Conventions

- Every source file opens with a banner comment explaining *why* it exists — keep that habit.
- Document types and functions with Doxygen javadoc-style comments (`/** ... */` with `@brief`, `@param`, `@return`) — match the existing headers (`bench_registry.hpp`, `bench_tensor.hpp`). Every file's banner is a `/** @file ... @brief ... */` block.
- Prefer one class/concern per file; when a header grows unwieldy, split it (declarations in the `.hpp`, non-trivial bodies in a sibling `.cpp`) rather than letting it sprawl — see `bench_generic_adapters.*`.
- C++ style is enforced by `.clang-format` (4-space indent, attached braces, right-aligned pointers); `.clang-tidy` is also present. Format staged changes: `clang-format -i $(git diff --name-only --diff-filter=ACM '*.cpp' '*.hpp')`. Non-C++ whitespace comes from `.editorconfig`.
- Markdown: write paragraphs as single lines (no hard-wrapping); newlines only between blocks.
