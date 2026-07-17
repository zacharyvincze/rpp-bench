# Architecture

This document describes how the benchmark harness is structured after the **library-decoupling** refactor: a dependency-free *core* that knows nothing about any imaging library, and self-contained *library modules* (today only [RPP](https://github.com/ROCm/rpp)) that plug into it. The design is deliberately a **compile-time plugin system** — a new library is a directory under `libraries/` with its own CMake target, linked into the binary when its dependency is found, and nothing in the core changes.

## Contents

- [Terminology: library vs backend](#terminology-library-vs-backend)
- [The spine](#the-spine)
- [Layout: core vs library modules](#layout-core-vs-library-modules)
- [The core boundary](#the-core-boundary)
- [Inside a library module (RPP)](#inside-a-library-module-rpp)
- [Registration: how a library plugs in](#registration-how-a-library-plugs-in)
- [Config and naming](#config-and-naming)
- [Build system](#build-system)
- [Adding a new library](#adding-a-new-library)
- [Design decisions](#design-decisions)

## Terminology: library vs backend

Two axes that are easy to conflate:

- **library** — *which* implementation runs the op: `rpp` (and, in future, others). It is a **per-op scalar** in the config (defaults to `rpp`). Each `(op, library)` config line is self-contained and carries that library's own native params — there is no cross-library param sharing, merging, or canonical vocabulary. Comparability across libraries is an explicit authoring choice, not harness machinery.
- **backend** — *where* the op runs: `HOST` or `HIP`. This keeps RPP's existing meaning exactly. It stays a **fan-out axis** alongside dtype / layout / batch / size, because one library's params are identical across its own backends.

A library can expose more than one backend (RPP exposes HOST and HIP), which is why the library is not itself called "the backend".

## The spine

The end-to-end flow, wired together in [`src/bench_main.cpp`](../src/bench_main.cpp), is unchanged in shape:

```
load_config(JSON)  ->  register_benchmarks()  ->  RunSpecifiedBenchmarks()  ->  release_active_resources()
     |                       |                          |                              |
  BenchConfig         cartesian expansion         Google Benchmark               free last case
  (OpSpec list)       one bench per combo         drives timing loop
```

What changed is *who* does the library-specific work. The runner no longer allocates RPP descriptors or calls `rppCreate`; it asks a **`Library`** to build a **`BenchCase`** for each fully-resolved sweep point and drives that case through a neutral interface.

## Layout: core vs library modules

```
common/core/            neutral spine — no rpp/*.h, no HIP, no library headers
  bench_types.hpp         Backend / DataType / Layout enums + string<->enum
  bench_point.hpp         BenchPoint: one fully-resolved, library-neutral sweep point
  bench_config.*          config parsing (adds the per-op "library" scalar)
  bench_library.*         Library + BenchCase + OpCapabilities + LibraryRegistry
  bench_name.*            name encoding (adds the library: field)
  bench_runner.*          matrix expansion + timed body (library-driven, HIP-free)
common/cli/             argument handling + the --progress reporters (already neutral)

libraries/
  rpp/                    the RPP module — the ONLY place rpp/*.h and HIP appear
    rpp_enums.hpp           neutral <-> Rppt* translation, layout_info, dtype_size
    bench_context.hpp       BenchContext: the RPP-typed sweep point adapters receive
    bench_tensor.*          TensorBuffer + TensorSpec + device/pinned allocators
    bench_registry.*        OpAdapter interface + the RPP module's own op registry
    bench_simple_adapter.hpp / bench_generic_adapters.*   the adapter convenience bases
    rpp_library.*           RppLibrary : Library (owns the op registry, makeCase)
    rpp_case.*              RppCase : BenchCase (handle/stream/tensors, alloc, run, sync)
    ops/bench_*.cpp         the operator adapters (one per op)

src/bench_main.cpp      entry point; wires config -> runner -> reporters

scripts/                json2csv.py (adds a library column) + plot.py
```

The core compiles with a plain C++17 host compiler and links nothing library-specific. Every RPP and HIP symbol lives under `libraries/rpp/`.

## The core boundary

The core defines three small interfaces and never sees an RPP type.

**`BenchPoint`** — one fully-resolved, neutral sweep point. It carries the same scalar fields as before (`batch`, `width`, `height`, `dstWidth`, `dstHeight`, `params`, …) but with neutral enums (`Backend`, `DataType`, `Layout`) plus the `library` name. It is what the core hands to a library to build a case, and what `encode_name()` renders.

**`Library`** — a plugin. It owns its own operators and knows how to turn a `BenchPoint` into a runnable case:

```cpp
// common/core/bench_library.hpp
struct OpCapabilities {                       // neutral capability report for sweep filtering
    std::vector<DataType> dtypes;
    std::vector<Layout>   layouts;
    std::vector<Backend>  backends;
};

class BenchCase {                             // one built, ready-to-run sweep point
public:
    virtual ~BenchCase() = default;           // frees tensors/handle/stream, runs adapter teardown
    virtual std::string setup()   = 0;        // build buffers + session + adapter; "" on success
    virtual std::string runOnce() = 0;        // resetRoi + launch + synchronize; "" on success
    virtual size_t srcDataBytes() const = 0;  // for the throughput counter
};

class Library {
public:
    virtual ~Library() = default;
    virtual std::string name() const = 0;                       // "rpp"
    virtual bool hasOp(const std::string &op) const = 0;
    virtual std::vector<std::string> opNames() const = 0;
    virtual std::vector<Backend> availableBackends() const = 0; // compiled-in (HOST, maybe HIP)
    virtual OpCapabilities capabilities(const std::string &op) const = 0;
    virtual std::unique_ptr<BenchCase> makeCase(const BenchPoint &) const = 0;
};
```

The critical boundary decision: **buffer allocation is delegated wholesale to the library**. The core never learns that a tensor has an "image" vs "generic" descriptor dialect, never computes a stride or a halo, never touches a ROI. All of that — the RPP-kernel-specific allocation, guard padding, fill, and ROI packing — lives behind `BenchCase::setup()` / `runOnce()`. The core only starts/stops timing, counts iterations, and reads `srcDataBytes()`.

`BenchCase::runOnce()` also absorbs the device synchronization that used to sit inline in the runner (`hipStreamSynchronize`). The core runner therefore has **no `#if RPP_BENCH_HIP` and no HIP include** — "the op has finished" is the library's definition, behind `runOnce()`.

The runner keeps the same case-reuse discipline as before: expensive per-case state is built once (keyed by `caseId`) and reused across Google Benchmark's calibration ramp and repetitions; a new case releases the previous one, so at most one case's device memory is resident. What was `CaseResources` is now a thin core wrapper holding a `std::unique_ptr<BenchCase>`; the RPP-specific resources moved into `RppCase`.

## Inside a library module (RPP)

The RPP module is a self-contained implementation of the two core interfaces. Nothing here leaks into the core.

- **`RppLibrary : Library`** owns the RPP operator registry (its own table of name → adapter factory — there is no global op registry in the core), reports `availableBackends()` from the compiled-in `RPP_BENCH_HIP`, answers `capabilities(op)` by probing the adapter and translating its RPP-typed support lists to neutral enums, and `makeCase()` constructs an `RppCase`.
- **`RppCase : BenchCase`** is essentially the old `CaseResources`: it builds the `rppHandle_t` + `hipStream_t`, allocates the `TensorBuffer`s (image path via `init(w,h)`, generic path via `initGeneric(spec)`), fills them, runs the adapter, and on `runOnce()` refreshes the ROI, issues the timed `rppt_*` call, and synchronizes the stream. Its destructor frees everything and runs the adapter's `teardown()`.
- **`BenchContext`** is the **RPP-typed** sweep point the adapters receive — the same fields adapters always used (`ctx.backend` is an `RppBackend`, `ctx.dtype` an `RpptDataType`, `ctx.param<T>()` unchanged). `RppCase` builds it from the neutral `BenchPoint`. This is the typed bridge: the core speaks `BenchPoint`, adapters speak `BenchContext`, and the conversion happens once per case.
- **`OpAdapter`** and its convenience bases (`SimpleOpAdapter`, `GenericOpAdapter`, `VoxelOpAdapter`, `BroadcastOpAdapter`), **`TensorBuffer`**/`TensorSpec`, and the enum translation are exactly as before — just relocated under `libraries/rpp/`. The adapter interface stays fully RPP-typed (`run()` returns `RppStatus`, takes `rppHandle_t`), because it is internal to this module. The RPP-specific allocation hints (`srcSpecs`/`dstSpecs`/`srcOffsetBytes`/`srcAdditionalStride`) live on the RPP `OpAdapter` and are consumed by `RppCase`, so the core interface stays free of them.

The net effect on the ~70 operator adapters is a **one-line include change** each (`harness/…` → `rpp/…`); their types, signatures, capability overrides, and `REGISTER_RPP_BENCH("name", Adapter)` registration are untouched.

## Registration: how a library plugs in

Two independent, static-init self-registrations, both inside the library module:

1. Each operator registers into **its own library's** op registry via `REGISTER_RPP_BENCH("name", Adapter)` (unchanged spelling; it now targets `RppLibrary`'s table, not a global one).
2. The library registers **itself** into the core `LibraryRegistry` once, via a `LibraryRegistrar` at file scope in `rpp_library.cpp`.

Because each library module is built as a **CMake `OBJECT` library** whose objects are linked directly into the executable, these static initializers are guaranteed to run — sidestepping the classic "static registration stripped from a static archive" trap that would otherwise silently drop every op the moment the module became a separate link unit. `--list-ops` walks the `LibraryRegistry` and prints each library's ops.

An invariant worth stating: a library's adapters may assume the tensors and session they receive were produced by that same library. The core guarantees this — everything for a case is built by the one `Library` named in its `BenchPoint`.

## Config and naming

A per-op **`library`** scalar selects the implementation; it defaults to `rpp`, so existing configs run unchanged. The `backends` axis (`HOST`/`HIP`) is unchanged.

```jsonc
{
  "defaults": {
    "library": "rpp",                 // optional; default. Can be overridden per op.
    "backends": ["HOST", "HIP"],
    "dtypes":  ["U8", "F32"]
    // ...
  },
  "ops": [
    { "name": "brightness", "params": { "alpha": 1.75, "beta": 50 } }
    // a future second library would appear as its own line:
    // { "name": "brightness", "library": "other", "params": { ... } }
  ]
}
```

The matrix expansion skips a line whose library is not registered (with a warning), whose op that library does not provide, whose requested backend the library was not compiled with (the old HOST-only-build behavior, now reported via `Library::availableBackends()`), or whose dtype/layout/backend the adapter does not support.

**Name encoding** gains a `library:` field immediately after `op:`:

```
op:resize/library:rpp/backend:HIP/dtype:F32/layout:PKD3/batch:8/size:1920x1080/dst:960x540/params:interpolation=BICUBIC
```

This is a change to the name contract, so [`scripts/json2csv.py`](../scripts/json2csv.py) gains a `library` column. Field order is otherwise unchanged and params remain one sorted, comma-joined field.

## Build system

```cmake
add_library(bench_core STATIC common/core/*.cpp common/cli/**/*.cpp)   # no library deps
target_link_libraries(bench_core PUBLIC benchmark::benchmark nlohmann_json::nlohmann_json)

# Each library is an optional OBJECT module, gated by its dependency being found.
if(rpp_FOUND)                          # find_package(rpp) at configure time
    add_library(rpp_module OBJECT libraries/rpp/*.cpp libraries/rpp/ops/*.cpp)
    target_link_libraries(rpp_module PRIVATE bench_core rpp::rpp)      # + hip, rocrand on HIP
    # RPP_BENCH_HIP / GPU_SUPPORT set here, from rpp_BACKEND_TYPE
endif()

add_executable(rpp_bench src/bench_main.cpp)
target_link_libraries(rpp_bench PRIVATE bench_core)
if(TARGET rpp_module)
    target_link_libraries(rpp_bench PRIVATE $<TARGET_OBJECTS:rpp_module> rpp::rpp ...)
endif()
```

`bench_core` builds anywhere. A machine without ROCm still builds the core (the binary just registers no libraries and exits with a clear message). The active RPP backend (HOST vs HIP) is still inherited from the installed RPP and surfaced as `RPP_BENCH_HIP`, but that definition is scoped to `rpp_module` — the core never sees it.

## Adding a new library

1. Create `libraries/<name>/` with the module's sources and (optionally) a `CMakeLists.txt`.
2. Implement `Library` (owning your op registry + `capabilities` + `makeCase`) and `BenchCase` (allocation, run, sync).
3. Register the library once with `LibraryRegistrar`, and each op into your own registry.
4. Add an `if(<dep>_FOUND) add_library(<name>_module OBJECT ...)` block; link its objects into `rpp_bench` (or rename the executable). The `CONFIGURE_DEPENDS` glob picks up new op files on the next build.

No core file changes. Select the library from config with `"library": "<name>"`.

## Design decisions

- **library (scalar) vs backend (axis).** Matches RPP's own `HOST`/`HIP` "backend" meaning; the library is the new, orthogonal selector. No cross-library param sharing — each `(op, library)` line is self-contained.
- **Allocation delegated to the library.** The core never learns a descriptor dialect, stride, halo, or ROI. Keeps the core genuinely dependency-free and honors "the library owns its own functionality".
- **Compile-time OBJECT-library plugins.** Simplest model that gives plug-and-play: drop a directory, relink. No frozen ABI, no version-matched shared libraries, no `dlopen`. OBJECT libraries guarantee static-init registration survives the link. (Runtime `.so` loading remains a possible future add-on behind the same registration boundary.)
- **Typed bridge via `BenchContext`.** Adapters keep their ergonomic RPP-typed context and buffers; the neutral↔RPP conversion happens once per case in `RppCase`, so the ~70 adapters were untouched apart from an include path.
- **Sync behind `BenchCase::runOnce()`.** "The op finished" is the library's definition; the core timing loop is HIP-free.
