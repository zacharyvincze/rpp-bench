---
name: add-op-benchmark
description: Add one or more RPP operator benchmarks to rpp-bench — writes the adapter under src/ops/, registers it, adds it to the smoke and example configs, and confirms it runs on smoke. Use when asked to "add a benchmark for <op>", "benchmark rppt_<op>", or add several ops at once (fan out one agent per op). Stops to ask the user when an op needs harness/adapter changes beyond a new src/ops/ file.
---

# Add an RPP op benchmark

This skill adds a benchmark for one or more RPP `rppt_*` operators. The end state for each op: a self-registering adapter at `src/ops/bench_<name>.cpp`, an entry in both `config/smoke.json` and `config/example.json`, and a confirmed run on the smoke config.

Read `CLAUDE.md` first if you haven't this session — it is the source of truth for the architecture, the adapter bases, and the two contracts (generic `roiTensor` packing, benchmark name). This skill assumes that context.

## 0. Gather the op list

Collect the op name(s) the user wants (the config/registry name, e.g. `tensor_sum`, `brightness`) and any params they specified. If the user named an `rppt_*` C function, strip the `rppt_` prefix for the config name. If nothing about params was said, you will pick representative defaults during research.

**If more than one op is requested, fan out — see "Parallelizing multiple ops" below — otherwise do the single-op flow inline.**

## Single-op flow

### 1. Research the op

The installed RPP lives under `$ROCM_PATH` (fallback `/opt/rocm`). Two sources:

- **Header**: `$ROCM_PATH/include/rpp/rppt_*.h`. Grep for `rppt_<name>(` to get the exact signature and the doc block. The `srcDescPtr` doc line lists **Restrictions** (allowed `dataType`, `layout`, channel counts) — these drive your `supportedDtypes()` / `supportedLayouts()` overrides.
- **Test suite** (invaluable for param setup and output typing): `$ROCM_PATH/share/rpp/test/`. Grep `HOST/Tensor_image_host.cpp` / `HIP/Tensor_image_hip.cpp` (or the voxel test files) for the `rppt_<name>(` call site to see how param/ROI/output tensors are sized and typed. This is how the `tensor_sum` output array width (`Rpp64u` for U8/I8) was determined.

Note the signature shape: how many source and destination image buffers, whether it takes `RpptDescPtr` (4D image) or `RpptGenericDescPtr` (generic), what extra param tensors it needs, and whether it ends with a trailing `RppBackend executionBackend` (most do — pass `ctx.backend`).

### 2. Choose the adapter base

| Op shape | Base class | Header to include |
|---|---|---|
| 1 source, 1 destination, `RpptDescPtr` image | `SimpleOpAdapter` | `harness/bench_simple_adapter.hpp` |
| Multiple sources/dests, or a reduction (writes an array, no dst image), `RpptDescPtr` | `OpAdapter` directly (+ override `numSrc()`/`numDst()`) | `harness/bench_registry.hpp` |
| `RpptGenericDescPtr`, ND tensor + flat `roiTensor` | `GenericOpAdapter` | `harness/bench_generic_adapters.hpp` |
| `RpptGenericDescPtr`, 5D voxel + `RpptROI3D` | `VoxelOpAdapter` | `harness/bench_generic_adapters.hpp` |
| `RpptGenericDescPtr`, two ND sources + broadcast mode | `BroadcastOpAdapter` | `harness/bench_generic_adapters.hpp` |

Reference adapters to copy the shape from:
- Simple param op: `src/ops/bench_brightness.cpp`
- Two-source image op: `src/ops/bench_bitwise_and.cpp`
- Reduction (no dst image): `src/ops/bench_tensor_sum.cpp`
- Generic ND: `src/ops/bench_transpose.cpp`; voxel: `src/ops/bench_add_scalar.cpp`; broadcast: `src/ops/bench_tensor_add_tensor.cpp`
- Filter op needing halo padding: `src/ops/bench_flip.cpp` (and its `srcOffsetBytes()`/`srcAdditionalStride()` neighbours)

### 3. Discrepancy gate — STOP and ask the user

If the op fits one of the bases above and needs only a new `src/ops/*.cpp` file plus config entries, proceed. **Otherwise pause and ask the user (via AskUserQuestion) before writing any code**, describing the mismatch and your proposed extension. Trigger this gate when the op would require any of:

- A descriptor dialect or ROI shape none of the existing bases model (would need a new `TensorKind`, a new `TensorBuffer` member/view, or a new adapter base in `common/harness/`).
- A change to the runner, the name encoding, or the config parser (e.g. it needs a matrix axis that doesn't exist, or an output whose type/shape the runner can't allocate).
- Any edit outside `src/ops/` and the config files (i.e. touching `common/harness/*`, `common/config/*`, `CMakeLists.txt`).
- An `rppt_*` variant not present in the installed headers (backend/GPU_SUPPORT-gated and unavailable), so it wouldn't compile.

When a fan-out agent hits this gate it must not guess — it returns a `needs_decision` report (see the agent contract) and the coordinator surfaces it with AskUserQuestion.

### 4. Write the adapter

Create `src/ops/bench_<name>.cpp`, mirroring the closest reference adapter and the repo conventions in `CLAUDE.md`:

- Open with the `/** @file ... @brief ... */` banner and a comment noting the `rppt_*` signature (and any backend/dtype/kernel quirk).
- `setup()`: allocate op-specific param/output tensors with `bench_pinned_alloc(bytes, ctx.isHip)`; read config with `ctx.param<T>("key", fallback)`.
- `run()`: issue the `rppt_*` call; pass `ctx.backend` for the trailing `executionBackend` arg; return its `RppStatus`.
- `teardown()`: `bench_pinned_free` everything you allocated.
- Constrain the sweep with `supportedDtypes()` / `supportedLayouts()` / `supportedBackends()` per the header's Restrictions.
- End with `REGISTER_RPP_BENCH("<name>", AdapterClass)` at file scope, inside `namespace rppbench`.

No CMake edit is needed — the `CONFIGURE_DEPENDS` glob over `src/ops/*.cpp` picks the file up on the next build.

### 5. Add to both configs

Add `{ "name": "<name>"[, "params": { ... }] }` to **`config/smoke.json`** and **`config/example.json`**. Group it near related ops (e.g. the `tensor_*` block). Include a `params` object only if the op reads params; use representative values. If the op is backend/dtype/layout-constrained, add the matching override keys (`"backends"`, `"dtypes"`, `"layouts"`, `"dst_sizes"`) so smoke actually exercises a valid case — e.g. a HIP-only op needs `"backends": ["HIP"]`, an F32-only op `"dtypes": ["F32"]`.

### 6. Build and verify on smoke

```shell
cmake --build build -j          # if build/ is missing: cmake -S . -B build && cmake --build build -j
./build/rpp_bench --list-ops | grep <name>
./build/rpp_bench --config=config/smoke.json --benchmark_filter='<name>'
```

The op must appear in `--list-ops` and produce timing rows (not `SkipWithError`) on smoke. If it errors, fix the adapter before reporting success. Run `clang-format -i src/ops/bench_<name>.cpp`.

## Parallelizing multiple ops

When several ops are requested, fan out to keep it fast — but serialize the shared-file work to avoid write conflicts:

1. **Spawn one subagent per op** (Agent tool, `general-purpose`). For a large batch, cap concurrency at a handful and batch the rest. Each agent does **only steps 1–4** for its op: research and write its own `src/ops/bench_<op>.cpp` (a distinct new file — no two agents touch the same path). Agents must **not** edit the configs, run CMake, or touch `common/`.
2. **Agent return contract** — each agent returns a compact report:
   - `status`: `written` | `needs_decision`
   - `op`: the config name
   - `adapter_path`: the file it created (when `written`)
   - `config_entry`: the exact JSON object to add (name + any params/overrides)
   - `constraints`: dtype/layout/backend restrictions it applied and why
   - `notes`: quirks, or — when `needs_decision` — the mismatch and proposed extension (no file written)
3. **Coordinator (you) finishes serially**: for every `needs_decision` report, ask the user with AskUserQuestion before doing anything further with that op. For each `written` op, add its `config_entry` to `config/smoke.json` and `config/example.json`. Then build **once**, and verify each op with `--benchmark_filter`. Fix or re-delegate any that fail.

Give each agent this skill's steps 1–4, the target op name/params, and the reference adapter paths so it stays consistent with the codebase.

## Final summary

Report to the user:
- Each op added, its adapter path, adapter base used, and the applied dtype/layout/backend constraints.
- Confirmation that smoke built and each op produced timing rows (mention HOST vs HIP as run).
- Any op that hit the discrepancy gate and what was decided.
- Anything noteworthy: params chosen, output typing subtleties, ops a HOST-only build will skip, or kernel quirks found in the RPP docs/test suite.
