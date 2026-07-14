# rpp-mark

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Config-driven micro-benchmarks for [RPP](https://github.com/ROCm/rpp) (ROCm Performance Primitives) operators, built on [Google Benchmark].

The benchmark *matrix* — which ops, backends, datatypes, layouts, batch sizes, image sizes, and per-op params to sweep — is described in a JSON file and read at runtime, so exploring a new set of cases means editing a config, not recompiling. It's a standalone CMake project: it consumes an *installed* RPP via `find_package(rpp)` and fetches Google Benchmark + nlohmann/json at configure time.

- [Requirements](#requirements)
- [Build](#build)
- [Run](#run)
- [Filtering benchmarks](#filtering-benchmarks)
- [Analysing results](#analysing-results)
- [Config format](#config-format)
- [Project layout](#project-layout)
- [Adding an op](#adding-an-op)
- [Contributing](#contributing)
- [License](#license)

## Requirements

- **An installed RPP** discoverable via `find_package(rpp)` — typically from a ROCm install under `/opt/rocm`, or a local RPP build. The active backend (HOST vs HIP) is inherited from it.
- **CMake ≥ 3.20** and any C++17 host compiler. The benchmarks only issue host-side HIP calls, so no ROCm/amdclang toolchain is needed to build.
- **Python 3** for the result-analysis scripts in [scripts/](scripts/). `json2csv.py` uses only the standard library; `plot_results.py` needs `matplotlib` and `numpy`.

Google Benchmark and nlohmann/json are fetched automatically at configure time — no system install of either is required.

## Build

```shell
cmake -S . -B build            # uses $ROCM_PATH or /opt/rocm
cmake --build build -j
```

Useful configure options:

- `-DROCM_PATH=/opt/rocm-x.y.z` — pick a specific ROCm.
- `-Drpp_DIR=/path/to/lib/cmake/rpp` — point straight at the RPP package config.

A HIP-enabled RPP can still run HOST benchmarks; a HOST-only build skips any `HIP` entries in the config with a note.

## Run

```shell
# Run a sweep described by a config file.
./build/rpp_bench --config=config/example.json

# List the op adapters compiled in.
./build/rpp_bench --list-ops

# Compact live progress instead of a result block per benchmark (great for big
# sweeps). Combine with --benchmark_out to still capture the full results.
./build/rpp_bench --config=config/example.json --progress \
                  --benchmark_out=results.json --benchmark_out_format=json

# Standard Google Benchmark flags are forwarded, e.g. filter / JSON output.
./build/rpp_bench --config=config/smoke.json \
                  --benchmark_filter='op:resize.*backend:HIP' \
                  --benchmark_out=results.json --benchmark_out_format=json
```

Two configs ship in [config/](config/): `smoke.json` (small and fast, for a sanity check) and `example.json` (a fuller sweep). Each benchmark reports `real_time`, `cpu_time`, plus the rate counters `images_per_sec`, `pixels_per_sec`, and `bytes_per_second`.

## Filtering benchmarks

Use Google Benchmark's `--benchmark_filter=<regex>` to select which cases run. The regex is matched (as a substring search) against the full benchmark name, so you can target any encoded field:

```
op:resize/backend:HIP/dtype:F32/layout:PKD3/batch:8/size:1920x1080/dst:960x540/params:interpolation=BICUBIC
```

```shell
# Preview what a filter matches without running anything.
./build/rpp_bench --config=config/example.json --benchmark_list_tests \
                  --benchmark_filter='op:resize'

# One op / one backend / one dtype.
./build/rpp_bench --config=config/example.json --benchmark_filter='op:gaussian_blur'
./build/rpp_bench --config=config/example.json --benchmark_filter='backend:HIP'
./build/rpp_bench --config=config/example.json --benchmark_filter='dtype:F32'

# Combine fields with .* (name fields are in a fixed order: op, backend, dtype,
# layout, batch, size, dst, params).
./build/rpp_bench --config=config/example.json --benchmark_filter='op:resize.*dtype:F32.*BICUBIC'

# Alternation: two ops.
./build/rpp_bench --config=config/example.json --benchmark_filter='op:(brightness|resize)'

# A specific size or param variant (trailing '/' avoids partial matches like batch:64 vs 640).
./build/rpp_bench --config=config/example.json --benchmark_filter='size:1920x1080'
./build/rpp_bench --config=config/example.json --benchmark_filter='batch:64/'
./build/rpp_bench --config=config/example.json --benchmark_filter='interpolation=NEAREST_NEIGHBOR'
```

**Excluding** benchmarks: prefix the regex with `-` to run everything that does *not* match.

```shell
./build/rpp_bench --config=config/example.json --benchmark_filter='-backend:HOST'   # skip HOST
./build/rpp_bench --config=config/example.json --benchmark_filter='-op:gaussian_blur'  # skip one op
```

Filtering happens at run time, so it also trims the JSON/CSV output to just the selected cases — a quick way to produce a focused results file. `--progress` accounts for the active filter, so its denominator reflects the number of cases actually selected (not all registered ones).

## Analysing results

Run with `--benchmark_out=results.json --benchmark_out_format=json` to capture machine-readable results, then post-process them with the scripts in [scripts/](scripts/).

**CSV** — Google Benchmark's built-in CSV reporter is deprecated, so emit native JSON and convert:

```shell
./build/rpp_bench --config=config/example.json \
                  --benchmark_out=results.json --benchmark_out_format=json
python3 scripts/json2csv.py results.json -o results.csv
```

The benchmark name encodes the combo (`op:<name>/backend:<HOST|HIP>/dtype:<...>/layout:<...>/batch:<n>/size:<WxH>[/dst:<WxH>]`); `json2csv.py` splits it back into columns and joins the numeric counters, one row per measurement (aggregate mean/median/stddev rows are dropped).

**Plot** — `plot_results.py` renders a faceted small-multiples figure (rows = batch size, columns = source size, bars = dtype × layout) on a shared log y-axis, so scaling across every axis is comparable at a glance:

```shell
python3 scripts/plot_results.py results.json -o results.png
python3 scripts/plot_results.py results.json               # writes <input>.png
```

## Config format

```jsonc
{
  "min_time_sec": 0.2,        // optional: min wall time per benchmark
  "repetitions": 3,           // optional: repeats -> mean/median/stddev
  "warmup_iterations": 5,     // optional: untimed iterations run once per case first
  "defaults": {               // base matrix; every op inherits these
    "backends":    ["HOST", "HIP"],
    "dtypes":      ["U8", "F32"],        // U8 | F32 | F16 | I8
    "layouts":     ["PKD3", "PLN3"],     // PKD3 (NHWC c3) | PLN3 (NCHW c3) | PLN1 (NCHW c1)
    "batch_sizes": [1, 8, 64],
    "image_sizes": [[224, 224], [1920, 1080]]
  },
  "ops": [
    { "name": "brightness", "params": { "alpha": 1.75, "beta": 50 } },
    { "name": "gaussian_blur", "params": { "kernel_size": 5, "std_dev": 1.5 } },
    { "name": "flip", "params": { "horizontal": 1, "vertical": 0 } },  // h+v mirror underflows the HIP kernel; see bench_flip.cpp
    { "name": "resize",                       // per-op axis overrides win over defaults
      "dtypes": ["U8", "F32"],
      "dst_sizes": [[112, 112], [960, 540]],  // resize target sizes (source from matrix)
      "param_sets": [                         // swept: one case per entry
        { "interpolation": "BILINEAR" },
        { "interpolation": "NEAREST_NEIGHBOR" },
        { "interpolation": "BICUBIC" }
      ] }
  ]
}
```

Any matrix axis set inside an op overrides the default for that op.

### Params: `params` vs `param_sets`

- **`params`** — a single op-specific set applied to every combo.
- **`param_sets`** — a *list* of sets, swept as an extra axis (one benchmark per entry × the rest of the matrix). Use it to compare, e.g., interpolation types or kernel sizes. `params` and `param_sets` are mutually exclusive.

The active param set is encoded in the benchmark name as `.../params:k1=v1,k2=v2`, so swept cases stay uniquely named and the values flow through to the JSON output and to the CSV's `params` column (comma-quoted). Reading a param in an adapter: `ctx.param<T>("key", fallback)`.

## Project layout

```
common/     harness: config parsing, tensor setup, the op registry, the runner
src/        bench_main.cpp (entry point) and op adapters under src/ops/
config/     example sweep configs (smoke.json, example.json)
scripts/    result post-processing (json2csv.py, plot_results.py)
```

## Adding an op

Each RPP op has a distinct C signature, so each needs a small adapter under `src/ops/bench_<name>.cpp`:

1. Subclass `OpAdapter` (see [common/bench_registry.hpp](common/bench_registry.hpp)).
2. In `setup()` allocate any op-specific param tensors (use `bench_pinned_alloc` so they work on HIP); in `run()` call the `rppt_*` function; free in `teardown()`.
3. Optionally override `supportedDtypes()` / `supportedLayouts()` and, for filter-style ops, `srcOffsetBytes()` / `srcAdditionalStride()` to request the halo padding HIP kernels need.
4. Register with `REGISTER_RPP_BENCH("config_name", AdapterClass)`.

The CMake glob picks up new files under `src/ops/`; re-run `cmake --build`. See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow.

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for build, style, and PR guidelines.

## License

Released under the [MIT License](LICENSE).

[Google Benchmark]: https://github.com/google/benchmark
