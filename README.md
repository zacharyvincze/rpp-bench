# rpp-mark

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Config-driven micro-benchmarks for [RPP](https://github.com/ROCm/rpp) (ROCm Performance Primitives) operators, built on [Google Benchmark].

- [Requirements](#requirements)
- [Build](#build)
- [Run](#run)
- [Filtering benchmarks](#filtering-benchmarks)
- [Analysing results](#analysing-results)
- [Config format](#config-format)
- [Project layout](#project-layout)
- [Operators](#operators)
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

## Adding an op

Each RPP op has a distinct C signature, so each needs a small adapter under `src/ops/bench_<name>.cpp`:

1. Subclass `OpAdapter` (see [common/harness/bench_registry.hpp](common/harness/bench_registry.hpp)).
2. In `setup()` allocate any op-specific param tensors (use `bench_pinned_alloc` so they work on HIP); in `run()` call the `rppt_*` function; free in `teardown()`.
3. Optionally override `supportedDtypes()` / `supportedLayouts()` / `supportedBackends()` to constrain the sweep (e.g. RGB-only or HIP-only ops), and, for filter-style ops, `srcOffsetBytes()` / `srcAdditionalStride()` to request the halo padding HIP kernels need.
4. Register with `REGISTER_RPP_BENCH("config_name", AdapterClass)`.

The CMake glob picks up new files under `src/ops/`; re-run `cmake --build`. See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow.

## Operators

| Category | Operator | Done | Params / notes |
|---|---|:--:|---|
| Color | `brightness` | ✅ | `alpha`, `beta` |
| Color | `gamma_correction` | ✅ | `gamma` |
| Color | `contrast` | ✅ | `factor`, `center` |
| Color | `exposure` | ✅ | `factor` |
| Color | `hue` | ✅ | `hue` — RGB only (PKD3/PLN3) |
| Color | `saturation` | ✅ | `saturation` — RGB only (PKD3/PLN3) |
| Color | `color_temperature` | ✅ | `adjustment` |
| Color | `color_twist` | ✅ | `brightness`, `contrast`, `hue`, `saturation` |
| Color | `color_jitter` | ✅ | `brightness`, `contrast`, `hue`, `saturation` |
| Color | `histogram_equalize` | ✅ | no params — U8 only; HOST only (HIP kernel leaks an internal scratch buffer) |
| Color | `color_cast` | ✅ | `r`, `g`, `b`, `alpha` — RGB only (PKD3/PLN3) |
| Color | `lut` | ✅ | identity 65536-entry table — U8/I8 only |
| Color | `blend` | ☐ | two-source |
| Effects | `vignette` | ✅ | `intensity` |
| Effects | `solarize` | ✅ | `threshold` |
| Effects | `posterize` | ✅ | `level_bits` |
| Effects | `channel_dropout` | ✅ | per-channel keep/drop mask |
| Effects | `fog` | ✅ | `intensity`, `grey` |
| Effects | `snow` | ✅ | `brightness_coefficient`, `threshold`, `dark_mode` |
| Effects | `gaussian_noise` | ✅ | `mean`, `std_dev`, `seed` |
| Effects | `shot_noise` | ✅ | `factor`, `seed` |
| Effects | `salt_and_pepper_noise` | ✅ | `noise_probability`, `salt_probability`, `salt_value`, `pepper_value`, `seed` |
| Effects | `jitter` | ✅ | `kernel_size`, `seed` |
| Effects | `gridmask` | ✅ | `tile_width`, `grid_ratio`, `grid_angle`, `translate_x`, `translate_y` |
| Effects | `spatter` | ✅ | `r`, `g`, `b` (RpptRGB by value) — RGB only (PKD3/PLN3) |
| Effects | `rain` | ✅ | `rain_percentage`, `rain_width`, `rain_height`, `slant_angle`, `alpha` — stages via RPP host scratch (~0.4 GB × batch); keep batch modest on low-RAM hosts |
| Effects | `erase` | ✅ | one centred box/image, RGB colour — 3-channel only |
| Effects | `glitch` | ✅ | `r_x`/`r_y`/`g_x`/`g_y`/`b_x`/`b_y` channel offsets — 3-channel only |
| Effects | `non_linear_blend` | ☐ | two-source |
| Effects | `water` | ☐ | |
| Effects | `ricap` | ☐ | |
| Effects | `pixelate` | ☐ | external scratch buffer |
| Effects | `cutout_dropout` | ☐ | |
| Effects | `grid_dropout` | ☐ | |
| Effects | `random_erase` | ☐ | |
| Effects | `coarse_dropout` | ☐ | |
| Effects | `gaussian_noise_voxel` | ☐ | generic 3D descriptor |
| Filter | `gaussian_blur` | ✅ | `kernel_size`, `std_dev` — maps to `rppt_gaussian_filter` |
| Filter | `box_filter` | ✅ | `kernel_size` |
| Filter | `median_filter` | ✅ | `kernel_size` |
| Filter | `emboss` | ✅ | `kernel_size`, `strength` |
| Filter | `sobel_filter` | ✅ | `sobel_type`, `kernel_size` — single-channel (PLN1); dst is 1-channel |
| Morphological | `erode` | ✅ | `kernel_size` — HIP only |
| Morphological | `dilate` | ✅ | `kernel_size` — HIP only |
| Geometric | `flip` | ✅ | `horizontal`, `vertical` |
| Geometric | `resize` | ✅ | `interpolation` (+ `dst_sizes`) |
| Geometric | `rotate` | ✅ | `angle`, `interpolation` |
| Geometric | `warp_affine` | ✅ | `angle`, `interpolation` — rotation affine built from `angle` |
| Geometric | `warp_perspective` | ✅ | `angle`, `interpolation` — rotation homography built from `angle` |
| Geometric | `crop` | ✅ | crop window from `dst_sizes` (anchored top-left) |
| Geometric | `crop_mirror_normalize` | ✅ | `offset`, `multiplier`, `mirror` (+ `dst_sizes`) |
| Geometric | `resize_mirror_normalize` | ✅ | `interpolation`, `mean`, `std_dev`, `mirror` (+ `dst_sizes`) |
| Geometric | `resize_crop_mirror` | ✅ | `interpolation`, `mirror` (+ `dst_sizes`) |
| Geometric | `remap` | ☐ | remap tables |
| Geometric | `lens_correction` | ☐ | remap tables + matrices |
| Geometric | `transpose` | ☐ | generic 3D descriptor |
| Geometric | `slice` | ☐ | generic 3D descriptor |
| Geometric | `concat` | ☐ | generic 3D descriptor |
| Geometric | `phase` | ☐ | two-source |
| Geometric | `crop_and_patch` | ☐ | two-source |
| Geometric | `flip_voxel` | ☐ | generic 3D descriptor |
| Geometric | `jpeg_compression_distortion` | ☐ | |
| Geometric | `fisheye` | ☐ | |
| Data exchange | `copy` | ✅ | no params; memory-bandwidth baseline |
| Data exchange | `channel_permute` | ✅ | `perm0`, `perm1`, `perm2` — RGB only (PKD3/PLN3) |
| Data exchange | `color_to_greyscale` | ☐ | dst is 1-channel |
| Data exchange | `yuv_to_rgb` | ☐ | separate Y/UV planes |
| Data exchange | `yuv_to_rgb_cubic_v` | ☐ | separate Y/UV planes |
| Data exchange | `yuv_to_rgb_linear_v` | ☐ | separate Y/UV planes |
| Arithmetic | `add_scalar` | ☐ | |
| Arithmetic | `subtract_scalar` | ☐ | |
| Arithmetic | `multiply_scalar` | ☐ | |
| Arithmetic | `fused_multiply_add_scalar` | ☐ | |
| Arithmetic | `magnitude` | ☐ | two-source |
| Arithmetic | `log` | ☐ | |
| Arithmetic | `log1p` | ☐ | |
| Arithmetic | `tensor_add_tensor` | ☐ | two-source, broadcast modes |
| Arithmetic | `tensor_subtract_tensor` | ☐ | two-source, broadcast modes |
| Arithmetic | `tensor_multiply_tensor` | ☐ | two-source, broadcast modes |
| Arithmetic | `tensor_divide_tensor` | ☐ | two-source, broadcast modes |
| Statistical | `tensor_sum` | ☐ | |
| Statistical | `tensor_min` | ☐ | |
| Statistical | `tensor_max` | ☐ | |
| Statistical | `tensor_mean` | ☐ | |
| Statistical | `tensor_stddev` | ☐ | |
| Statistical | `normalize` | ☐ | |
| Statistical | `threshold` | ☐ | |
| Bitwise | `bitwise_not` | ✅ | no params — U8 only |
| Bitwise | `bitwise_and` | ☐ | two-source |
| Bitwise | `bitwise_or` | ☐ | two-source |
| Bitwise | `bitwise_xor` | ☐ | two-source |
| Bitwise | `tensor_and_tensor` | ☐ | two-source |
| Bitwise | `tensor_or_tensor` | ☐ | two-source |
| Bitwise | `tensor_xor_tensor` | ☐ | two-source |
| Audio | `non_silent_region_detection` | ☐ | |
| Audio | `to_decibels` | ☐ | |
| Audio | `pre_emphasis_filter` | ☐ | |
| Audio | `down_mixing` | ☐ | |
| Audio | `spectrogram` | ☐ | |
| Audio | `mel_filter_bank` | ☐ | |
| Audio | `resample` | ☐ | |
| Audio | `audio_tensor_add_tensor` | ☐ | |
| Audio | `audio_tensor_mul_scalar` | ☐ | |

## Contributing

Contributions are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md) for build, style, and PR guidelines.

## License

Released under the [MIT License](LICENSE).

[Google Benchmark]: https://github.com/google/benchmark
