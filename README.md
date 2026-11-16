# rpp-bench

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Config-driven micro-benchmarks for [RPP](https://github.com/ROCm/rpp) (ROCm Performance Primitives) operators, built on [Google Benchmark]. The benchmark matrix (ops × backends × dtypes × layouts × batch × sizes × params) lives in a runtime JSON file, so exploring new cases means editing a config, not recompiling.

Needs an installed RPP discoverable via `find_package(rpp)` (typically from ROCm under `/opt/rocm`), CMake ≥ 3.20, and a C++17 host compiler. The active backend (HOST vs HIP) is inherited from the installed RPP. Google Benchmark and nlohmann/json are fetched automatically at configure time.

## Quick start

```shell
git clone https://github.com/zacharyvincze/rpp-bench.git
cd rpp-bench
cmake -S . -B build
cmake --build build -j
./build/rpp_bench --config=config/smoke.json --progress
```

## Build

```shell
cmake -S . -B build            # uses -DROCM_PATH / $ROCM_PATH / /opt/rocm, in that order
cmake --build build -j
```

Point at a specific RPP with `-DROCM_PATH=/opt/rocm-x.y.z` or `-Drpp_DIR=/path/to/lib/cmake/rpp`.

## Run

```shell
./build/rpp_bench --config=config/example.json      # run a sweep (also saved under results/)
./build/rpp_bench --list-ops                         # list compiled-in op adapters
./build/rpp_bench --help                             # all options

# Choose the result file yourself instead of the default name.
./build/rpp_bench --config=config/example.json --benchmark_out=my_run.json
```

Configs ship in [config/](config/); see [config/README.md](config/README.md) for what each one covers and when to use it. Use `--benchmark_filter=<regex>` (standard Google Benchmark substring match; leading `-` inverts) to select cases, and `--progress` for a compact live progress line.

## Results and run metadata

Every run writes its results as JSON to `results/<config-stem>_<YYYYmmdd-HHMMSS>_<host>.json` (git-ignored), unless you pass `--benchmark_out=<file>` (with `--benchmark_out_format` choosing json/csv) or `--no-results-file`. A run that executes no cases (a filter that matches nothing, `--benchmark_list_tests`) writes nothing.

Each JSON file describes the run that produced it, so results stay comparable across machines and builds. Next to Google Benchmark's own fields (host, CPU count and caches, load, date), the `context` block records:

| Key | What |
|---|---|
| `rpp_bench_schema` | Version of these keys and of the benchmark-name format |
| `rpp_bench_commit` | rpp-bench commit it was built from (`-dirty` if there were uncommitted changes) |
| `rpp_version`, `rpp_backend` | RPP version from the headers it was compiled against, and HOST/HIP |
| `rpp_library` | The `librpp` the loader actually picked up (matters when several are installed) |
| `rocm_path`, `rocm_version` | ROCm install it was built against |
| `hip_runtime_version`, `hip_driver_version` | HIP versions at run time (HIP builds) |
| `gpu_*` | Device name, gfx arch, CU count, max clocks, VRAM, PCI bus ID, and `gpu_perf_level` (the amdgpu clock policy; `auto` means clocks float, so timings are noisier) |
| `cpu_model`, `os_kernel` | CPU model and kernel release |
| `command_line` | The exact invocation |
| `config_path`, `config_hash` | Sweep config and a hash of its content (whitespace-insensitive) |
| `env_*` | `OMP_NUM_THREADS`, `HIP_VISIBLE_DEVICES`, `ROCR_VISIBLE_DEVICES`, `LD_LIBRARY_PATH`, when set |

The config itself is stored under the top-level `rpp_bench_config` key, so you can reproduce a run after the config file has changed.

## Graphing results

Post-process result files with the scripts in [scripts/](scripts/) (Python 3; `plot.py` needs `matplotlib` + `numpy`). The dependencies are pinned in [pyproject.toml](pyproject.toml)/`uv.lock`; with [uv](https://docs.astral.sh/uv/), `uv sync` creates `.venv/` and `uv run` uses it automatically.

```shell
./build/rpp_bench --config=config/example.json --benchmark_out=results.json

uv sync                                              # once: create .venv with matplotlib + numpy
uv run scripts/plot.py results.json -o plots         # -o is a directory (default: <input_dir>/per_op)
uv run scripts/json2csv.py results.json -o results.csv   # CSV (stdlib only; default: stdout)
```

`plot.py` writes one horizontal bar chart per operator into the output directory (one bar per fully-resolved config, sorted by runtime and coloured by dtype). Add `--per-image` to normalize runtime by batch, or `--linear` for a linear x-axis.

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
| Color | `blend` | ✅ | `alpha` — two-source (alpha-blend) |
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
| Effects | `non_linear_blend` | ✅ | `std_dev` — two-source (Gaussian-mask blend) |
| Effects | `water` | ✅ | `amplitude_x/y`, `frequency_x/y`, `phase_x/y` — six per-image float tensors |
| Effects | `ricap` | ✅ | no params — patches output from 4 crop regions of permuted batch images; requires batch > 1 (config overrides `batch_sizes`) |
| Effects | `pixelate` | ✅ | `pixelation_percentage` (0–100) — needs an F32 intermediate scratch buffer (n × nStride × sizeof(float)) |
| Effects | `cutout_dropout` | ✅ | `max_boxes` — erases N boxes/image with solid colour (black) |
| Effects | `grid_dropout` | ✅ | `boxes`, `hole_w`, `hole_h` — erases a regular grid of holes |
| Effects | `random_erase` | ✅ | no params — one box/image filled from a tiled noise buffer |
| Effects | `coarse_dropout` | ✅ | `max_boxes` — erases N non-overlapping boxes/image (filled internally) |
| Effects | `gaussian_noise_voxel` | ✅ | `mean`, `stddev`, `seed` — 3D voxel (generic 5D descriptor); F32 only |
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
| Geometric | `remap` | ✅ | `interpolation` — per-pixel row/col remap tables (identity map); NEAREST_NEIGHBOR/BILINEAR |
| Geometric | `lens_correction` | ✅ | no params — barrel-distortion correction; camera matrix + distortion coeffs seeded internally |
| Geometric | `transpose` | ✅ | `perm` — permute spatial axes (generic ND descriptor) |
| Geometric | `slice` | ✅ | `fraction` — sub-tensor extract (generic ND descriptor); U8/F32 |
| Geometric | `concat` | ✅ | `axis` — two-source concat (generic ND descriptor) |
| Geometric | `phase` | ✅ | no params — two-source |
| Geometric | `crop_and_patch` | ✅ | no params — two-source; fixed centred half-size crop/patch window |
| Geometric | `flip_voxel` | ✅ | `horizontal`, `vertical`, `depth_flip`, `depth` — 3D voxel (generic 5D descriptor); U8/F32 |
| Geometric | `jpeg_compression_distortion` | ☐ | |
| Geometric | `fisheye` | ☐ | |
| Data exchange | `copy` | ✅ | no params; memory-bandwidth baseline |
| Data exchange | `channel_permute` | ✅ | `perm0`, `perm1`, `perm2` — RGB only (PKD3/PLN3) |
| Data exchange | `color_to_greyscale` | ☐ | dst is 1-channel |
| Data exchange | `yuv_to_rgb` | ☐ | separate Y/UV planes |
| Data exchange | `yuv_to_rgb_cubic_v` | ☐ | separate Y/UV planes |
| Data exchange | `yuv_to_rgb_linear_v` | ☐ | separate Y/UV planes |
| Arithmetic | `add_scalar` | ✅ | `add`, `depth` — 3D voxel (generic 5D descriptor); F32 only |
| Arithmetic | `subtract_scalar` | ✅ | `subtract`, `depth` — 3D voxel (generic 5D descriptor); F32 only |
| Arithmetic | `multiply_scalar` | ✅ | `multiply`, `depth` — 3D voxel (generic 5D descriptor); F32 only |
| Arithmetic | `fused_multiply_add_scalar` | ✅ | `multiply`, `add`, `depth` — 3D voxel (generic 5D descriptor); F32 only |
| Arithmetic | `magnitude` | ✅ | no params — two-source |
| Arithmetic | `log` | ✅ | no params — generic ND descriptor; F32 only |
| Arithmetic | `log1p` | ☐ | |
| Arithmetic | `tensor_add_tensor` | ✅ | two-source (generic ND descriptor); F32 only |
| Arithmetic | `tensor_subtract_tensor` | ✅ | two-source (generic ND descriptor); F32 only |
| Arithmetic | `tensor_multiply_tensor` | ✅ | two-source (generic ND descriptor); F32 only |
| Arithmetic | `tensor_divide_tensor` | ✅ | two-source (generic ND descriptor); F32 only |
| Statistical | `tensor_sum` | ✅ | |
| Statistical | `tensor_min` | ✅ | Reduction — channel-wise + overall min; no dst image |
| Statistical | `tensor_max` | ✅ | Reduction — channel-wise + overall max; no dst image |
| Statistical | `tensor_mean` | ✅ | Reduction — channel-wise + total mean (F32 output); no dst image |
| Statistical | `tensor_stddev` | ✅ | Reduction — channel-wise + total stddev (F32 output); takes a `meanTensor` input; no dst image |
| Statistical | `normalize` | ✅ | `axisMask`, `scale`, `shift` — generic ND descriptor; mean/stddev computed internally; F32 only |
| Statistical | `threshold` | ✅ | `min`, `max` — per-channel cutoffs (always float, size batch×channels); binary mask output |
| Bitwise | `bitwise_not` | ✅ | no params — U8 only |
| Bitwise | `bitwise_and` | ✅ | two-source, no params — U8 only |
| Bitwise | `bitwise_or` | ✅ | two-source, no params — U8 only |
| Bitwise | `bitwise_xor` | ✅ | two-source, no params — U8 only |
| Bitwise | `tensor_and_tensor` | ✅ | two-source (generic ND descriptor) — U8 only |
| Bitwise | `tensor_or_tensor` | ✅ | two-source (generic ND descriptor) — U8 only |
| Bitwise | `tensor_xor_tensor` | ✅ | two-source (generic ND descriptor) — U8 only |
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

See [CONTRIBUTING.md](CONTRIBUTING.md) for build, style, and PR guidelines, including how to add an op.

## License

Released under the [MIT License](LICENSE).

[Google Benchmark]: https://github.com/google/benchmark
