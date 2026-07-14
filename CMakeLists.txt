# ============================================================================
# rpp-mark - config-driven micro-benchmarks for RPP operators.
#
# Standalone project: it consumes an *installed* RPP via find_package(rpp) and
# fetches Google Benchmark + nlohmann/json at configure time (neither needs to
# be installed on the system).
#
# Usage:
#   cmake -S . -B build                                   # uses $ROCM_PATH or /opt/rocm
#   cmake -S . -B build -DROCM_PATH=/opt/rocm-6.2.0       # a specific ROCm
#   cmake -S . -B build -Drpp_DIR=/path/to/lib/cmake/rpp  # or point at the config
#   cmake --build build -j
#   ./build/rpp_bench --config=config/example.json
# ============================================================================
cmake_minimum_required(VERSION 3.20)

project(rpp_benchmarks LANGUAGES CXX)

# ----------------------------------------------------------------------------
# Build settings. The benchmarks only issue host-side HIP calls, so the default
# host compiler is fine - no ROCm/amdclang toolchain required to build.
# ----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)                       # compile_commands.json for clangd

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)     # benchmarks want optimized code
endif()

# ----------------------------------------------------------------------------
# Find the installed RPP.
# ROCm location precedence: -DROCM_PATH  >  $ENV{ROCM_PATH}  >  /opt/rocm.
# RPP installs its package config to <prefix>/lib/cmake/rpp.
# ----------------------------------------------------------------------------
if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
        set(ROCM_PATH $ENV{ROCM_PATH})
    else()
        set(ROCM_PATH /opt/rocm)
    endif()
endif()
set(ROCM_PATH "${ROCM_PATH}" CACHE PATH "ROCm installation path")
list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}" "${ROCM_PATH}/lib/cmake")

find_package(rpp REQUIRED)
message(STATUS "rpp-mark: found RPP (${rpp_BACKEND_TYPE} backend) at ${rpp_INCLUDE_DIR}")

# ----------------------------------------------------------------------------
# Third-party dependencies, fetched at configure time.
# ----------------------------------------------------------------------------
include(FetchContent)

set(BENCHMARK_ENABLE_TESTING     OFF CACHE BOOL "" FORCE)   # skip benchmark's own tests
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)   # don't pull gtest
set(BENCHMARK_ENABLE_INSTALL     OFF CACHE BOOL "" FORCE)
set(JSON_BuildTests              OFF CACHE INTERNAL "")

FetchContent_Declare(benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.9.1                                   # pin a released tag
    GIT_SHALLOW    TRUE)
FetchContent_Declare(nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE)
FetchContent_MakeAvailable(benchmark nlohmann_json)

# ----------------------------------------------------------------------------
# Benchmark executable.
#
# The harness lives in common/ (config parsing, tensor setup, registry, runner)
# and each op adapter is one .cpp under src/ops/. The glob keeps op registration
# decentralized - re-run cmake after adding a file. We supply our own main()
# (src/bench_main.cpp), so we link the plain benchmark lib, not benchmark_main.
# ----------------------------------------------------------------------------
file(GLOB RPP_BENCH_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/common/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/ops/*.cpp")

add_executable(rpp_bench ${RPP_BENCH_SOURCES})

target_include_directories(rpp_bench PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/common")

target_link_libraries(rpp_bench PRIVATE
    rpp::rpp
    benchmark::benchmark
    nlohmann_json::nlohmann_json)

# ----------------------------------------------------------------------------
# HIP backend: RPP's public headers pull in HIP, and setup code calls host-side
# HIP APIs (hipMalloc/hipMemcpy). rpp-config already runs find_dependency(HIP),
# so hip::host is available; rocRAND fills input tensors directly in device
# memory, avoiding a host fill + H2D copy per case.
# ----------------------------------------------------------------------------
if(rpp_BACKEND_TYPE STREQUAL "HIP")
    find_package(hip REQUIRED)
    find_package(rocrand REQUIRED)
    target_link_libraries(rpp_bench PRIVATE hip::host roc::rocrand)
    target_compile_definitions(rpp_bench PRIVATE RPP_BENCH_HIP=1)
else()
    target_compile_definitions(rpp_bench PRIVATE RPP_BENCH_HIP=0)
endif()
