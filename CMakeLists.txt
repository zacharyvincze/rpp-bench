# ============================================================================
# rpp-bench - config-driven micro-benchmarks for imaging-library operators.
#
# Standalone project: it consumes an *installed* RPP via find_package(rpp) and
# fetches Google Benchmark + nlohmann/json at configure time (neither needs to
# be installed on the system).
#
# Structure: a dependency-free core (common/) plus one optional module per
# library (libraries/<name>/), each built as an OBJECT library and linked into
# the executable when its dependency is found. Today the only module is RPP.
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
# Third-party dependencies, fetched at configure time. These back the neutral
# core (they are not tied to any imaging library).
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
# Neutral core: config parsing, the Library/BenchCase interfaces + registry,
# matrix expansion, name encoding, CLI, and the --progress reporters. Depends on
# nobody's imaging-library headers - builds with a plain C++17 host compiler.
# Headers are included path-qualified from the common/ root (e.g.
# "core/bench_runner.hpp", "cli/bench_cli.hpp").
# ----------------------------------------------------------------------------
file(GLOB BENCH_CORE_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/common/core/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common/cli/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/common/cli/progress/*.cpp")

add_library(bench_core STATIC ${BENCH_CORE_SOURCES})
target_include_directories(bench_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/common")
target_link_libraries(bench_core PUBLIC
    benchmark::benchmark
    nlohmann_json::nlohmann_json)

# ----------------------------------------------------------------------------
# Library modules. Each is optional and self-contained: an OBJECT library that
# links bench_core plus its own dependency, and self-registers with the core at
# static-init time. We use OBJECT libraries so the static registrations are never
# stripped from the final link (a plain static archive would drop them). New
# modules follow the same shape under libraries/<name>/.
# ----------------------------------------------------------------------------
set(RPP_BENCH_MODULE_OBJECTS "")   # $<TARGET_OBJECTS:...> for each enabled module
set(RPP_BENCH_MODULE_LIBS "")      # imaging-library link deps for the executable

# ---- RPP module ----
# ROCm location precedence: -DROCM_PATH  >  $ENV{ROCM_PATH}  >  /opt/rocm.
if(NOT DEFINED ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH})
        set(ROCM_PATH $ENV{ROCM_PATH})
    else()
        set(ROCM_PATH /opt/rocm)
    endif()
endif()
set(ROCM_PATH "${ROCM_PATH}" CACHE PATH "ROCm installation path")
list(APPEND CMAKE_PREFIX_PATH "${ROCM_PATH}" "${ROCM_PATH}/lib/cmake")

find_package(rpp CONFIG)
if(rpp_FOUND)
    message(STATUS "rpp-bench: found RPP (${rpp_BACKEND_TYPE} backend) at ${rpp_INCLUDE_DIR}")

    file(GLOB RPP_MODULE_SOURCES CONFIGURE_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/libraries/rpp/*.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/libraries/rpp/ops/*.cpp")

    add_library(rpp_module OBJECT ${RPP_MODULE_SOURCES})
    target_include_directories(rpp_module PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/common"
        "${CMAKE_CURRENT_SOURCE_DIR}/libraries")
    target_link_libraries(rpp_module PRIVATE bench_core rpp::rpp)

    # HIP backend: RPP's public headers pull in HIP, and setup code calls
    # host-side HIP APIs (hipMalloc/hipStream). rpp-config already runs
    # find_dependency(HIP), so hip::host is available; rocRAND fills input tensors
    # directly in device memory, avoiding a host fill + H2D copy per case.
    if(rpp_BACKEND_TYPE STREQUAL "HIP")
        find_package(hip REQUIRED)
        find_package(rocrand REQUIRED)
        target_link_libraries(rpp_module PRIVATE hip::host roc::rocrand)
        # GPU_SUPPORT gates RPP's GPU-capable, backend-dispatch API overloads in
        # the installed headers (e.g. rppt_erode / rppt_dilate). It must match how
        # the HIP RPP was built for those declarations to be visible. RPP_BENCH_HIP
        # is scoped to this module - the core never sees it.
        target_compile_definitions(rpp_module PRIVATE RPP_BENCH_HIP=1 GPU_SUPPORT=1)
        list(APPEND RPP_BENCH_MODULE_LIBS rpp::rpp hip::host roc::rocrand)
    else()
        target_compile_definitions(rpp_module PRIVATE RPP_BENCH_HIP=0)
        list(APPEND RPP_BENCH_MODULE_LIBS rpp::rpp)
    endif()

    list(APPEND RPP_BENCH_MODULE_OBJECTS $<TARGET_OBJECTS:rpp_module>)
else()
    message(WARNING "rpp-bench: RPP not found - building the core without the RPP module. "
                    "The binary will register no libraries. Set -DROCM_PATH or -Drpp_DIR.")
endif()

# ----------------------------------------------------------------------------
# Executable. Links the neutral core plus every enabled module's objects (as
# object files, so their static self-registrations survive the link) and the
# corresponding imaging-library deps. We supply our own main (src/bench_main.cpp),
# so we link the plain benchmark lib, not benchmark_main.
# ----------------------------------------------------------------------------
add_executable(rpp_bench src/bench_main.cpp ${RPP_BENCH_MODULE_OBJECTS})
target_link_libraries(rpp_bench PRIVATE bench_core ${RPP_BENCH_MODULE_LIBS})
