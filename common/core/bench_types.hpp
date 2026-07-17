/**
 * @file bench_types.hpp
 * @brief Library-neutral vocabulary types for the benchmark core.
 *
 * The core describes a sweep in these types alone - no library (RPP) or device
 * (HIP) headers. Each library module translates them to its own enums at its
 * boundary (see libraries/rpp/rpp_enums.hpp). This is the file that lets the
 * config, matrix expansion, name encoding, and runner stay dependency-free.
 */
#ifndef RPP_BENCH_CORE_TYPES_HPP
#define RPP_BENCH_CORE_TYPES_HPP

#include <cstdint>
#include <stdexcept>
#include <string>

namespace rppbench {

// Where an op runs. Mirrors RPP's own HOST/HIP "backend" meaning; a library may
// expose one or both. Kept distinct from the *library* (which implementation),
// which is a separate scalar on a BenchPoint.
enum class Backend : uint8_t { Host, Hip };

// Element data type of a tensor. The neutral counterpart of a library's own
// dtype enum.
enum class DataType : uint8_t { U8, F32, F16, I8 };

// Channel layout convention (see CLAUDE.md):
//   PKD3 - 3-channel packed/interleaved (NHWC, c=3)
//   PLN3 - 3-channel planar             (NCHW, c=3)
//   PLN1 - 1-channel planar             (NCHW, c=1)
// Shared verbatim with the library modules (they map it to their own layout
// enum), so it lives in the neutral core rather than being duplicated.
enum class Layout : uint8_t { PKD3, PLN3, PLN1 };

inline Backend parse_backend(const std::string &s) {
    if (s == "HOST")
        return Backend::Host;
    if (s == "HIP")
        return Backend::Hip;
    throw std::runtime_error("unknown backend '" + s + "' (expected HOST/HIP)");
}

inline const char *backend_name(Backend b) {
    switch (b) {
    case Backend::Host:
        return "HOST";
    case Backend::Hip:
        return "HIP";
    }
    return "?";
}

inline DataType parse_dtype(const std::string &s) {
    if (s == "U8")
        return DataType::U8;
    if (s == "F32")
        return DataType::F32;
    if (s == "F16")
        return DataType::F16;
    if (s == "I8")
        return DataType::I8;
    throw std::runtime_error("unknown dtype '" + s + "' (expected U8/F32/F16/I8)");
}

inline const char *dtype_name(DataType t) {
    switch (t) {
    case DataType::U8:
        return "U8";
    case DataType::F32:
        return "F32";
    case DataType::F16:
        return "F16";
    case DataType::I8:
        return "I8";
    }
    return "?";
}

inline Layout parse_layout(const std::string &s) {
    if (s == "PKD3")
        return Layout::PKD3;
    if (s == "PLN3")
        return Layout::PLN3;
    if (s == "PLN1")
        return Layout::PLN1;
    throw std::runtime_error("unknown layout '" + s + "' (expected PKD3/PLN3/PLN1)");
}

inline const char *layout_name(Layout l) {
    switch (l) {
    case Layout::PKD3:
        return "PKD3";
    case Layout::PLN3:
        return "PLN3";
    case Layout::PLN1:
        return "PLN1";
    }
    return "?";
}

} // namespace rppbench

#endif // RPP_BENCH_CORE_TYPES_HPP
