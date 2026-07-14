// ============================================================================
// bench_enums.hpp - string <-> RPP enum helpers used across the harness.
//
// The config file speaks in human strings ("U8", "PKD3", "BILINEAR"); the RPP
// API speaks in enums. Keep the mapping in one place so config parsing, name
// encoding and adapters all agree.
// ============================================================================
#ifndef RPP_BENCH_ENUMS_HPP
#define RPP_BENCH_ENUMS_HPP

#include <rpp/rppdefs.h>
#include <stdexcept>
#include <string>

namespace rppbench {

// RPP's layout naming convention (see CLAUDE.md):
//   PKD3 - 3-channel packed/interleaved (NHWC, c=3)
//   PLN3 - 3-channel planar             (NCHW, c=3)
//   PLN1 - 1-channel planar             (NCHW, c=1)
enum class Layout : uint8_t { PKD3, PLN3, PLN1 };

struct LayoutInfo {
    RpptLayout rpptLayout;
    int channels;
};

inline LayoutInfo layout_info(Layout l) {
    switch (l) {
    case Layout::PKD3:
        return {RpptLayout::NHWC, 3};
    case Layout::PLN3:
        return {RpptLayout::NCHW, 3};
    case Layout::PLN1:
        return {RpptLayout::NCHW, 1};
    }
    throw std::runtime_error("unknown layout");
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

inline RpptDataType parse_dtype(const std::string &s) {
    if (s == "U8")
        return RpptDataType::U8;
    if (s == "F32")
        return RpptDataType::F32;
    if (s == "F16")
        return RpptDataType::F16;
    if (s == "I8")
        return RpptDataType::I8;
    throw std::runtime_error("unknown dtype '" + s + "' (expected U8/F32/F16/I8)");
}

inline const char *dtype_name(RpptDataType t) {
    switch (t) {
    case RpptDataType::U8:
        return "U8";
    case RpptDataType::F32:
        return "F32";
    case RpptDataType::F16:
        return "F16";
    case RpptDataType::I8:
        return "I8";
    default:
        return "?";
    }
}

inline int dtype_size(RpptDataType t) {
    switch (t) {
    case RpptDataType::U8:
    case RpptDataType::I8:
        return 1;
    case RpptDataType::F16:
        return 2;
    case RpptDataType::F32:
        return 4;
    default:
        throw std::runtime_error("unsupported dtype size");
    }
}

inline RppBackend parse_backend(const std::string &s) {
    if (s == "HOST")
        return RppBackend::RPP_HOST_BACKEND;
    if (s == "HIP")
        return RppBackend::RPP_HIP_BACKEND;
    throw std::runtime_error("unknown backend '" + s + "' (expected HOST/HIP)");
}

inline RpptInterpolationType parse_interpolation(const std::string &s) {
    if (s == "NEAREST_NEIGHBOR")
        return RpptInterpolationType::NEAREST_NEIGHBOR;
    if (s == "BILINEAR")
        return RpptInterpolationType::BILINEAR;
    if (s == "BICUBIC")
        return RpptInterpolationType::BICUBIC;
    if (s == "LANCZOS")
        return RpptInterpolationType::LANCZOS;
    if (s == "GAUSSIAN")
        return RpptInterpolationType::GAUSSIAN;
    if (s == "TRIANGULAR")
        return RpptInterpolationType::TRIANGULAR;
    throw std::runtime_error("unknown interpolation '" + s + "'");
}

} // namespace rppbench

#endif // RPP_BENCH_ENUMS_HPP
