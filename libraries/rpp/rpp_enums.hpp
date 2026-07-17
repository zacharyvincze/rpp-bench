/**
 * @file rpp_enums.hpp
 * @brief Translation between the neutral core vocabulary and RPP's own enums.
 *
 * The core speaks core/bench_types.hpp (Backend/DataType/Layout); RPP speaks
 * Rppt* enums. This is the single boundary where the two are mapped, plus the
 * RPP-typed helpers the adapters use (layout_info, dtype_size, interpolation
 * parsing). It is the RPP module's counterpart to the old config/bench_enums.hpp.
 */
#ifndef RPP_BENCH_RPP_ENUMS_HPP
#define RPP_BENCH_RPP_ENUMS_HPP

#include <rpp/rppdefs.h>
#include "core/bench_types.hpp"

#include <stdexcept>
#include <string>

namespace rppbench {

// ---- neutral <-> RPP scalar conversions ----

inline RpptDataType to_rpp_dtype(DataType t) {
    switch (t) {
    case DataType::U8:
        return RpptDataType::U8;
    case DataType::F32:
        return RpptDataType::F32;
    case DataType::F16:
        return RpptDataType::F16;
    case DataType::I8:
        return RpptDataType::I8;
    }
    throw std::runtime_error("unknown neutral dtype");
}

inline DataType to_neutral_dtype(RpptDataType t) {
    switch (t) {
    case RpptDataType::U8:
        return DataType::U8;
    case RpptDataType::F32:
        return DataType::F32;
    case RpptDataType::F16:
        return DataType::F16;
    case RpptDataType::I8:
        return DataType::I8;
    default:
        throw std::runtime_error("unsupported RpptDataType");
    }
}

inline RppBackend to_rpp_backend(Backend b) {
    return b == Backend::Hip ? RppBackend::RPP_HIP_BACKEND : RppBackend::RPP_HOST_BACKEND;
}

inline Backend to_neutral_backend(RppBackend b) {
    return b == RppBackend::RPP_HIP_BACKEND ? Backend::Hip : Backend::Host;
}

// ---- RPP-typed helpers used by the adapters and tensor code ----

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

#endif // RPP_BENCH_RPP_ENUMS_HPP
