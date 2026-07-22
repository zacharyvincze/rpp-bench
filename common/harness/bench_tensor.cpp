#include "harness/bench_tensor.hpp"

#include <cstdlib>
#include <cstring>
#include <stdexcept>

#if RPP_BENCH_HIP
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocrand/rocrand.h>
#define HIP_CHECK(call)                                                                            \
    do {                                                                                           \
        hipError_t _e = (call);                                                                    \
        if (_e != hipSuccess)                                                                      \
            throw std::runtime_error(std::string("HIP error: ") + hipGetErrorString(_e));          \
    } while (0)
#define ROCRAND_CHECK(call)                                                                        \
    do {                                                                                           \
        rocrand_status _s = (call);                                                                \
        if (_s != ROCRAND_STATUS_SUCCESS)                                                          \
            throw std::runtime_error("rocRAND error " + std::to_string(_s));                       \
    } while (0)
#endif

namespace rppbench {

#if RPP_BENCH_HIP
namespace {
/**
 * @brief Obtain the process-lifetime rocRAND generator.
 *
 * Created lazily and reused across every case so per-case setup is just a
 * device-side generate (no host fill, no H2D copy). Not thread-safe, which
 * matches the single-threaded benchmark registration.
 * @return The shared rocRAND generator.
 */
rocrand_generator device_rng() {
    static rocrand_generator gen = [] {
        rocrand_generator g = nullptr;
        ROCRAND_CHECK(rocrand_create_generator(&g, ROCRAND_RNG_PSEUDO_DEFAULT));
        ROCRAND_CHECK(rocrand_set_seed(g, 0x52505F5FULL)); // fixed => reproducible
        return g;
    }();
    return gen;
}
} // namespace
#endif

namespace {

// ---- per-dtype fill behavior, table-driven ----
//
// Both fill paths (HOST in-place, HIP device-side via rocRAND) used to carry a
// parallel switch over the dtype; adding a dtype meant editing both arms. Here
// each dtype is one row: `hostFill` writes a representative in-place pattern
// (well-formed values that won't skew kernel timing), and `deviceFill` (HIP
// only) generates the same range directly in device memory. An unknown dtype has
// no row and falls back to a zero-fill in fill().

void host_fill_u8(void *data, size_t sizeBytes) {
    auto *p = static_cast<unsigned char *>(data);
    for (size_t i = 0; i < sizeBytes; ++i)
        p[i] = static_cast<unsigned char>(i & 0xFF);
}
void host_fill_i8(void *data, size_t sizeBytes) {
    auto *p = static_cast<signed char *>(data);
    for (size_t i = 0; i < sizeBytes; ++i)
        p[i] = static_cast<signed char>((i & 0xFF) - 128);
}
void host_fill_f32(void *data, size_t sizeBytes) {
    auto *p = static_cast<float *>(data);
    size_t n = sizeBytes / sizeof(float);
    for (size_t i = 0; i < n; ++i)
        p[i] = 0.5F;
}
void host_fill_f16(void *data, size_t sizeBytes) {
    // IEEE half 0.5 == 0x3800; store as raw 16-bit to avoid a half type dep.
    auto *p = static_cast<unsigned short *>(data);
    size_t n = sizeBytes / sizeof(unsigned short);
    for (size_t i = 0; i < n; ++i)
        p[i] = 0x3800;
}

#if RPP_BENCH_HIP
// Uniform (0,1] for floats matches RPP's expected F32/F16 range; random bytes
// cover the full U8/I8 range. The count arg spans the whole allocation (incl.
// halo/guard) so boundary reads always hit valid data.
void device_fill_char(rocrand_generator g, void *data, size_t sizeBytes) {
    ROCRAND_CHECK(rocrand_generate_char(g, static_cast<unsigned char *>(data), sizeBytes));
}
void device_fill_f32(rocrand_generator g, void *data, size_t sizeBytes) {
    ROCRAND_CHECK(
        rocrand_generate_uniform(g, static_cast<float *>(data), sizeBytes / sizeof(float)));
}
void device_fill_f16(rocrand_generator g, void *data, size_t sizeBytes) {
    ROCRAND_CHECK(
        rocrand_generate_uniform_half(g, static_cast<half *>(data), sizeBytes / sizeof(half)));
}
#endif

struct DtypeFill {
    RpptDataType dtype;
    void (*hostFill)(void *data, size_t sizeBytes);
#if RPP_BENCH_HIP
    void (*deviceFill)(rocrand_generator g, void *data, size_t sizeBytes);
#endif
};

constexpr DtypeFill kDtypeFills[] = {
#if RPP_BENCH_HIP
    {RpptDataType::U8, host_fill_u8, device_fill_char},
    {RpptDataType::I8, host_fill_i8, device_fill_char},
    {RpptDataType::F32, host_fill_f32, device_fill_f32},
    {RpptDataType::F16, host_fill_f16, device_fill_f16},
#else
    {RpptDataType::U8, host_fill_u8},
    {RpptDataType::I8, host_fill_i8},
    {RpptDataType::F32, host_fill_f32},
    {RpptDataType::F16, host_fill_f16},
#endif
};

/**
 * @brief Look up the fill row for a dtype, or nullptr if unsupported.
 */
const DtypeFill *find_dtype_fill(RpptDataType dt) {
    for (const auto &e : kDtypeFills)
        if (e.dtype == dt)
            return &e;
    return nullptr;
}

} // namespace

void *bench_device_alloc(size_t bytes, bool isHip) {
#if RPP_BENCH_HIP
    if (isHip) {
        void *p = nullptr;
        HIP_CHECK(hipMalloc(&p, bytes));
        return p;
    }
#else
    (void)isHip;
#endif
    return std::malloc(bytes);
}

void bench_device_free(void *p, bool isHip) {
    if (!p)
        return;
#if RPP_BENCH_HIP
    if (isHip) {
        (void)hipFree(p);
        return;
    }
#else
    (void)isHip;
#endif
    std::free(p);
}

void *bench_pinned_alloc(size_t bytes, bool isHip) {
#if RPP_BENCH_HIP
    if (isHip) {
        void *p = nullptr;
        HIP_CHECK(hipHostMalloc(&p, bytes));
        return p;
    }
#else
    (void)isHip;
#endif
    return std::malloc(bytes);
}

void bench_pinned_free(void *p, bool isHip) {
    if (!p)
        return;
#if RPP_BENCH_HIP
    if (isHip) {
        (void)hipHostFree(p);
        return;
    }
#else
    (void)isHip;
#endif
    std::free(p);
}

void TensorBuffer::init(RppBackend backend, RpptDataType dt, Layout layout, int n, int w, int h,
                        int offsetInBytes, int additionalStride) {
    isHip = (backend == RppBackend::RPP_HIP_BACKEND);
    dtype = dt;
    batch = n;
    width = w;
    height = h;

    LayoutInfo li = layout_info(layout);
    channels = li.channels;

    desc = RpptDesc{};
    desc.numDims = 4;
    desc.offsetInBytes = static_cast<Rpp32u>(offsetInBytes);
    desc.dataType = dt;
    desc.layout = li.rpptLayout;
    desc.n = n;
    desc.c = channels;
    desc.h = h;
    // Pad width up to a multiple of 8 for SIMD-friendly strides (plus any halo
    // columns), matching set_descriptor_dims_and_strides. ROI keeps the real w/h.
    desc.w = ((w / 8) * 8) + 8 + additionalStride;

    if (desc.layout == RpptLayout::NHWC) {
        desc.strides.nStride = desc.c * desc.w * desc.h;
        desc.strides.hStride = desc.c * desc.w;
        desc.strides.wStride = desc.c;
        desc.strides.cStride = 1;
    } else { // NCHW
        desc.strides.nStride = desc.c * desc.w * desc.h;
        desc.strides.cStride = desc.w * desc.h;
        desc.strides.hStride = desc.w;
        desc.strides.wStride = 1;
    }

    // Allocation = leading halo offset + data + trailing guard. The guard (a few
    // rows) absorbs vectorized over-reads/writes the kernels do near the last
    // row/column; RPP kernels apply desc.offsetInBytes to reach the real data.
    dataBytes = static_cast<size_t>(desc.n) * desc.strides.nStride * dtype_size(dt);
    const size_t guardBytes =
        (static_cast<size_t>(desc.strides.hStride) * dtype_size(dt) * 4) + 256;
    sizeBytes = static_cast<size_t>(offsetInBytes) + dataBytes + guardBytes;
    data = bench_device_alloc(sizeBytes, isHip);

    // Full-image ROI (XYWH) per image, using the real (unpadded) dimensions.
    roi = static_cast<RpptROI *>(bench_pinned_alloc(sizeof(RpptROI) * n, isHip));
    roiType = RpptRoiType::XYWH;
    resetRoi();
}

void TensorBuffer::initGeneric(RppBackend backend, const TensorSpec &spec) {
    isHip = (backend == RppBackend::RPP_HIP_BACKEND);
    kind = spec.kind;
    dtype = spec.dtype;
    dims = spec.dims;
    numDims = static_cast<int>(dims.size());
    batch = numDims > 0 ? static_cast<int>(dims[0]) : 0;

    // The generic descriptor must be device-accessible (the HIP kernels read its
    // stride/dim arrays on-device), so allocate it pinned rather than using a
    // pageable host member. bench_pinned_alloc falls back to malloc on HOST.
    gdescPtr = static_cast<RpptGenericDescPtr>(bench_pinned_alloc(sizeof(RpptGenericDesc), isHip));
    *gdescPtr = RpptGenericDesc{};
    gdescPtr->numDims = numDims;
    gdescPtr->offsetInBytes = 0;
    gdescPtr->dataType = dtype;
    gdescPtr->layout = spec.layout;
    for (int i = 0; i < numDims; ++i)
        gdescPtr->dims[i] = dims[i];
    // Packed, contiguous strides (mirrors the test suite's compute_strides):
    // innermost stride 1, each outer stride the product of the inner dims. No
    // width padding here (unlike the image path) - the generic kernels don't
    // assume a SIMD-aligned row stride.
    if (numDims > 0) {
        Rpp32u v = 1;
        for (int i = numDims - 1; i > 0; --i) {
            gdescPtr->strides[i] = v;
            v *= gdescPtr->dims[i];
        }
        gdescPtr->strides[0] = v; // per-sample element count
    }

    // Pull spatial extents out by layout so the VOXEL ROI (and any width/height
    // throughput math) is correct regardless of packed/planar ordering.
    if (spec.layout == RpptLayout::NDHWC) { // {n, d, h, w, c}
        depth = numDims > 1 ? static_cast<int>(dims[1]) : 0;
        height = numDims > 2 ? static_cast<int>(dims[2]) : 0;
        width = numDims > 3 ? static_cast<int>(dims[3]) : 0;
        channels = numDims > 4 ? static_cast<int>(dims[4]) : 1;
    } else if (spec.layout == RpptLayout::NCDHW) { // {n, c, d, h, w}
        channels = numDims > 1 ? static_cast<int>(dims[1]) : 1;
        depth = numDims > 2 ? static_cast<int>(dims[2]) : 0;
        height = numDims > 3 ? static_cast<int>(dims[3]) : 0;
        width = numDims > 4 ? static_cast<int>(dims[4]) : 0;
    } else if (spec.layout == RpptLayout::NHWC) { // {n, h, w, c} (2D generic)
        height = numDims > 1 ? static_cast<int>(dims[1]) : 0;
        width = numDims > 2 ? static_cast<int>(dims[2]) : 0;
        channels = numDims > 3 ? static_cast<int>(dims[3]) : 1;
    } else { // NCHW: {n, c, h, w}
        channels = numDims > 1 ? static_cast<int>(dims[1]) : 1;
        height = numDims > 2 ? static_cast<int>(dims[2]) : 0;
        width = numDims > 3 ? static_cast<int>(dims[3]) : 0;
    }

    // Allocation = data + trailing guard (a few inner "rows") to absorb the
    // vectorized over-reads/writes the kernels do near the tensor's end.
    dataBytes = static_cast<size_t>(batch) * gdescPtr->strides[0] * dtype_size(dtype);
    const size_t rowStride = numDims >= 2 ? gdescPtr->strides[numDims - 2] : gdescPtr->strides[0];
    const size_t guardBytes = (rowStride * dtype_size(dtype) * 4) + 256;
    sizeBytes = dataBytes + guardBytes;
    data = bench_device_alloc(sizeBytes, isHip);

    // ROI companion: the flat begin/length roiTensor for the GenericND ops, or a
    // per-sample RpptROI3D for the voxel ops. resetRoi() (re)writes the values.
    const int nSpatial = numDims - 1;
    if (kind == TensorKind::VOXEL) {
        roi3d = static_cast<RpptROI3D *>(bench_pinned_alloc(sizeof(RpptROI3D) * batch, isHip));
        roi3dType = RpptRoi3DType::XYZWHD;
    } else if (nSpatial > 0) {
        roiTensor =
            static_cast<Rpp32u *>(bench_pinned_alloc(sizeof(Rpp32u) * batch * nSpatial * 2, isHip));
    }
    resetRoi();
}

void TensorBuffer::resetRoi() const {
    if (kind == TensorKind::VOXEL) {
        if (!roi3d)
            return;
        for (int i = 0; i < batch; ++i) {
            roi3d[i].xyzwhdROI.xyz.x = 0;
            roi3d[i].xyzwhdROI.xyz.y = 0;
            roi3d[i].xyzwhdROI.xyz.z = 0;
            roi3d[i].xyzwhdROI.roiWidth = width;
            roi3d[i].xyzwhdROI.roiHeight = height;
            roi3d[i].xyzwhdROI.roiDepth = depth;
        }
        return;
    }
    if (kind == TensorKind::GENERIC_ND) {
        if (!roiTensor)
            return;
        // Per sample: [begin_0..begin_{s-1}, length_0..length_{s-1}], s spatial
        // dims (batch excluded), matching the test suite's fill_roi_values.
        const int s = numDims - 1;
        for (int b = 0; b < batch; ++b) {
            Rpp32u *r = roiTensor + static_cast<size_t>(b) * s * 2;
            for (int j = 0; j < s; ++j) {
                r[j] = 0;               // begin
                r[s + j] = dims[1 + j]; // length
            }
        }
        return;
    }
    // IMAGE: rewrite the canonical full-image XYWH ROI. Several RPP ops convert
    // the ROI in place (e.g. resize's xywh->ltrb: roi.z += roi.x - 1) on every call; in a
    // benchmark loop that repeatedly mutates the same tensor until indices go
    // negative and the kernel reads out of bounds. Resetting before each timed
    // call keeps the ROI stable and matches single-call (test-suite) semantics.
    for (int i = 0; i < batch; ++i) {
        roi[i].xywhROI.xy.x = 0;
        roi[i].xywhROI.xy.y = 0;
        roi[i].xywhROI.roiWidth = width;
        roi[i].xywhROI.roiHeight = height;
    }
}

void TensorBuffer::fill() const {
    if (!data || sizeBytes == 0)
        return;

    const DtypeFill *f = find_dtype_fill(dtype);

#if RPP_BENCH_HIP
    if (isHip) {
        // Generate the input directly in device memory - no host fill, no H2D.
        if (f)
            f->deviceFill(device_rng(), data, sizeBytes);
        else
            HIP_CHECK(hipMemset(data, 0, sizeBytes));
        // Generation runs on rocRAND's stream; ensure it completes before the
        // (untimed) setup returns and kernels on the case stream read the data.
        HIP_CHECK(hipDeviceSynchronize());
        return;
    }
#endif

    // HOST backend: fill in place (buffer is host memory). Seed with well-formed
    // values per dtype (avoid NaN/denormals that could skew kernel timing).
    if (f)
        f->hostFill(data, sizeBytes);
    else
        std::memset(data, 0, sizeBytes);
}

void TensorBuffer::free() {
    if (data) {
        bench_device_free(data, isHip);
        data = nullptr;
    }
    if (roi) {
        bench_pinned_free(roi, isHip);
        roi = nullptr;
    }
    if (roiTensor) {
        bench_pinned_free(roiTensor, isHip);
        roiTensor = nullptr;
    }
    if (roi3d) {
        bench_pinned_free(roi3d, isHip);
        roi3d = nullptr;
    }
    if (gdescPtr) {
        bench_pinned_free(gdescPtr, isHip);
        gdescPtr = nullptr;
    }
    sizeBytes = 0;
}

} // namespace rppbench
