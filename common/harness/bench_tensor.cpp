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

void TensorBuffer::resetRoi() const {
    // Rewrite the canonical full-image XYWH ROI. Several RPP ops convert the ROI
    // in place (e.g. resize's xywh->ltrb: roi.z += roi.x - 1) on every call; in a
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

#if RPP_BENCH_HIP
    if (isHip) {
        // Generate the input directly in device memory - no host fill, no H2D.
        // Uniform (0,1] for floats matches RPP's expected F32/F16 range; random
        // bytes cover the full U8/I8 range. The whole allocation (incl. halo and
        // guard) is filled so boundary reads always hit valid data.
        rocrand_generator g = device_rng();
        switch (dtype) {
        case RpptDataType::U8:
        case RpptDataType::I8:
            ROCRAND_CHECK(rocrand_generate_char(g, static_cast<unsigned char *>(data), sizeBytes));
            break;
        case RpptDataType::F32:
            ROCRAND_CHECK(
                rocrand_generate_uniform(g, static_cast<float *>(data), sizeBytes / sizeof(float)));
            break;
        case RpptDataType::F16:
            ROCRAND_CHECK(rocrand_generate_uniform_half(g, static_cast<half *>(data),
                                                        sizeBytes / sizeof(half)));
            break;
        default:
            HIP_CHECK(hipMemset(data, 0, sizeBytes));
        }
        // Generation runs on rocRAND's stream; ensure it completes before the
        // (untimed) setup returns and kernels on the case stream read the data.
        HIP_CHECK(hipDeviceSynchronize());
        return;
    }
#endif

    // HOST backend: fill in place (buffer is host memory). Seed with well-formed
    // values per dtype (avoid NaN/denormals that could skew kernel timing).
    switch (dtype) {
    case RpptDataType::U8: {
        auto *p = static_cast<unsigned char *>(data);
        for (size_t i = 0; i < sizeBytes; ++i)
            p[i] = static_cast<unsigned char>(i & 0xFF);
        break;
    }
    case RpptDataType::I8: {
        auto *p = static_cast<signed char *>(data);
        for (size_t i = 0; i < sizeBytes; ++i)
            p[i] = static_cast<signed char>((i & 0xFF) - 128);
        break;
    }
    case RpptDataType::F32: {
        auto *p = static_cast<float *>(data);
        size_t n = sizeBytes / sizeof(float);
        for (size_t i = 0; i < n; ++i)
            p[i] = 0.5F;
        break;
    }
    case RpptDataType::F16: {
        // IEEE half 0.5 == 0x3800; store as raw 16-bit to avoid a half type dep.
        auto *p = static_cast<unsigned short *>(data);
        size_t n = sizeBytes / sizeof(unsigned short);
        for (size_t i = 0; i < n; ++i)
            p[i] = 0x3800;
        break;
    }
    default:
        std::memset(data, 0, sizeBytes);
    }
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
    sizeBytes = 0;
}

} // namespace rppbench
