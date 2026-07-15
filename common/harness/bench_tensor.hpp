/**
 * @file bench_tensor.hpp
 * @brief Tensor allocation + descriptor/ROI setup for benchmarks.
 *
 * Centralizes what the RPP test suite spreads across hundreds of lines:
 * strided descriptor setup (mirrors set_descriptor_dims_and_strides), host/HIP
 * allocation, ROI tensors, and dtype byte-size math. Adapters get a ready-to-use
 * src/dst buffer and only build their own op-specific param tensors.
 */
#ifndef RPP_BENCH_TENSOR_HPP
#define RPP_BENCH_TENSOR_HPP

#include <rpp/rppdefs.h>
#include "config/bench_enums.hpp"
#include <cstddef>

namespace rppbench {

// ---- raw memory helpers (HIP-aware; fall back to host malloc when HOST-only) ----
/**
 * @brief Allocate "device" memory: hipMalloc for HIP, malloc for HOST.
 *
 * This is the memory rppt reads/writes.
 * @param bytes Number of bytes to allocate.
 * @param isHip Whether to use the HIP allocator.
 * @return Pointer to the allocation.
 */
void *bench_device_alloc(size_t bytes, bool isHip);
/**
 * @brief Free memory obtained from bench_device_alloc().
 * @param p Pointer to free (may be null).
 * @param isHip Must match the flag passed to bench_device_alloc().
 */
void bench_device_free(void *p, bool isHip);
/**
 * @brief Allocate "pinned" memory: hipHostMalloc for HIP, malloc for HOST.
 *
 * Used for param/ROI tensors the API expects in host-accessible-yet-device-
 * visible memory.
 * @param bytes Number of bytes to allocate.
 * @param isHip Whether to use the HIP allocator.
 * @return Pointer to the allocation.
 */
void *bench_pinned_alloc(size_t bytes, bool isHip);
/**
 * @brief Free memory obtained from bench_pinned_alloc().
 * @param p Pointer to free (may be null).
 * @param isHip Must match the flag passed to bench_pinned_alloc().
 */
void bench_pinned_free(void *p, bool isHip);

// A single input or output tensor: descriptor + backing buffer + full-image ROI.
class TensorBuffer {
public:
    RpptDesc desc{};
    RpptDescPtr descPtr = &desc;
    void *data = nullptr;   // device (HIP) or host (HOST) buffer
    RpptROI *roi = nullptr; // per-image ROI (pinned), XYWH full-image
    RpptRoiType roiType = RpptRoiType::XYWH;
    size_t sizeBytes = 0; // total allocation (offset + data + guard)
    size_t dataBytes = 0; // logical tensor bytes (for throughput)
    int batch = 0, width = 0, height = 0, channels = 0;
    RpptDataType dtype = RpptDataType::U8;

    /**
     * @brief Configure descriptor/strides/ROI and allocate.
     *
     * A trailing guard pad is always added to absorb vectorized boundary
     * reads/writes near the last row/column.
     * @param backend HOST or HIP; selects the allocator.
     * @param dt Element data type.
     * @param layout Packed/planar layout.
     * @param n Batch size.
     * @param w Image width.
     * @param h Image height.
     * @param offsetInBytes Leading halo padding (HIP filter kernels require
     *                      >= 12*(kernelSize/2)); stored in desc.offsetInBytes.
     * @param additionalStride Extra columns added to the padded width (halo).
     */
    void init(RppBackend backend, RpptDataType dt, Layout layout, int n, int w, int h,
              int offsetInBytes = 0, int additionalStride = 0);
    /**
     * @brief Fill the buffer with representative, well-formed values for the dtype.
     *
     * HIP buffers are generated directly in device memory via rocRAND (no H2D);
     * HOST buffers are filled in place.
     */
    void fill() const;
    /**
     * @brief Rewrite the canonical full-image XYWH ROI.
     *
     * Call before each timed op call: some RPP ops convert the ROI in place,
     * corrupting it across repeated calls.
     */
    void resetRoi() const;
    void free();
    ~TensorBuffer() { free(); }

    bool isHip = false;
};

} // namespace rppbench

#endif // RPP_BENCH_TENSOR_HPP
