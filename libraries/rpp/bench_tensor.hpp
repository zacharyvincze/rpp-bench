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
#include "rpp/rpp_enums.hpp"
#include <cstddef>
#include <vector>

namespace rppbench {

// Which descriptor dialect a tensor speaks. The vast majority of ops are 4D
// images (RpptDesc + RpptROI); the generic-descriptor ops (transpose, slice,
// normalize, the voxel/broadcast families) use RpptGenericDesc plus either a
// flat Rpp32u roiTensor (GenericND) or an RpptROI3D (Voxel).
enum class TensorKind : uint8_t { IMAGE, GENERIC_ND, VOXEL };

// A tensor's shape as an adapter declares it to the runner. The runner turns
// each spec into an allocated TensorBuffer (via init() for Image, initGeneric()
// for the generic kinds), so ops no longer depend on the runner assuming 4D
// images. `dims` is in physical/layout order and INCLUDES the batch as dims[0]
// (e.g. NHWC image {n,h,w,c}; NCDHW voxel {n,c,d,h,w}).
struct TensorSpec {
    TensorKind kind = TensorKind::IMAGE;
    std::vector<Rpp32u> dims;
    RpptLayout layout = RpptLayout::NHWC;
    RpptDataType dtype = RpptDataType::U8;
};

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
//
// Serves two descriptor dialects off one shared byte core (allocation + fill +
// guard, all dialect-agnostic). init() builds the 4D image view (RpptDesc +
// RpptROI); initGeneric() builds the RpptGenericDesc view plus its matching ROI
// companion (a flat roiTensor for GenericND, an RpptROI3D for Voxel). `kind`
// records which view is live; image adapters touch only desc/descPtr/roi/roiType
// and generic adapters only gdesc/gdescPtr/roiTensor/roi3d.
class TensorBuffer {
public:
    TensorKind kind = TensorKind::IMAGE;

    // ---- image view (kind == IMAGE) ----
    RpptDesc desc{};
    RpptDescPtr descPtr = &desc;
    RpptROI *roi = nullptr; // per-image ROI (pinned), XYWH full-image
    RpptRoiType roiType = RpptRoiType::XYWH;

    // ---- generic view (kind == GENERIC_ND / VOXEL) ----
    // The generic descriptor is allocated in PINNED memory (see initGeneric): the
    // HIP kernels for these ops dereference gdescPtr->strides / ->dims ON THE
    // DEVICE, so a pageable host struct would stall the GPU on page faults. (Image
    // kernels read RpptDesc host-side, so `desc` above can stay an inline member.)
    RpptGenericDescPtr gdescPtr = nullptr;
    Rpp32u *roiTensor = nullptr; // flat begin/length per dim: batch*numDims*2
    RpptROI3D *roi3d = nullptr;  // per-sample 3D ROI (VOXEL only)
    RpptRoi3DType roi3dType = RpptRoi3DType::XYZWHD;
    int numDims = 0;          // logical dim count of the generic descriptor
    std::vector<Rpp32u> dims; // physical dims (dims[0] == batch)

    // ---- shared byte core ----
    void *data = nullptr; // device (HIP) or host (HOST) buffer
    size_t sizeBytes = 0; // total allocation (offset + data + guard)
    size_t dataBytes = 0; // logical tensor bytes (for throughput)
    int batch = 0, width = 0, height = 0, depth = 0, channels = 0;
    RpptDataType dtype = RpptDataType::U8;

    /**
     * @brief Configure the 4D image descriptor/strides/ROI and allocate.
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
     * @brief Configure the generic (RpptGenericDesc) descriptor and allocate.
     *
     * Computes packed (contiguous) strides over spec.dims, builds a flat
     * begin/length roiTensor covering the full tensor, and for VOXEL specs also
     * builds a per-sample RpptROI3D. spec.dims[0] is the batch; the remaining
     * dims are the per-sample shape in layout order. Shares the same guard-padded
     * allocation and fill() as the image path.
     * @param backend HOST or HIP; selects the allocator.
     * @param spec The tensor shape/layout/dtype the adapter declared.
     */
    void initGeneric(RppBackend backend, const TensorSpec &spec);
    /**
     * @brief Fill the buffer with representative, well-formed values for the dtype.
     *
     * HIP buffers are generated directly in device memory via rocRAND (no H2D);
     * HOST buffers are filled in place.
     */
    void fill() const;
    /**
     * @brief Rewrite the canonical full-tensor ROI for the live descriptor view.
     *
     * Call before each timed op call: some RPP ops convert the ROI in place,
     * corrupting it across repeated calls. Rewrites the XYWH image ROI, or the
     * flat roiTensor / RpptROI3D for the generic views.
     */
    void resetRoi() const;
    void free();
    ~TensorBuffer() { free(); }

    bool isHip = false;
};

} // namespace rppbench

#endif // RPP_BENCH_TENSOR_HPP
