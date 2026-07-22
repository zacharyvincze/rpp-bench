/**
 * @file bench_lens_correction.cpp
 * @brief Adapter for rppt_lens_correction - barrel-distortion (lens) correction.
 *
 * Signature:
 *   rppt_lens_correction(src, srcDesc, dst, dstDesc,
 *                        rowRemapTable, colRemapTable, tableDesc,
 *                        cameraMatrixTensor, distortionCoeffsTensor,
 *                        roi, roiType, handle, backend)
 *
 * Per the header Restrictions the op accepts every dtype the harness sweeps
 * (U8/F16/F32/I8), both layouts (NHWC=PKD3, NCHW=PLN3/PLN1) with c = 1/3, and
 * both backends - so the default supported* sets already match and are left
 * unrestricted.
 *
 * The op needs several auxiliary tensors alongside the image:
 *  - rowRemapTable / colRemapTable : Rpp32f scratch, size width*height*batch
 *    each. The kernel builds these internally from the camera parameters, so
 *    their initial contents are irrelevant (allocated, not filled).
 *  - cameraMatrixTensor : Rpp32f, 9 floats/sample (3x3 intrinsics).
 *  - distortionCoeffsTensor : Rpp32f, 8 floats/sample.
 *  - tableDesc : an auxiliary RpptDesc describing the remap tables (numDims=4,
 *    F32, c=1, packed strides). Built inline from the source descriptor,
 *    mirroring init_lens_correction() in RPP's test suite.
 *
 * Camera matrix / distortion coeffs are seeded with the representative sample
 * values from RPP's own test suite (a real ~640x480 calibration). A zero-
 * determinant camera matrix yields a black image (per the header note).
 */
#include "harness/bench_simple_adapter.hpp"
#include <rpp/rppt_tensor_geometric_augmentations.h>

namespace rppbench {

class LensCorrectionAdapter : public SimpleOpAdapter {
public:
    void setup(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &) override {
        isHip_ = ctx.isHip;

        // Auxiliary table descriptor: copy the source descriptor, then force a
        // single-channel F32 NHWC view with packed strides over the source's
        // *padded* width (RpptDesc contract for the remap tables). Mirrors
        // init_lens_correction() in the RPP test suite - the kernel writes the
        // internally-built maps using these strides, so the allocation below must
        // follow the padded extent or the kernel writes out of bounds.
        tableDesc_ = *src.descPtr;
        tableDesc_.dataType = RpptDataType::F32;
        tableDesc_.layout = RpptLayout::NHWC;
        tableDesc_.offsetInBytes = 0;
        tableDesc_.c = 1;
        tableDesc_.strides.nStride = src.descPtr->h * src.descPtr->w;
        tableDesc_.strides.hStride = src.descPtr->w;
        tableDesc_.strides.wStride = 1;
        tableDesc_.strides.cStride = 1;

        const auto tableElems = static_cast<size_t>(tableDesc_.strides.nStride) * ctx.batch;

        rowRemapTable_ =
            static_cast<float *>(bench_pinned_alloc(sizeof(float) * tableElems, isHip_));
        colRemapTable_ =
            static_cast<float *>(bench_pinned_alloc(sizeof(float) * tableElems, isHip_));
        cameraMatrix_ =
            static_cast<float *>(bench_pinned_alloc(sizeof(float) * 9 * ctx.batch, isHip_));
        distortionCoeffs_ =
            static_cast<float *>(bench_pinned_alloc(sizeof(float) * 8 * ctx.batch, isHip_));

        // Representative intrinsics/distortion from RPP's init_lens_correction().
        static const std::array<float, 9> kCameraMatrix = {
            534.07088364F, 0.0F, 341.53407554F, 0.0F, 534.11914595F,
            232.94565259F, 0.0F, 0.0F,          1.0F};
        static const std::array<float, 8> kDistortionCoeffs = {
            -0.29297164F, 0.10770696F, 0.00131038F, -0.0000311F, 0.0434798F, 0.0F, 0.0F, 0.0F};
        for (int i = 0; i < ctx.batch; ++i) {
            for (int k = 0; k < 9; ++k)
                cameraMatrix_[(i * 9) + k] = kCameraMatrix[k];
            for (int k = 0; k < 8; ++k)
                distortionCoeffs_[(i * 8) + k] = kDistortionCoeffs[k];
        }
    }

    RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                  rppHandle_t handle) override {
        return rppt_lens_correction(src.data, src.descPtr, dst.data, dst.descPtr, rowRemapTable_,
                                    colRemapTable_, &tableDesc_, cameraMatrix_, distortionCoeffs_,
                                    src.roi, src.roiType, handle, ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(rowRemapTable_, isHip_);
        rowRemapTable_ = nullptr;
        bench_pinned_free(colRemapTable_, isHip_);
        colRemapTable_ = nullptr;
        bench_pinned_free(cameraMatrix_, isHip_);
        cameraMatrix_ = nullptr;
        bench_pinned_free(distortionCoeffs_, isHip_);
        distortionCoeffs_ = nullptr;
    }

private:
    float *rowRemapTable_ = nullptr;
    float *colRemapTable_ = nullptr;
    float *cameraMatrix_ = nullptr;
    float *distortionCoeffs_ = nullptr;
    RpptDesc tableDesc_{};
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("lens_correction", LensCorrectionAdapter)

} // namespace rppbench
