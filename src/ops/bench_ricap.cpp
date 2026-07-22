/**
 * @file bench_ricap.cpp
 * @brief Adapter for rppt_ricap - Random Image Crop And Patch.
 *
 * RICAP tiles each output image from 4 rectangular regions, where each region
 * is cropped from a (permuted) image in the batch. It requires all images in a
 * batch to share the same dimensions - the harness sweep uses uniform per-case
 * dims, so this holds.
 *
 * NOTE: rppt_ricap requires batchSize > 1 (it patches each output from *other*
 * images in the batch); a batch of 1 makes it fail with status -1. Config
 * entries therefore override batch_sizes to values >= 2.
 *
 * Two shared param tensors drive it:
 *   - permutationTensor (Rpp32u*, batchSize * 4): for each output image n, four
 *     source-image indices in [0, batchSize) selecting which batch image each of
 *     the 4 regions is cropped from. Laid out as permutationTensor[n*4 + k].
 *     Here we use (n + k) % batchSize (all 0 when batch == 1, the only valid
 *     choice for a single-image batch).
 *   - roiPtrInputCropRegion (RpptROI[4], shared across the batch): the 4 crop
 *     *windows* (size + anchor) taken from the four source images; their widths
 *     sum to the output width and their heights to its height. The test suite
 *     randomizes both the split point and each window's anchor within valid
 *     ranges whose lower bound is always 0, so we use a fixed centre split
 *     (cx = width/2, cy = height/2) anchored at (0,0) - deterministic and always
 *     in-bounds. Anchoring the wider/taller windows at (0,0) matters: the kernel
 *     requires each window to fit with margin (the test even subtracts 8 from the
 *     x upper bound for HIP's 8-pixels-at-once processing), so a non-zero anchor
 *     that pushes x+width to the image edge makes rppt_ricap fail with status -1.
 *       region0: {0,0}, w=cx,        h=cy
 *       region1: {0,0}, w=width-cx,  h=cy
 *       region2: {0,0}, w=cx,        h=height-cy
 *       region3: {0,0}, w=width-cx,  h=height-cy
 *     ROIs are XYWH (RpptRoiType::XYWH), matching the test suite.
 *
 * Signature:
 *   rppt_ricap(src, srcDesc, dst, dstDesc, permutationTensor,
 *              roiPtrInputCropRegion, roiType, handle, backend)
 */
#include "harness/bench_simple_adapter.hpp"
#include <rpp/rppt_tensor_effects_augmentations.h>

namespace rppbench {

class RicapAdapter : public SimpleOpAdapter {
public:
    void setup(const BenchContext &ctx, TensorBuffer &, TensorBuffer &) override {
        isHip_ = ctx.isHip;
        perm_ =
            static_cast<Rpp32u *>(bench_pinned_alloc(sizeof(Rpp32u) * ctx.batch * 4, ctx.isHip));
        cropRegions_ = static_cast<RpptROI *>(bench_pinned_alloc(sizeof(RpptROI) * 4, ctx.isHip));

        const int cx = ctx.width / 2;
        const int cy = ctx.height / 2;
        cropRegions_[0].xywhROI = {{0, 0}, cx, cy};
        cropRegions_[1].xywhROI = {{0, 0}, ctx.width - cx, cy};
        cropRegions_[2].xywhROI = {{0, 0}, cx, ctx.height - cy};
        cropRegions_[3].xywhROI = {{0, 0}, ctx.width - cx, ctx.height - cy};

        for (int n = 0; n < ctx.batch; ++n)
            for (int k = 0; k < 4; ++k)
                perm_[n * 4 + k] = static_cast<Rpp32u>((n + k) % ctx.batch);
    }

    RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                  rppHandle_t handle) override {
        return rppt_ricap(src.data, src.descPtr, dst.data, dst.descPtr, perm_, cropRegions_,
                          RpptRoiType::XYWH, handle, ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(perm_, isHip_);
        perm_ = nullptr;
        bench_pinned_free(cropRegions_, isHip_);
        cropRegions_ = nullptr;
    }

private:
    Rpp32u *perm_ = nullptr;
    RpptROI *cropRegions_ = nullptr;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("ricap", RicapAdapter)

} // namespace rppbench
