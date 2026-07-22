/**
 * @file bench_random_erase.cpp
 * @brief Adapter for rppt_random_erase - erase rectangular regions and refill
 *        them with tiled noise.
 *
 * The number of erase boxes per image is NOT part of the C signature - it is a
 * fixed count baked into the kernel. The RPP test suite drives this op with
 * `boxesInEachImage = 1` (HOST/Tensor_image_host.cpp RANDOM_ERASE case), so the
 * anchor-box tensor is a flat RpptRoiLtrb array of exactly `batch * 1` entries.
 * We place one non-overlapping, in-bounds rectangle per image (a centred
 * quarter-area box).
 *
 * The noiseBuffer holds the source pixels used to refill the erased region,
 * accessed tiled. Its size is fixed by RPP as
 * `RANDOM_ERASE_NOISE_BUFFER_SIDE * RANDOM_ERASE_NOISE_BUFFER_SIDE * channels`
 * (rppdefs.h defines the side as 255 - changing it breaks QA tiling), i.e.
 * `255 * 255 * channels * dtype_size` bytes. It is filled with mid-range values
 * (memset), which lie within every supported dtype's range.
 *
 * All harness dtypes (U8/F16/F32/I8) and all layouts (PKD3/PLN3/PLN1) are
 * supported, on both backends - no overrides needed.
 *
 * Signature:
 *   rppt_random_erase(src, srcDesc, dst, dstDesc, anchorBoxInfoTensor,
 *                     noiseBuffer, roi, roiType, handle, backend)
 */
#include "harness/bench_simple_adapter.hpp"
#include <rpp/rppt_tensor_effects_augmentations.h>

#include <cstring>

namespace rppbench {

class RandomEraseAdapter : public SimpleOpAdapter {
public:
    void setup(const BenchContext &ctx, TensorBuffer &, TensorBuffer &) override {
        isHip_ = ctx.isHip;
        channels_ = layout_info(ctx.layout).channels;

        // Fixed box count baked into the kernel: 1 box per image (matches the
        // RPP test suite's boxesInEachImage = 1).
        boxes_ = static_cast<RpptRoiLtrb *>(
            bench_pinned_alloc(sizeof(RpptRoiLtrb) * ctx.batch, ctx.isHip));

        // Noise buffer: 255 * 255 * channels elements in the source dtype.
        const size_t noiseBytes = static_cast<size_t>(RANDOM_ERASE_NOISE_BUFFER_SIDE) *
                                  RANDOM_ERASE_NOISE_BUFFER_SIDE * channels_ *
                                  dtype_size(ctx.dtype);
        noise_ = bench_pinned_alloc(noiseBytes, ctx.isHip);
        // Mid-range fill: valid for U8 (~127), F16/F32 (small positive), I8 (a
        // benign value); values are only tiled into the erased region.
        std::memset(noise_, 0x40, noiseBytes);

        for (int i = 0; i < ctx.batch; ++i) {
            boxes_[i].lt = {ctx.width / 4, ctx.height / 4};
            boxes_[i].rb = {ctx.width * 3 / 4, ctx.height * 3 / 4};
        }
    }

    RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                  rppHandle_t handle) override {
        return rppt_random_erase(src.data, src.descPtr, dst.data, dst.descPtr, boxes_, noise_,
                                 src.roi, src.roiType, handle, ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(boxes_, isHip_);
        boxes_ = nullptr;
        bench_pinned_free(noise_, isHip_);
        noise_ = nullptr;
    }

private:
    RpptRoiLtrb *boxes_ = nullptr;
    void *noise_ = nullptr;
    int channels_ = 3;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("random_erase", RandomEraseAdapter)

} // namespace rppbench
