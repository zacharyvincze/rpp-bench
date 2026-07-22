/**
 * @file bench_lut.cpp
 * @brief Adapter for rppt_lut - per-pixel value remap via a lookup table.
 *
 * A single lookup table (length 65536 per the API) is shared across the batch,
 * indexed by the source pixel value. Restricted to the integer dtypes (U8/I8)
 * where a LUT is meaningful; the table element type matches the dtype. The LUT
 * lives in pinned/host memory even for the HIP backend, per the API.
 *
 * Signature:
 *   rppt_lut(src, srcDesc, dst, dstDesc, lutPtr, roi, roiType, handle, backend)
 */
#include "harness/bench_simple_adapter.hpp"
#include <rpp/rppt_tensor_color_augmentations.h>

#include <cstdint>

namespace rppbench {

class LutAdapter : public SimpleOpAdapter {
public:
    std::vector<RpptDataType> supportedDtypes() const override {
        return {RpptDataType::U8, RpptDataType::I8};
    }

    void setup(const BenchContext &ctx, TensorBuffer &, TensorBuffer &) override {
        isHip_ = ctx.isHip;
        // 65536 entries of the dtype's width (1 byte for U8/I8); identity ramp.
        const size_t entries = 65536;
        lut_ =
            static_cast<uint8_t *>(bench_pinned_alloc(entries * dtype_size(ctx.dtype), ctx.isHip));
        for (size_t i = 0; i < entries; ++i)
            lut_[i] = static_cast<uint8_t>(i & 0xFF);
    }

    RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                  rppHandle_t handle) override {
        return rppt_lut(src.data, src.descPtr, dst.data, dst.descPtr, lut_, src.roi, src.roiType,
                        handle, ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(lut_, isHip_);
        lut_ = nullptr;
    }

private:
    uint8_t *lut_ = nullptr;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("lut", LutAdapter)

} // namespace rppbench
