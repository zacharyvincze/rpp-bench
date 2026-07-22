/**
 * @file bench_remap.cpp
 * @brief Adapter for rppt_remap - per-pixel geometric remap via row/col tables.
 *
 * For every destination pixel, output(x,y) = input(colRemapTable(x,y),
 * rowRemapTable(x,y)). Two Rpp32f per-pixel tables (row = y/source-row numbers,
 * col = x/source-column numbers) drive the sampling, plus an interpolation type.
 *
 * Table sizing/typing (mirrors the RPP test suite's init_remap + ioBufferSize):
 * the tables are a single-channel (c=1), F32, NHWC tensor described by a
 * dedicated tableDescPtr whose strides use the source descriptor's *padded*
 * width (nStride = h*paddedW, hStride = paddedW, wStride = cStride = 1) - the
 * RPP kernel indexes the tables with these strides, so both the strides and the
 * allocation must follow the SIMD-padded image width (src.descPtr->w), not the
 * real ctx width, or the kernel reads/writes out of bounds (HIP illegal access).
 * The per-image ROI region (real ctx dims) is filled with an identity map
 * (row=i, col=j) so output == input - a stable, representative remap that keeps
 * kernel timing meaningful.
 *
 * On HIP the tables are pinned host memory (device-accessible on ROCm), matching
 * the harness param-tensor pattern; the tableDesc is read host-side by RPP so it
 * stays a plain member. Restrictions (from the header): dataType = U8/F16/F32/I8,
 * layout = NCHW/NHWC, c = 1/3, and interpolationType supports only
 * NEAREST_NEIGHBOR and BILINEAR - so all sweep dtypes/layouts/backends apply and
 * no supported* overrides are needed.
 *
 * Signature:
 *   rppt_remap(src, srcDesc, dst, dstDesc, rowRemapTable, colRemapTable,
 *              tableDesc, interpolationType, roi, roiType, handle, backend)
 */
#include "harness/bench_simple_adapter.hpp"
#include <rpp/rppt_tensor_geometric_augmentations.h>

namespace rppbench {

class RemapAdapter : public SimpleOpAdapter {
public:
    void setup(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &) override {
        isHip_ = ctx.isHip;
        interp_ = parse_interpolation(ctx.param<std::string>("interpolation", "BILINEAR"));

        const auto w = static_cast<Rpp32u>(ctx.width);  // real ROI width
        const auto h = static_cast<Rpp32u>(ctx.height); // real ROI height
        const auto n = static_cast<Rpp32u>(ctx.batch);
        const auto paddedW = src.descPtr->w; // SIMD-padded stride width

        // Single-channel F32 NHWC table descriptor. Strides use the source's
        // padded width, exactly as the RPP test suite's init_remap does - the
        // kernel indexes the tables with these strides.
        tableDesc_ = RpptDesc{};
        tableDesc_.numDims = 4;
        tableDesc_.offsetInBytes = 0;
        tableDesc_.dataType = RpptDataType::F32;
        tableDesc_.layout = RpptLayout::NHWC;
        tableDesc_.n = n;
        tableDesc_.c = 1;
        tableDesc_.h = h;
        tableDesc_.w = paddedW;
        tableDesc_.strides.nStride = h * paddedW;
        tableDesc_.strides.hStride = paddedW;
        tableDesc_.strides.wStride = 1;
        tableDesc_.strides.cStride = 1;

        // Allocate to the padded stride extent so kernel indexing stays in bounds.
        const size_t count = static_cast<size_t>(n) * tableDesc_.strides.nStride;
        const size_t bytes = count * sizeof(Rpp32f);
        rowRemapTable_ = static_cast<Rpp32f *>(bench_pinned_alloc(bytes, ctx.isHip));
        colRemapTable_ = static_cast<Rpp32f *>(bench_pinned_alloc(bytes, ctx.isHip));

        // Identity map over the real ROI: sample the co-located source pixel
        // (output == input). Rows are addressed via the padded hStride.
        for (Rpp32u b = 0; b < n; ++b) {
            Rpp32f *row = rowRemapTable_ + (static_cast<size_t>(b) * tableDesc_.strides.nStride);
            Rpp32f *col = colRemapTable_ + (static_cast<size_t>(b) * tableDesc_.strides.nStride);
            for (Rpp32u y = 0; y < h; ++y) {
                Rpp32f *rowLine = row + (static_cast<size_t>(y) * tableDesc_.strides.hStride);
                Rpp32f *colLine = col + (static_cast<size_t>(y) * tableDesc_.strides.hStride);
                for (Rpp32u x = 0; x < w; ++x) {
                    rowLine[x] = static_cast<Rpp32f>(y);
                    colLine[x] = static_cast<Rpp32f>(x);
                }
            }
        }
    }

    RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                  rppHandle_t handle) override {
        return rppt_remap(src.data, src.descPtr, dst.data, dst.descPtr, rowRemapTable_,
                          colRemapTable_, &tableDesc_, interp_, src.roi, src.roiType, handle,
                          ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(rowRemapTable_, isHip_);
        rowRemapTable_ = nullptr;
        bench_pinned_free(colRemapTable_, isHip_);
        colRemapTable_ = nullptr;
    }

private:
    Rpp32f *rowRemapTable_ = nullptr;
    Rpp32f *colRemapTable_ = nullptr;
    RpptDesc tableDesc_{};
    RpptInterpolationType interp_ = RpptInterpolationType::BILINEAR;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("remap", RemapAdapter)

} // namespace rppbench
