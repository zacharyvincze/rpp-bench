/**
 * @file bench_phase.cpp
 * @brief Adapter for rppt_phase - per-pixel phase angle atan2(src2, src1).
 *
 * A two-source op: both sources share one srcDesc/ROI, so it overrides numSrc()
 * to 2 and inherits OpAdapter directly. No params; supports all dtypes/layouts
 * (c = 1/3).
 *
 * Signature:
 *   rppt_phase(src1, src2, srcDesc, dst, dstDesc, roi, roiType, handle, backend)
 */
#include "harness/bench_registry.hpp"
#include <rpp/rppt_tensor_geometric_augmentations.h>

namespace rppbench {

class PhaseAdapter : public OpAdapter {
public:
    int numSrc() const override { return 2; }

    void setup(const BenchContext &, std::vector<TensorBuffer> &,
               std::vector<TensorBuffer> &) override {}

    RppStatus run(const BenchContext &ctx, std::vector<TensorBuffer> &src,
                  std::vector<TensorBuffer> &dst, rppHandle_t handle) override {
        return rppt_phase(src[0].data, src[1].data, src[0].descPtr, dst[0].data, dst[0].descPtr,
                          src[0].roi, src[0].roiType, handle, ctx.backend);
    }
};

REGISTER_RPP_BENCH("phase", PhaseAdapter)

} // namespace rppbench
