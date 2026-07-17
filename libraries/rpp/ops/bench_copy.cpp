/**
 * @file bench_copy.cpp
 * @brief Adapter for rppt_copy - a plain buffer copy, no params and no ROI.
 *
 * Useful as a memory-bandwidth baseline: it isolates the src->dst copy cost
 * that every other op also pays, so richer ops can be read relative to it.
 *
 * Signature:
 *   rppt_copy(src, srcDesc, dst, dstDesc, handle, backend)
 */
#include "rpp/bench_simple_adapter.hpp"
#include <rpp/rppt_tensor_data_exchange_operations.h>

namespace rppbench {

class CopyAdapter : public SimpleOpAdapter {
public:
    void setup(const BenchContext &, TensorBuffer &, TensorBuffer &) override {}

    RppStatus run(const BenchContext &ctx, TensorBuffer &src, TensorBuffer &dst,
                  rppHandle_t handle) override {
        return rppt_copy(src.data, src.descPtr, dst.data, dst.descPtr, handle, ctx.backend);
    }
};

REGISTER_RPP_BENCH("copy", CopyAdapter)

} // namespace rppbench
