/**
 * @file bench_log.cpp
 * @brief Adapter for rppt_log - natural log of a generic ND tensor, element-wise.
 *
 * Generic-descriptor op (RpptGenericDesc + flat roiTensor) via GenericOpAdapter;
 * see bench_transpose.cpp for the pattern. Shape-preserving with no op-specific
 * param tensors - the base's default 1-in/1-out spec is exactly right. F32 only.
 *
 * Signature:
 *   rppt_log(src, srcGenericDesc, dst, dstGenericDesc, roiTensor, handle, backend)
 */
#include "rpp/bench_generic_adapters.hpp"
#include <rpp/rppt_tensor_arithmetic_operations.h>

#include <vector>

namespace rppbench {

class LogAdapter : public GenericOpAdapter {
public:
    std::vector<RpptDataType> supportedDtypes() const override { return {RpptDataType::F32}; }

    void setup(const BenchContext &, std::vector<TensorBuffer> &,
               std::vector<TensorBuffer> &) override {}

    RppStatus run(const BenchContext &ctx, std::vector<TensorBuffer> &src,
                  std::vector<TensorBuffer> &dst, rppHandle_t handle) override {
        return rppt_log(src[0].data, src[0].gdescPtr, dst[0].data, dst[0].gdescPtr,
                        src[0].roiTensor, handle, ctx.backend);
    }
};

REGISTER_RPP_BENCH("log", LogAdapter)

} // namespace rppbench
