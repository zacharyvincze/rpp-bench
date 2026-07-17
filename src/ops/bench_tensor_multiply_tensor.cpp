/**
 * @file bench_tensor_multiply_tensor.cpp
 * @brief Adapter for rppt_tensor_multiply_tensor - element-wise multiply of two generic ND tensors.
 *
 * Two-source generic op via BroadcastOpAdapter (see bench_tensor_add_tensor.cpp):
 * both sources share one generic-ND shape so broadcasting is disabled. F32 only.
 */
#include "harness/bench_generic_adapters.hpp"
#include <rpp/rppt_tensor_arithmetic_operations.h>

#include <vector>

namespace rppbench {

class TensorMultiplyTensorAdapter : public BroadcastOpAdapter {
public:
    std::vector<RpptDataType> supportedDtypes() const override { return {RpptDataType::F32}; }

    void setup(const BenchContext &, std::vector<TensorBuffer> &,
               std::vector<TensorBuffer> &) override {}

    RppStatus run(const BenchContext &ctx, std::vector<TensorBuffer> &src,
                  std::vector<TensorBuffer> &dst, rppHandle_t handle) override {
        return rppt_tensor_multiply_tensor(src[0].data, src[1].data, src[0].gdescPtr,
                                           src[1].gdescPtr, dst[0].data, dst[0].gdescPtr,
                                           broadcastMode(ctx), src[0].roiTensor, src[1].roiTensor,
                                           handle, ctx.backend);
    }
};

REGISTER_RPP_BENCH("tensor_multiply_tensor", TensorMultiplyTensorAdapter)

} // namespace rppbench
