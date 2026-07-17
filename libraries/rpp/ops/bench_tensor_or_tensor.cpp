/**
 * @file bench_tensor_or_tensor.cpp
 * @brief Adapter for rppt_tensor_or_tensor - bitwise OR of two generic ND tensors.
 *
 * Two-source generic op via BroadcastOpAdapter (see bench_tensor_add_tensor.cpp):
 * both sources share one generic-ND shape so broadcasting is disabled. U8 only
 * (bitwise kernels are defined for 8-bit data).
 */
#include "rpp/bench_generic_adapters.hpp"
#include <rpp/rppt_tensor_bitwise_operations.h>

#include <vector>

namespace rppbench {

class TensorOrTensorAdapter : public BroadcastOpAdapter {
public:
    std::vector<RpptDataType> supportedDtypes() const override { return {RpptDataType::U8}; }

    void setup(const BenchContext &, std::vector<TensorBuffer> &,
               std::vector<TensorBuffer> &) override {}

    RppStatus run(const BenchContext &ctx, std::vector<TensorBuffer> &src,
                  std::vector<TensorBuffer> &dst, rppHandle_t handle) override {
        return rppt_tensor_or_tensor(src[0].data, src[1].data, src[0].gdescPtr, src[1].gdescPtr,
                                     dst[0].data, dst[0].gdescPtr, broadcastMode(ctx),
                                     src[0].roiTensor, src[1].roiTensor, handle, ctx.backend);
    }
};

REGISTER_RPP_BENCH("tensor_or_tensor", TensorOrTensorAdapter)

} // namespace rppbench
