/**
 * @file bench_tensor_add_tensor.cpp
 * @brief Adapter for rppt_tensor_add_tensor - element-wise add of two ND tensors.
 *
 * First two-source generic op: it takes two RpptGenericDesc sources (each with
 * its own flat roiTensor) plus a broadcast mode, so it subclasses
 * BroadcastOpAdapter (numSrc()==2, both sources the same shape => broadcasting
 * disabled). No op-specific param tensors. F32 only.
 *
 * Signature:
 *   rppt_tensor_add_tensor(src1, src2, srcGenericDesc1, srcGenericDesc2, dst,
 *                          dstGenericDesc, broadcastMode, roiTensor1, roiTensor2,
 *                          handle, backend)
 */
#include "rpp/bench_generic_adapters.hpp"
#include <rpp/rppt_tensor_arithmetic_operations.h>

#include <vector>

namespace rppbench {

class TensorAddTensorAdapter : public BroadcastOpAdapter {
public:
    std::vector<RpptDataType> supportedDtypes() const override { return {RpptDataType::F32}; }

    void setup(const BenchContext &, std::vector<TensorBuffer> &,
               std::vector<TensorBuffer> &) override {}

    RppStatus run(const BenchContext &ctx, std::vector<TensorBuffer> &src,
                  std::vector<TensorBuffer> &dst, rppHandle_t handle) override {
        return rppt_tensor_add_tensor(src[0].data, src[1].data, src[0].gdescPtr, src[1].gdescPtr,
                                      dst[0].data, dst[0].gdescPtr, broadcastMode(ctx),
                                      src[0].roiTensor, src[1].roiTensor, handle, ctx.backend);
    }
};

REGISTER_RPP_BENCH("tensor_add_tensor", TensorAddTensorAdapter)

} // namespace rppbench
