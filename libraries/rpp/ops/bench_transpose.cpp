/**
 * @file bench_transpose.cpp
 * @brief Adapter for rppt_transpose - permute the axes of a generic ND tensor.
 *
 * First generic-descriptor op: it uses RpptGenericDesc (not the 4D RpptDesc) and
 * a flat begin/length roiTensor, so it subclasses GenericOpAdapter. The source is
 * the standard image-derived ND tensor; the destination has the same dims with
 * its spatial axes permuted. permTensor indexes the source's spatial dims (batch
 * excluded) and defaults to swapping the two outermost spatial axes (H<->W); set
 * a "perm" param (comma-separated spatial indices) to override.
 *
 * Signature:
 *   rppt_transpose(src, srcGenericDesc, dst, dstGenericDesc, permTensor,
 *                  roiTensor, handle, backend)
 */
#include "rpp/bench_generic_adapters.hpp"
#include <rpp/rppt_tensor_geometric_augmentations.h>

#include <algorithm>
#include <sstream>
#include <vector>

namespace rppbench {

class TransposeAdapter : public GenericOpAdapter {
public:
    std::vector<TensorSpec> dstSpecs(const BenchContext &ctx) const override {
        TensorSpec spec = tensorSpec(ctx);
        const int nSpatial = static_cast<int>(spec.dims.size()) - 1;
        const std::vector<Rpp32u> perm = permutation(ctx, nSpatial);
        std::vector<Rpp32u> src = spec.dims;
        for (int i = 0; i < nSpatial; ++i)
            spec.dims[1 + i] = src[1 + perm[i]]; // dst spatial[i] = src spatial[perm[i]]
        return {spec};
    }

    void setup(const BenchContext &ctx, std::vector<TensorBuffer> &src,
               std::vector<TensorBuffer> &) override {
        isHip_ = ctx.isHip;
        const int nSpatial = src[0].numDims - 1;
        const std::vector<Rpp32u> perm = permutation(ctx, nSpatial);
        perm_ = static_cast<Rpp32u *>(bench_pinned_alloc(sizeof(Rpp32u) * nSpatial, ctx.isHip));
        std::copy(perm.begin(), perm.end(), perm_);
    }

    RppStatus run(const BenchContext &ctx, std::vector<TensorBuffer> &src,
                  std::vector<TensorBuffer> &dst, rppHandle_t handle) override {
        return rppt_transpose(src[0].data, src[0].gdescPtr, dst[0].data, dst[0].gdescPtr, perm_,
                              src[0].roiTensor, handle, ctx.backend);
    }

    void teardown() override {
        bench_pinned_free(perm_, isHip_);
        perm_ = nullptr;
    }

private:
    // Spatial-axis permutation: an explicit "perm" param (e.g. "1,0,2") if given
    // and well-formed, else swap the two outermost spatial axes (identity when
    // there is only one spatial axis).
    std::vector<Rpp32u> permutation(const BenchContext &ctx, int nSpatial) const {
        std::vector<Rpp32u> p(nSpatial);
        for (int i = 0; i < nSpatial; ++i)
            p[i] = static_cast<Rpp32u>(i);
        const auto spec = ctx.param<std::string>("perm", std::string());
        if (!spec.empty()) {
            std::vector<Rpp32u> parsed;
            std::stringstream ss(spec);
            std::string tok;
            while (std::getline(ss, tok, ','))
                if (!tok.empty())
                    parsed.push_back(static_cast<Rpp32u>(std::stoul(tok)));
            if (static_cast<int>(parsed.size()) == nSpatial && isPermutation(parsed))
                return parsed;
        }
        if (nSpatial >= 2)
            std::swap(p[0], p[1]);
        return p;
    }

    static bool isPermutation(const std::vector<Rpp32u> &p) {
        std::vector<Rpp32u> s = p;
        std::sort(s.begin(), s.end());
        for (size_t i = 0; i < s.size(); ++i)
            if (s[i] != i)
                return false;
        return true;
    }

    Rpp32u *perm_ = nullptr;
    bool isHip_ = false;
};

REGISTER_RPP_BENCH("transpose", TransposeAdapter)

} // namespace rppbench
