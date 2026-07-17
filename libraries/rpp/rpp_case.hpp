/**
 * @file rpp_case.hpp
 * @brief RppCase - the RPP implementation of the core BenchCase.
 *
 * Holds all of one sweep point's RPP resources (handle, HIP stream, src/dst
 * TensorBuffers, the op adapter) and drives them. This is where the old
 * CaseResources::build/run_one logic now lives - the allocation, descriptor
 * setup, fill, timed rppt_* call, and device synchronization that the core
 * deliberately does not know about. RppLibrary::makeCase() builds one per case.
 */
#ifndef RPP_BENCH_RPP_CASE_HPP
#define RPP_BENCH_RPP_CASE_HPP

#include "core/bench_library.hpp"
#include "core/bench_point.hpp"
#include "rpp/bench_registry.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace rppbench {

/**
 * @brief One built, ready-to-run RPP sweep point (implements BenchCase).
 *
 * Buffer allocation is enclosed here: image ops go through TensorBuffer::init(),
 * generic-descriptor ops through initGeneric(), driven by the adapter's
 * srcSpecs()/dstSpecs()/halo hooks. The destructor frees everything and runs the
 * adapter's teardown, so the core releases a case by dropping the pointer.
 */
class RppCase : public BenchCase {
public:
    /**
     * @brief Build (but do not yet set up) a case.
     * @param factory Factory for the op adapter (from the RPP op registry).
     * @param point The resolved, neutral sweep point (converted to a BenchContext).
     */
    RppCase(AdapterFactory factory, const BenchPoint &point);
    ~RppCase() override;

    std::string setup() override;
    std::string runOnce() override;
    size_t srcDataBytes() const override;

private:
    AdapterFactory factory_;
    BenchContext ctx_;
    std::unique_ptr<OpAdapter> adapter_;
    std::vector<TensorBuffer> srcs_, dsts_;
    rppHandle_t handle_ = nullptr;
    void *hipStream_ = nullptr; // hipStream_t (opaque here to keep HIP out of the header)
    bool ready_ = false;
};

} // namespace rppbench

#endif // RPP_BENCH_RPP_CASE_HPP
