/**
 * @file rpp_library.hpp
 * @brief RppLibrary - the RPP implementation of the core Library plugin.
 *
 * The one object the core sees for RPP. It owns the RPP op registry (via
 * OpRegistry), reports each op's capabilities in neutral terms for matrix
 * filtering, reports which backends this build supports, and builds an RppCase
 * per sweep point. It self-registers into the core LibraryRegistry at static-init
 * time (see rpp_library.cpp).
 */
#ifndef RPP_BENCH_RPP_LIBRARY_HPP
#define RPP_BENCH_RPP_LIBRARY_HPP

#include "core/bench_library.hpp"

#include <memory>
#include <string>
#include <vector>

namespace rppbench {

/**
 * @brief The RPP library plugin.
 */
class RppLibrary : public Library {
public:
    /** @brief The process-lifetime instance (registered with the core registry). */
    static RppLibrary &instance();

    std::string name() const override { return "rpp"; }
    bool hasOp(const std::string &op) const override;
    std::vector<std::string> opNames() const override;
    std::vector<Backend> availableBackends() const override;
    OpCapabilities capabilities(const std::string &op) const override;
    std::unique_ptr<BenchCase> makeCase(const BenchPoint &point) const override;
};

} // namespace rppbench

#endif // RPP_BENCH_RPP_LIBRARY_HPP
