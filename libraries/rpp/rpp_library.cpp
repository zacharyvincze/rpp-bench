/**
 * @file rpp_library.cpp
 * @brief RppLibrary implementation + self-registration (see rpp_library.hpp).
 */
#include "rpp/rpp_library.hpp"

#include "rpp/bench_registry.hpp"
#include "rpp/rpp_case.hpp"
#include "rpp/rpp_enums.hpp"

namespace rppbench {

RppLibrary &RppLibrary::instance() {
    static RppLibrary lib;
    return lib;
}

bool RppLibrary::hasOp(const std::string &op) const {
    return OpRegistry::instance().has(op);
}

std::vector<std::string> RppLibrary::opNames() const {
    return OpRegistry::instance().names();
}

std::vector<Backend> RppLibrary::availableBackends() const {
    // A HIP build can still run HOST cases; a HOST-only build has no HIP.
#if RPP_BENCH_HIP
    return {Backend::Host, Backend::Hip};
#else
    return {Backend::Host};
#endif
}

OpCapabilities RppLibrary::capabilities(const std::string &op) const {
    // Probe the adapter once and translate its RPP-typed support lists to the
    // neutral enums the core filters on. (Layout is already the shared neutral
    // enum, so it passes through unchanged.)
    OpCapabilities caps;
    std::unique_ptr<OpAdapter> probe = OpRegistry::instance().find(op)();
    for (RpptDataType d : probe->supportedDtypes())
        caps.dtypes.push_back(to_neutral_dtype(d));
    caps.layouts = probe->supportedLayouts();
    for (RppBackend b : probe->supportedBackends())
        caps.backends.push_back(to_neutral_backend(b));
    return caps;
}

std::unique_ptr<BenchCase> RppLibrary::makeCase(const BenchPoint &point) const {
    return std::make_unique<RppCase>(OpRegistry::instance().find(point.opName), point);
}

// Self-register the RPP library with the core at static-init time. Because the
// RPP module is linked as a CMake OBJECT library, this initializer is guaranteed
// to run (its object is part of the executable), so RPP is always discoverable.
namespace {
const LibraryRegistrar g_rpp_library_registrar(&RppLibrary::instance());
} // namespace

} // namespace rppbench
