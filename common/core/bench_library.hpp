/**
 * @file bench_library.hpp
 * @brief The Library plugin interface, the BenchCase it produces, and the registry.
 *
 * A Library is one imaging implementation (today only RPP). It is the sole
 * extension point the core knows about: it owns its own operators, reports what
 * they support (in neutral terms, for sweep filtering), and turns a BenchPoint
 * into a runnable BenchCase. All library- and device-specific work - descriptor
 * setup, allocation, the timed call, device synchronization - lives behind these
 * interfaces, so the core never includes a library or HIP header.
 *
 * Libraries self-register into the global LibraryRegistry at static-init time
 * (see LibraryRegistrar). Each library also owns its own op registry internally;
 * there is deliberately no global op table in the core.
 */
#ifndef RPP_BENCH_CORE_LIBRARY_HPP
#define RPP_BENCH_CORE_LIBRARY_HPP

#include "core/bench_point.hpp"
#include "core/bench_types.hpp"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace rppbench {

/**
 * @brief What an op supports, in neutral terms, for matrix filtering.
 *
 * A library reports this per op so the runner can skip combos the op does not
 * implement without the core knowing any library-specific detail.
 */
struct OpCapabilities {
    std::vector<DataType> dtypes;
    std::vector<Layout> layouts;
    std::vector<Backend> backends;
};

/**
 * @brief One built, ready-to-run sweep point.
 *
 * Holds all the case's library resources (buffers, session/handle, adapter). The
 * core builds one per case, reuses it across Google Benchmark's re-invocations,
 * and drives it through this neutral surface. The destructor releases everything
 * (and runs any op teardown), so releasing a case is just resetting the pointer.
 */
class BenchCase {
public:
    virtual ~BenchCase() = default;

    /**
     * @brief Allocate buffers, create the session, and run op setup. Untimed.
     * @return Empty on success, else a human-readable error (the case is skipped).
     */
    virtual std::string setup() = 0;
    /**
     * @brief One invocation of the op: refresh ROI, launch, and synchronize. Timed.
     *
     * The device synchronization ("the op has finished") lives here, so the core
     * timing loop needs no device headers.
     * @return Empty on success, else a human-readable error.
     */
    virtual std::string runOnce() = 0;
    /**
     * @brief Logical bytes of the primary source, for the throughput counter.
     * @return Byte count, or 0 if not applicable.
     */
    virtual size_t srcDataBytes() const = 0;
};

/**
 * @brief One imaging implementation the harness can benchmark.
 *
 * Owns its operators and knows how to build a case for a resolved sweep point.
 * The concrete subclass lives entirely in its library module (libraries/<name>).
 */
class Library {
public:
    virtual ~Library() = default;

    /** @brief The library's config/registry name, e.g. "rpp". */
    virtual std::string name() const = 0;
    /** @brief Whether this library provides an op under @p op. */
    virtual bool hasOp(const std::string &op) const = 0;
    /** @brief Every op name this library provides (for --list-ops). */
    virtual std::vector<std::string> opNames() const = 0;
    /**
     * @brief Backends actually compiled into this library build (e.g. HOST, HIP).
     *
     * Distinct from an op's supportedBackends(): this reflects what the linked
     * library can do at all (a HOST-only build reports only HOST), letting the
     * runner skip a backend the build cannot provide with a clear note.
     */
    virtual std::vector<Backend> availableBackends() const = 0;
    /**
     * @brief The neutral capability report for one op (for matrix filtering).
     * @param op Op name (must satisfy hasOp()).
     */
    virtual OpCapabilities capabilities(const std::string &op) const = 0;
    /**
     * @brief Build a runnable case for a fully-resolved sweep point.
     * @param point The resolved combo (its library == name()).
     * @return The built (but not yet set-up) case.
     */
    virtual std::unique_ptr<BenchCase> makeCase(const BenchPoint &point) const = 0;
};

/**
 * @brief Global registry of libraries. Libraries self-register at static-init.
 *
 * The one registry the core owns. Op tables live inside each Library, not here.
 */
class LibraryRegistry {
public:
    static LibraryRegistry &instance();
    /** @brief Register a library instance (owned by the caller for the process life). */
    void add(Library *lib);
    /** @brief Look up a library by name, or nullptr if none is registered. */
    Library *find(const std::string &name) const;
    /** @brief Every registered library. */
    const std::vector<Library *> &all() const { return libs_; }

private:
    std::vector<Library *> libs_;
};

/**
 * @brief Helper that registers a library at static-init time.
 *
 * Place one at file scope in a library module. Keep the library instance alive
 * for the process lifetime (a function-local static is the usual pattern).
 */
struct LibraryRegistrar {
    explicit LibraryRegistrar(Library *lib) { LibraryRegistry::instance().add(lib); }
};

} // namespace rppbench

#endif // RPP_BENCH_CORE_LIBRARY_HPP
