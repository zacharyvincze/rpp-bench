/**
 * @file bench_library.cpp
 * @brief LibraryRegistry implementation (see bench_library.hpp).
 */
#include "core/bench_library.hpp"

namespace rppbench {

LibraryRegistry &LibraryRegistry::instance() {
    static LibraryRegistry reg;
    return reg;
}

void LibraryRegistry::add(Library *lib) {
    if (lib)
        libs_.push_back(lib);
}

Library *LibraryRegistry::find(const std::string &name) const {
    for (Library *lib : libs_)
        if (lib->name() == name)
            return lib;
    return nullptr;
}

} // namespace rppbench
