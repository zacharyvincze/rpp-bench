#include "harness/bench_registry.hpp"

#include <algorithm>
#include <stdexcept>

namespace rppbench {

OpRegistry &OpRegistry::instance() {
    static OpRegistry reg;
    return reg;
}

void OpRegistry::add(const std::string &name, AdapterFactory factory) {
    entries_.emplace_back(name, std::move(factory));
}

bool OpRegistry::has(const std::string &name) const {
    return std::any_of(entries_.begin(), entries_.end(),
                       [&](const auto &e) { return e.first == name; });
}

AdapterFactory OpRegistry::find(const std::string &name) const {
    for (const auto &e : entries_)
        if (e.first == name)
            return e.second;
    throw std::runtime_error("no registered benchmark adapter named '" + name + "'");
}

std::vector<std::string> OpRegistry::names() const {
    std::vector<std::string> out;
    out.reserve(entries_.size());
    for (const auto &e : entries_)
        out.push_back(e.first);
    return out;
}

} // namespace rppbench
