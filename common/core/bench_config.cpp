/**
 * @file bench_config.cpp
 * @brief JSON -> BenchConfig parsing (see bench_config.hpp).
 */
#include "core/bench_config.hpp"

#include <fstream>
#include <stdexcept>

using nlohmann::json;

namespace rppbench {

namespace {

// The library used by an op that does not name one, when "defaults" omits it too.
constexpr const char *kDefaultLibrary = "rpp";

std::pair<int, int> parse_size(const json &j) {
    // Accept [w, h] or {"width":..,"height":..}.
    if (j.is_array() && j.size() == 2)
        return {j[0].get<int>(), j[1].get<int>()};
    if (j.is_object())
        return {j.at("width").get<int>(), j.at("height").get<int>()};
    throw std::runtime_error("size must be [width,height] or {width,height}");
}

std::vector<std::pair<int, int>> parse_sizes(const json &j) {
    std::vector<std::pair<int, int>> out;
    for (const auto &e : j)
        out.push_back(parse_size(e));
    return out;
}

/**
 * @brief Overlay any axis present in @p j onto @p m (per-op overrides beat defaults).
 * @param m The matrix to update in place.
 * @param j The JSON object to read axis overrides from.
 */
void merge_matrix(Matrix &m, const json &j) {
    if (j.contains("backends"))
        m.backends = j.at("backends").get<std::vector<std::string>>();
    if (j.contains("dtypes")) {
        m.dtypes.clear();
        for (const auto &s : j.at("dtypes"))
            m.dtypes.push_back(parse_dtype(s.get<std::string>()));
    }
    if (j.contains("layouts")) {
        m.layouts.clear();
        for (const auto &s : j.at("layouts"))
            m.layouts.push_back(parse_layout(s.get<std::string>()));
    }
    if (j.contains("batch_sizes"))
        m.batchSizes = j.at("batch_sizes").get<std::vector<int>>();
    if (j.contains("image_sizes"))
        m.imageSizes = parse_sizes(j.at("image_sizes"));
}

void validate_matrix(const Matrix &m, const std::string &op) {
    auto need = [&](bool ok, const char *axis) {
        if (!ok)
            throw std::runtime_error("op '" + op + "': no " + axis +
                                     " set (provide in defaults or the op)");
    };
    need(!m.backends.empty(), "backends");
    need(!m.dtypes.empty(), "dtypes");
    need(!m.layouts.empty(), "layouts");
    need(!m.batchSizes.empty(), "batch_sizes");
    need(!m.imageSizes.empty(), "image_sizes");
}

} // namespace

BenchConfig load_config(const std::string &path) {
    std::ifstream in(path);
    if (!in)
        throw std::runtime_error("cannot open config file: " + path);

    json root;
    try {
        in >> root;
    } catch (const json::parse_error &e) {
        throw std::runtime_error("JSON parse error in " + path + ": " + e.what());
    }

    BenchConfig cfg;
    if (root.contains("min_time_sec"))
        cfg.minTimeSec = root.at("min_time_sec").get<double>();
    if (root.contains("iterations"))
        cfg.iterations = root.at("iterations").get<int>();
    if (root.contains("repetitions"))
        cfg.repetitions = root.at("repetitions").get<int>();
    if (root.contains("warmup_iterations")) {
        cfg.warmupIterations = root.at("warmup_iterations").get<int>();
        if (cfg.warmupIterations < 0)
            throw std::runtime_error("config \"warmup_iterations\" must be >= 0");
    }
    // Google Benchmark forbids fixing both an iteration count and a min time on
    // one case; reject the ambiguous config up front rather than trip its assert.
    if (cfg.iterations > 0 && cfg.minTimeSec > 0.0)
        throw std::runtime_error("config sets both \"iterations\" and \"min_time_sec\"; "
                                 "they are mutually exclusive (iterations fixes the count, "
                                 "min_time_sec fixes the wall time)");

    Matrix defaults;
    std::string defaultLibrary = kDefaultLibrary;
    if (root.contains("defaults")) {
        const auto &dj = root.at("defaults");
        merge_matrix(defaults, dj);
        if (dj.contains("library"))
            defaultLibrary = dj.at("library").get<std::string>();
    }

    if (!root.contains("ops") || !root.at("ops").is_array())
        throw std::runtime_error("config must contain an \"ops\" array");

    for (const auto &opj : root.at("ops")) {
        OpSpec op;
        op.name = opj.at("name").get<std::string>();
        op.library =
            opj.contains("library") ? opj.at("library").get<std::string>() : defaultLibrary;
        op.matrix = defaults;         // start from defaults
        merge_matrix(op.matrix, opj); // then apply per-op overrides
        validate_matrix(op.matrix, op.name);

        if (opj.contains("dst_sizes"))
            op.dstSizes = parse_sizes(opj.at("dst_sizes"));

        // Param sets: "param_sets" (list of objects, swept) takes precedence over
        // "params" (a single object). Default to one empty set so the op still runs.
        if (opj.contains("param_sets")) {
            const auto &ps = opj.at("param_sets");
            if (!ps.is_array() || ps.empty())
                throw std::runtime_error("op '" + op.name +
                                         "': param_sets must be a non-empty array");
            for (const auto &s : ps) {
                if (!s.is_object())
                    throw std::runtime_error("op '" + op.name +
                                             "': each param_sets entry must be an object");
                op.paramSets.push_back(s);
            }
            if (opj.contains("params"))
                throw std::runtime_error("op '" + op.name +
                                         "': use either params or param_sets, not both");
        } else if (opj.contains("params")) {
            op.paramSets.push_back(opj.at("params"));
        } else {
            op.paramSets.emplace_back(nlohmann::json::object());
        }

        cfg.ops.push_back(std::move(op));
    }

    if (cfg.ops.empty())
        throw std::runtime_error("config has no ops to run");

    return cfg;
}

} // namespace rppbench
