/**
 * @file bench_name.hpp
 * @brief The benchmark-name encoding.
 *
 * A benchmark's name is a parseable, filter-friendly encoding of its fully
 * resolved combo (op/library/backend/dtype/layout/batch/size[/dst][/params]).
 * This is a contract: scripts/json2csv.py splits the same format back into
 * columns, and --benchmark_filter targets its fields. Keep the two in sync.
 */
#ifndef RPP_BENCH_CORE_NAME_HPP
#define RPP_BENCH_CORE_NAME_HPP

#include "core/bench_point.hpp"

#include <string>

namespace rppbench {

/**
 * @brief Encode one fully-resolved combo as its benchmark name.
 * @param p The fully-resolved sweep point to encode.
 * @return The encoded benchmark name.
 */
std::string encode_name(const BenchPoint &p);

} // namespace rppbench

#endif // RPP_BENCH_CORE_NAME_HPP
