// ============================================================================
// bench_name.hpp - the benchmark-name encoding.
//
// A benchmark's name is a parseable, filter-friendly encoding of its fully
// resolved combo (op/backend/dtype/layout/batch/size[/dst][/params]). This is a
// contract: scripts/json2csv.py splits the same format back into columns, and
// --benchmark_filter targets its fields. Keep the two in sync.
// ============================================================================
#ifndef RPP_BENCH_NAME_HPP
#define RPP_BENCH_NAME_HPP

#include "harness/bench_registry.hpp"

#include <string>

namespace rppbench {

// Encode one fully-resolved combo as its benchmark name.
std::string encode_name(const BenchContext &c);

} // namespace rppbench

#endif // RPP_BENCH_NAME_HPP
