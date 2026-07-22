#!/usr/bin/env python3
"""Convert Google Benchmark JSON output from rpp_bench into a tidy CSV.

Google Benchmark's built-in CSV reporter is deprecated, so we emit native JSON
(--benchmark_out=results.json --benchmark_out_format=json) and reshape it here.
The benchmark name is an encoded key ("op:brightness/backend:HOST/dtype:U8/..."),
which we split back into columns and join with the numeric counters.

Usage:
    python3 json2csv.py results.json > results.csv
    python3 json2csv.py results.json -o results.csv
"""
import argparse
import csv
import json
import sys

# Categorical fields encoded in the benchmark name, in a stable column order.
# "params" holds the op-specific set as "k1=v1,k2=v2" (empty when the op has none).
NAME_KEYS = ["op", "backend", "dtype", "layout", "batch", "size", "dst", "params"]

# Numeric fields we pull from each benchmark record when present.
METRIC_KEYS = [
    "real_time",
    "cpu_time",
    "time_unit",
    "iterations",
    "images_per_sec",
    "pixels_per_sec",
    "bytes_per_second",
]


def parse_name(name):
    """"op:brightness/backend:HOST/..." -> {"op": "brightness", ...}."""
    fields = {k: "" for k in NAME_KEYS}
    for token in name.split("/"):
        if ":" not in token:
            continue
        key, _, val = token.partition(":")
        if key in fields:
            fields[key] = val
    return fields


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json_file", help="Google Benchmark JSON output")
    ap.add_argument("-o", "--output", help="CSV output path (default: stdout)")
    args = ap.parse_args()

    with open(args.json_file) as fh:
        data = json.load(fh)

    rows = []
    for bench in data.get("benchmarks", []):
        # Skip aggregate rows (mean/median/stddev) so each row is one measurement;
        # aggregates can be recomputed downstream if needed.
        if bench.get("run_type") == "aggregate":
            continue
        row = parse_name(bench.get("name", ""))
        for key in METRIC_KEYS:
            row[key] = bench.get(key, "")
        rows.append(row)

    columns = NAME_KEYS + METRIC_KEYS
    out = open(args.output, "w", newline="") if args.output else sys.stdout
    try:
        writer = csv.DictWriter(out, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    finally:
        if args.output:
            out.close()

    print(f"wrote {len(rows)} rows", file=sys.stderr)


if __name__ == "__main__":
    main()
