#!/usr/bin/env python3
"""Plot RPP benchmark kernel runtimes from Google Benchmark JSON output.

Renders a faceted small-multiples figure:
  - rows    = batch size
  - columns = source image size
  - x-axis  = operation (resize's destination size becomes its own category)
  - bars    = dtype x layout, grouped
  - y-axis  = mean kernel runtime (log scale), error bars = std dev across repetitions

A shared log y-axis makes the scaling across batch, resolution, dtype and layout
directly comparable in one view.

Usage:
    python3 plot_results.py build/results.json -o build/results.png
    python3 plot_results.py build/results.json            # writes <input>.png
"""
import argparse
import json
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np

# ns multiplier per Google Benchmark time_unit -> nanoseconds
_UNIT_TO_NS = {"ns": 1.0, "us": 1e3, "ms": 1e6, "s": 1e9}


def parse_name(name):
    fields = {}
    for tok in name.split("/"):
        if ":" in tok:
            key, _, val = tok.partition(":")
            fields[key] = val
    return fields


def load(path):
    """Return {combo_key: {...fields, times_ns:[...]}} aggregated over iterations."""
    data = json.load(open(path))
    combos = defaultdict(lambda: {"times_ns": []})
    for b in data.get("benchmarks", []):
        if b.get("run_type") == "aggregate":  # recompute from raw iteration rows
            continue
        f = parse_name(b["name"])
        if "op" not in f:
            continue
        key = (f["op"], f.get("dtype", ""), f.get("layout", ""),
               int(f.get("batch", 0)), f.get("size", ""), f.get("dst", ""))
        rec = combos[key]
        rec.update(op=f["op"], dtype=f.get("dtype", ""), layout=f.get("layout", ""),
                   batch=int(f.get("batch", 0)), size=f.get("size", ""),
                   dst=f.get("dst", ""), backend=f.get("backend", ""))
        rec["times_ns"].append(b["real_time"] * _UNIT_TO_NS.get(b.get("time_unit", "ns"), 1.0))
    for rec in combos.values():
        t = np.array(rec["times_ns"])
        rec["mean_ns"] = float(t.mean())
        rec["std_ns"] = float(t.std())
    return data.get("context", {}), list(combos.values())


def op_label(rec):
    return f"resize→{rec['dst']}" if rec["op"] == "resize" and rec["dst"] else rec["op"]


def sort_key_size(s):  # "1920x1080" -> pixel count, for ordering panels
    try:
        w, h = s.split("x")
        return int(w) * int(h)
    except Exception:
        return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json_file")
    ap.add_argument("-o", "--output", help="PNG path (default: <input>.png)")
    args = ap.parse_args()

    ctx, recs = load(args.json_file)
    if not recs:
        sys.exit("no benchmark records found")

    batches = sorted({r["batch"] for r in recs})
    sizes = sorted({r["size"] for r in recs}, key=sort_key_size)
    op_labels = sorted({op_label(r) for r in recs},
                       key=lambda s: (s.startswith("resize"), s))
    series = sorted({(r["dtype"], r["layout"]) for r in recs})

    # Color by dtype (family), distinguish layout by shade.
    dtypes = sorted({d for d, _ in series})
    base = {dtypes[0]: plt.cm.Blues, dtypes[-1]: plt.cm.Oranges}
    layouts = sorted({l for _, l in series})
    shade = {l: 0.55 + 0.30 * i / max(1, len(layouts) - 1) for i, l in enumerate(layouts)}
    color = {(d, l): base.get(d, plt.cm.Greens)(shade[l]) for d, l in series}

    # index for quick lookup
    idx = {(op_label(r), r["dtype"], r["layout"], r["batch"], r["size"]): r for r in recs}

    nrows, ncols = len(batches), len(sizes)
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 3.2 * nrows),
                             sharey=True, squeeze=False)

    x = np.arange(len(op_labels))
    width = 0.8 / len(series)

    # Explicit shared log limits so small (launch-latency-bound) bars aren't
    # clipped by autoscale; covers the full runtime range across all panels.
    all_us = [r["mean_ns"] / 1e3 for r in recs if r["mean_ns"] > 0]
    ylo, yhi = min(all_us) * 0.6, max(all_us) * 1.7

    for ri, batch in enumerate(batches):
        for ci, size in enumerate(sizes):
            ax = axes[ri][ci]
            for si, (dt, lay) in enumerate(series):
                means, errs = [], []
                for ol in op_labels:
                    r = idx.get((ol, dt, lay, batch, size))
                    means.append(r["mean_ns"] / 1e3 if r else np.nan)   # -> microseconds
                    errs.append(r["std_ns"] / 1e3 if r else 0.0)
                offset = (si - (len(series) - 1) / 2) * width
                ax.bar(x + offset, means, width, yerr=errs, capsize=2,
                       color=color[(dt, lay)], label=f"{dt}/{lay}",
                       error_kw=dict(lw=0.8, alpha=0.7))
            ax.set_yscale("log")
            ax.set_ylim(ylo, yhi)
            ax.set_xticks(x)
            ax.set_xticklabels(op_labels, rotation=20, ha="right", fontsize=8)
            ax.grid(axis="y", which="both", ls=":", alpha=0.4)
            if ri == 0:
                ax.set_title(f"image {size}", fontsize=11, weight="bold")
            if ci == 0:
                ax.set_ylabel(f"batch {batch}\n\nmean runtime (µs)", fontsize=9)

    # Single shared legend.
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, title="dtype / layout", loc="upper center",
               ncol=len(series), bbox_to_anchor=(0.5, 1.0), fontsize=9)

    backend = recs[0].get("backend", "?")
    host = ctx.get("host_name", "")
    fig.suptitle(f"RPP kernel runtimes  ·  {backend} backend"
                 + (f"  ·  {host}" if host else "")
                 + "   (log scale, lower is better)",
                 y=1.035, fontsize=13, weight="bold")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = args.output or (args.json_file.rsplit(".", 1)[0] + ".png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    print(f"wrote {out}  ({len(recs)} combos, {nrows}x{ncols} panels)")


if __name__ == "__main__":
    main()
