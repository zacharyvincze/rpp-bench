#!/usr/bin/env python3
"""Per-operator bar charts of RPP benchmark runtimes (Google Benchmark JSON).

Where scatter_results.py crams every op into one panel, this fans out: one
horizontal bar chart *per operator*, written to its own file under an output
directory so you can flip through ops one at a time. Within each chart:

  - one bar per fully-resolved configuration (dtype/layout/batch/size/dst/params)
  - bars sorted by runtime, fastest at the bottom
  - each bar labelled with its exact configuration string
  - bar length = median runtime (log x), coloured by dtype
  - runtime (µs) printed at the end of every bar

Usage:
    python3 plot_per_op.py build/results.json                    # -> build/per_op/<op>.png
    python3 plot_per_op.py build/results.json -o build/plots     # custom directory
    python3 plot_per_op.py build/results.json --per-image        # runtime / batch
    python3 plot_per_op.py build/results.json --linear           # linear x instead of log
"""
import argparse
import json
import os
import re
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# ns multiplier per Google Benchmark time_unit -> nanoseconds
_UNIT_TO_NS = {"ns": 1.0, "us": 1e3, "ms": 1e6, "s": 1e9}

# GitHub-dark-ish palette, matching scatter_results.py.
_BG = "#0d1117"
_FG = "#c9d1d9"
_GRID = "#30363d"

# Stable, high-contrast bar colours cycled across dtypes.
_DTYPE_COLORS = ["#58a6ff", "#f0883e", "#3fb950", "#bc8cff", "#f85149", "#e3b341"]


def _dark_rc():
    plt.rcParams.update({
        "figure.facecolor": _BG, "savefig.facecolor": _BG, "axes.facecolor": _BG,
        "axes.edgecolor": _GRID, "axes.labelcolor": _FG, "text.color": _FG,
        "xtick.color": _FG, "ytick.color": _FG, "grid.color": _GRID,
        "legend.edgecolor": _GRID, "legend.framealpha": 0.85, "font.size": 9,
    })


def parse_name(name):
    fields = {}
    for tok in name.split("/"):
        if ":" in tok:
            key, _, val = tok.partition(":")
            fields[key] = val
    return fields


def _px(size):  # "1920x1080" -> 2073600 (0 if unparseable)
    try:
        w, h = size.split("x")
        return int(w) * int(h)
    except Exception:
        return 0


def config_label(f):
    """Exact, human-readable configuration string (everything but the op)."""
    parts = [f.get("dtype", ""), f.get("layout", ""), f"b{f.get('batch', 0)}", f.get("size", "")]
    if f.get("dst"):
        parts.append(f"→{f['dst']}")   # -> dst for resize
    label = " ".join(p for p in parts if p)
    if f.get("params"):
        label += f"  [{f['params']}]"
    return label


def load(path):
    """Aggregate raw iteration rows into one record per fully-resolved combo."""
    data = json.load(open(path))
    combos = defaultdict(lambda: {"times_ns": []})
    for b in data.get("benchmarks", []):
        if b.get("run_type") == "aggregate" or b.get("error_occurred"):
            continue
        f = parse_name(b["name"])
        if "op" not in f:
            continue
        batch = int(f.get("batch", 0))
        key = (f["op"], f.get("dtype", ""), f.get("layout", ""), batch,
               f.get("size", ""), f.get("dst", ""), f.get("params", ""))
        rec = combos[key]
        rec.update(op=f["op"], dtype=f.get("dtype", ""), layout=f.get("layout", ""),
                   batch=batch, label=config_label(f), backend=f.get("backend", ""),
                   workload=batch * _px(f.get("size", "")))
        rec["times_ns"].append(b["real_time"] * _UNIT_TO_NS.get(b.get("time_unit", "ns"), 1.0))
    recs = []
    for rec in combos.values():
        t = np.array(rec["times_ns"])
        rec["runtime_ns"] = float(np.median(t))
        recs.append(rec)
    return data.get("context", {}), recs


def _fmt_ms(ms):
    if ms < 1:
        return f"{ms:.3f}"
    if ms < 10:
        return f"{ms:.2f}"
    if ms < 100:
        return f"{ms:.1f}"
    return f"{ms:,.0f}"


def plot_op(op, recs, ctx, outdir, per_image, log_x):
    for r in recs:
        r["value_ns"] = r["runtime_ns"] / max(1, r["batch"]) if per_image else r["runtime_ns"]
    recs = sorted(recs, key=lambda r: (r["value_ns"], r["dtype"], r["layout"], r["label"]))

    dtypes = sorted({r["dtype"] for r in recs})
    color = {d: _DTYPE_COLORS[i % len(_DTYPE_COLORS)] for i, d in enumerate(dtypes)}

    y = np.arange(len(recs))
    vals = np.array([r["value_ns"] / 1e6 for r in recs])  # -> milliseconds
    colors = [color[r["dtype"]] for r in recs]

    fig, ax = plt.subplots(figsize=(11, max(3.0, 0.34 * len(recs) + 1.2)))
    ax.barh(y, vals, color=colors, edgecolor=_GRID, linewidth=0.5, zorder=3)
    ax.set_axisbelow(True)
    ax.grid(axis="x", which="both", ls=":", alpha=0.35)
    ax.set_yticks(y)
    ax.set_yticklabels([r["label"] for r in recs], fontsize=8, fontfamily="monospace")
    ax.set_ylim(-0.6, len(recs) - 0.4)
    if log_x:
        ax.set_xscale("log")
    xlab = "median runtime / image (ms)" if per_image else "median runtime (ms)"
    ax.set_xlabel(xlab + ("  ·  log scale" if log_x else ""))

    xmax = vals.max()
    ax.set_xlim(right=xmax * (2.4 if log_x else 1.16))
    for yi, v in zip(y, vals):
        ax.text(v * 1.05 if log_x else v + xmax * 0.01, yi, _fmt_ms(v),
                va="center", ha="left", fontsize=7, color=_FG)

    handles = [Line2D([], [], marker="s", ls="", ms=8, color=color[d], label=f"dtype {d}")
               for d in dtypes]
    ax.legend(handles=handles, loc="lower right", fontsize=8, facecolor=_BG)

    backend = recs[0].get("backend", "?")
    host = ctx.get("host_name", "")
    ax.set_title(f"{op}   ·   {backend} backend" + (f"  ·  {host}" if host else "")
                 + "   (sorted by runtime, lower is better)",
                 fontsize=12, weight="bold", color=_FG)

    fig.tight_layout()
    safe = re.sub(r"[^0-9A-Za-z._-]", "_", op)
    out = os.path.join(outdir, f"{safe}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("json_file")
    ap.add_argument("-o", "--outdir", help="output directory (default: <input_dir>/per_op)")
    ap.add_argument("--per-image", action="store_true", help="normalize runtime by batch")
    ap.add_argument("--linear", action="store_true", help="linear x-axis (default: log)")
    args = ap.parse_args()

    _dark_rc()
    ctx, recs = load(args.json_file)
    if not recs:
        sys.exit("no benchmark records found")

    outdir = args.outdir or os.path.join(os.path.dirname(args.json_file) or ".", "per_op")
    os.makedirs(outdir, exist_ok=True)

    by_op = defaultdict(list)
    for r in recs:
        by_op[r["op"]].append(r)

    for op in sorted(by_op):
        out = plot_op(op, by_op[op], ctx, outdir, args.per_image, not args.linear)
        print(f"wrote {out}  ({len(by_op[op])} configs)")
    print(f"\n{len(by_op)} operators -> {outdir}/")


if __name__ == "__main__":
    main()
