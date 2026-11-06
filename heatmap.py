"""
Heat map of a PoisonedRAG metric over injections (n) and retrieved passages (k).

Reads a report made by report.py: the newest reports/report_*.csv by default,
or any report .csv / .md you pass in.

Usage:
    python3 heatmap.py                                  # poisoned ASR, newest report
    python3 heatmap.py --metric poisoned_acc            # correct answers under attack
    python3 heatmap.py --metric f1                      # retrieval F1
    python3 heatmap.py reports/report_X.md --model llama2-7b --out figures/asr.png
"""
import argparse
import csv
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

METRICS = {
    "poisoned_asr": "Attack success rate (ASR)",
    "poisoned_acc": "Correct-answer accuracy under attack",
    "f1": "Retrieval F1 of injected texts",
    "flip_rate": "Flip rate",
}

# Markdown column header -> CSV field name
MD_COLUMNS = {
    "Dataset": "dataset", "Retriever": "retriever", "Model": "model", "n": "n", "k": "k",
    "ASR (base)": "baseline_asr", "ASR (poison)": "poisoned_asr", "Flip rate": "flip_rate",
    "Acc (base)": "baseline_acc", "Acc (poison)": "poisoned_acc", "Retrieval F1": "f1",
}

# Red (bad) -> yellow -> green (good). For metrics where higher means the attack
# is working better, the ramp is flipped so high values show red.
RAMP = ["#c0392b", "#e8743b", "#f5b041", "#f7dc6f", "#b7d968", "#6cbf5c", "#2e8b57"]
HIGHER_IS_BAD = {"poisoned_asr", "f1", "flip_rate"}

# Baseline counterpart of each metric, used for the n = 0 column.
# Retrieval F1 and flip rate don't exist without injections, so n = 0 stays blank for those.
BASELINE_FIELD = {"poisoned_asr": "baseline_asr", "poisoned_acc": "baseline_acc"}
INK = "#1a1a19"
MUTED = "#6b6b66"


def load_rows(path):
    if path.endswith(".csv"):
        with open(path) as f:
            return list(csv.DictReader(f))
    rows, header = [], None
    for line in open(path):
        line = line.strip()
        if not line.startswith("|") or set(line) <= set("|-: "):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if header is None:
            header = [MD_COLUMNS.get(c, c) for c in cells]
        else:
            row = dict(zip(header, cells))
            # Markdown flip-rate cells look like "0.62 (59/95)"
            if row.get("flip_rate"):
                row["flip_rate"] = row["flip_rate"].split(" ")[0]
            rows.append(row)
    return rows


def number(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("report", nargs="?", help="report .csv or .md (default: newest reports/report_*.csv)")
    ap.add_argument("--metric", default="poisoned_asr", choices=METRICS)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--retriever", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default=None, help="output image (default: figures/heatmap_<metric>.png)")
    ap.add_argument("--full-range", action="store_true",
                    help="color scale from 0 to 1 instead of fitting the data's own range")
    ap.add_argument("--vmin", type=float, default=None, help="manual lower color limit")
    ap.add_argument("--vmax", type=float, default=None, help="manual upper color limit")
    args = ap.parse_args()

    path = args.report
    if path is None:
        found = sorted(glob.glob("reports/report_*.csv"))
        if not found:
            sys.exit("No reports found. Run report.py first, or pass a report file.")
        path = found[-1]

    rows = load_rows(path)
    for field in ("dataset", "retriever", "model"):
        wanted = getattr(args, field)
        if wanted:
            rows = [r for r in rows if r[field] == wanted]
    configs = sorted({(r["dataset"], r["retriever"], r["model"]) for r in rows})
    if not configs:
        sys.exit("No runs match those filters.")
    if len(configs) > 1:
        sys.exit("Report has several configurations; pick one with --dataset/--retriever/--model:\n  "
                 + "\n  ".join(" / ".join(c) for c in configs))
    dataset, retriever, model = configs[0]

    # Collect cells. Poisoned runs fill (n, k); baselines fill the n = 0 column,
    # using the baseline version of the metric where one exists.
    cells = {}
    for r in rows:
        k = int(r["k"])
        if number(r.get("n")) is not None:
            v = number(r.get(args.metric))
            if v is not None:
                cells[(int(float(r["n"])), k)] = v
        base_field = BASELINE_FIELD.get(args.metric)
        if base_field and number(r.get(base_field)) is not None:
            cells[(0, k)] = number(r[base_field])

    ns = sorted({n for n, _ in cells})
    ks = sorted({k for _, k in cells})
    # With k = 0 nothing is retrieved, so injections can't reach the model:
    # every n at k = 0 equals the n = 0 (baseline) run.
    if (0, 0) in cells:
        for n in ns:
            cells[(n, 0)] = cells[(0, 0)]

    grid = np.full((len(ks), len(ns)), np.nan)
    for (n, k), v in cells.items():
        grid[ks.index(k), ns.index(n)] = v
    if np.isnan(grid).all():
        sys.exit(f"No values for {args.metric} in {path}.")

    plt.rcParams.update({"font.family": ["DejaVu Sans"], "font.size": 11})
    ramp = RAMP[::-1] if args.metric in HIGHER_IS_BAD else RAMP
    cmap = LinearSegmentedColormap.from_list("red_green", ramp)
    cmap.set_bad("#f0efec")

    # Color limits: fit the data by default so small differences stay visible
    if args.full_range:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = np.nanmin(grid), np.nanmax(grid)
        if vmax - vmin < 1e-9:  # all cells equal
            vmin, vmax = vmin - 0.05, vmax + 0.05
    vmin = args.vmin if args.vmin is not None else vmin
    vmax = args.vmax if args.vmax is not None else vmax

    fig, ax = plt.subplots(figsize=(1.2 * len(ns) + 2.6, 0.62 * len(ks) + 2.0), dpi=200)
    fig.patch.set_facecolor("white")
    im = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto", origin="lower")

    # 2px white gaps between cells
    ax.set_xticks(np.arange(len(ns) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(ks) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", length=0)

    for i in range(len(ks)):
        for j in range(len(ns)):
            v = grid[i, j]
            if np.isnan(v):
                ax.text(j, i, "–", ha="center", va="center", color=MUTED)
                continue
            frac = np.clip((v - vmin) / (vmax - vmin), 0, 1)
            dark_cell = np.array(cmap(frac)[:3]) @ np.array([0.2126, 0.7152, 0.0722]) < 0.4
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10.5,
                    color="white" if dark_cell else INK)

    ax.set_xticks(range(len(ns)), [str(n) for n in ns])
    ax.set_yticks(range(len(ks)), [str(k) for k in ks])
    ax.set_xlabel("Injected texts per question (n)", color=INK)
    ax.set_ylabel("Passages retrieved (k)", color=INK)
    ax.tick_params(length=0, colors=INK)
    for s in ax.spines.values():
        s.set_visible(False)

    ax.set_title(f"{METRICS[args.metric]}\n", loc="left", fontsize=13, color=INK, fontweight="semibold")
    notes = [f"{dataset.upper()} · {retriever} · {model}"]
    if 0 in ns:
        notes.append("n = 0: no injections (baseline)")
    if 0 in ks:
        notes.append("k = 0: no retrieval")
    ax.text(0, 1.02, "   |   ".join(notes), transform=ax.transAxes,
            fontsize=9.5, color=MUTED)

    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, colors=MUTED, labelsize=9)

    out = args.out or f"figures/heatmap_{args.metric}.png"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
