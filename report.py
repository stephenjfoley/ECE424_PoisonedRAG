"""
Build a summary report of all PoisonedRAG runs launched with run.py.

Reads the run cache (updating it from the logs first; see crawl.py), takes the
most recent run for every (dataset, retriever, model, n, k), and pairs each
poisoned run with the baseline (n = 0) at the same k to compute the flip rate.

Usage:
    python3 report.py                 # writes reports/report_<timestamp>.md and .csv
    python3 report.py --no-save       # print only
"""
import argparse
import csv
import os
from datetime import datetime

from crawl import flip_rate, latest_runs, update_cache

REPORT_DIR = "reports"


def fmt(x, digits=2):
    return "–" if x is None else f"{x:.{digits}f}"


def build_rows(latest):
    """One row per poisoned (n, k) run, paired with the baseline at the same k.
    Baselines with no poisoned run at their k get a row of their own."""
    poisoned = {key for key in latest if key[3] > 0}
    keys = set(poisoned)
    for d, r, m, n, k in latest:
        if n == 0 and not any(p[:3] == (d, r, m) and p[4] == k for p in poisoned):
            keys.add((d, r, m, 0, k))

    rows = []
    for d, r, m, n, k in sorted(keys, key=lambda c: (c[0], c[1], c[2], c[4], c[3])):
        base = latest.get((d, r, m, 0, k))
        pois = latest.get((d, r, m, n, k)) if n > 0 else None
        flip, flips, eligible = flip_rate(base, pois) if base and pois else (None, None, None)
        when = lambda run: run["timestamp"].replace("T", " ")[:16] if run else None
        rows.append({
            "dataset": d, "retriever": r, "model": m,
            "n": n if n > 0 else None, "k": k,
            "baseline_asr": base["asr"] if base else None,
            "poisoned_asr": pois["asr"] if pois else None,
            "flip_rate": flip, "flips": flips, "eligible": eligible,
            "baseline_acc": base["accuracy"] if base else None,
            "poisoned_acc": pois["accuracy"] if pois else None,
            "f1": pois["f1"] if pois else None,
            "baseline_run": when(base), "poisoned_run": when(pois),
        })
    return rows


def to_markdown(rows):
    header = ["Dataset", "Retriever", "Model", "n", "k", "ASR (base)", "ASR (poison)", "Flip rate",
              "Acc (base)", "Acc (poison)", "Retrieval F1", "Baseline run", "Poisoned run"]
    # Text columns left-aligned, numeric columns right-aligned
    right = [False, False, False, True, True, True, True, True, True, True, True, False, False]
    table = []
    for r in rows:
        flip = f"{fmt(r['flip_rate'])} ({r['flips']}/{r['eligible']})" if r["flip_rate"] is not None else "–"
        table.append([
            r["dataset"], r["retriever"], r["model"],
            "–" if r["n"] is None else str(r["n"]), str(r["k"]),
            fmt(r["baseline_asr"]), fmt(r["poisoned_asr"]), flip,
            fmt(r["baseline_acc"]), fmt(r["poisoned_acc"]), fmt(r["f1"]),
            r["baseline_run"] or "–", r["poisoned_run"] or "–",
        ])

    # Pad every column to the same width so the table lines up in a terminal
    # (it stays valid Markdown, and the ":" markers keep the alignment when rendered)
    widths = [max(len(row[i]) for row in [header] + table) for i in range(len(header))]

    def line(cells):
        padded = [c.rjust(w) if ra else c.ljust(w) for c, w, ra in zip(cells, widths, right)]
        return "| " + " | ".join(padded) + " |"

    out = [
        "# PoisonedRAG results",
        "",
        f"Generated {datetime.now():%Y-%m-%d %H:%M}. Most recent completed run per configuration.",
        "",
        line(header),
        "|" + "|".join(("-" * (w + 1) + ":") if ra else ("-" * (w + 2)) for w, ra in zip(widths, right)) + "|",
        *[line(row) for row in table],
        "",
        "**ASR**: fraction of answers containing the attacker's target answer (substring match, as in the paper).  ",
        "**Flip rate**: among questions whose baseline answer does not contain the target, the fraction the poisoned run turns into the target.  ",
        "**Acc**: fraction of answers containing the correct answer (substring match).  ",
        "**n**: injected texts per target question. **k**: passages retrieved and given to the LLM. Each poisoned run is paired with the baseline at the same k.  ",
        "**Retrieval F1**: balance of precision (injected texts / k) and recall (injected texts retrieved / n).",
    ]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser(description="Summarize PoisonedRAG runs.")
    ap.add_argument("--no-save", action="store_true", help="print the report without writing files")
    args = ap.parse_args()

    cache, _ = update_cache()
    latest = latest_runs(cache)
    if not latest:
        print("No completed runs found (run.py writes logs to logs/main_logs/).")
        return

    rows = build_rows(latest)
    md = to_markdown(rows)
    print(md)

    if not args.no_save:
        os.makedirs(REPORT_DIR, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        md_path = os.path.join(REPORT_DIR, f"report_{stamp}.md")
        csv_path = os.path.join(REPORT_DIR, f"report_{stamp}.csv")
        with open(md_path, "w") as f:
            f.write(md)
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"Saved {md_path} and {csv_path}")


if __name__ == "__main__":
    main()
