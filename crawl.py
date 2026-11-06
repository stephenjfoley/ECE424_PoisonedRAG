"""
Crawl the run logs and keep a cache of every completed run's statistics.

The cache (results/run_cache.json) holds one entry per run: dataset, retriever,
model, n, k, timestamp, the summary metrics from the log, and per-question
outcomes from the results file, so flip rates can be computed later without
re-reading anything. Only new or changed logs are parsed on each crawl.

report.py and heatmap.py crawl automatically, so running this by hand is optional.

Usage:
    python3 crawl.py              # update the cache, print a summary
    python3 crawl.py --list       # also list every cached run (full history)
    python3 crawl.py --rebuild    # discard the cache and re-parse all logs
"""
import argparse
import glob
import json
import os
import re
from datetime import datetime

LOG_GLOBS = ["logs/main_logs/*.txt"]
RESULTS_DIR = "results/query_results/main"
CACHE_PATH = "results/run_cache.json"
CACHE_VERSION = 1

# <dataset>-<retriever>-<model>-<mode>[-n<N>-k<K>][-<YYYYmmdd-HHMMSS>]
# Runs from before --n/--k existed have no n/k part and used n=5 (poisoned) or 0 (baseline), k=5.
NAME_RE = re.compile(
    r"^(?P<dataset>nq|hotpotqa|msmarco)-"
    r"(?P<retriever>contriever-msmarco|contriever|ance)-"
    r"(?P<model>.+?)-"
    r"(?P<mode>baseline|poisoned)"
    r"(?:-n(?P<n>\d+)-k(?P<k>\d+))?"
    r"(?:-(?P<stamp>\d{8}-\d{6}))?$"
)


def clean_str(s):
    """Same normalization as src/utils.py, copied so this script needs no torch/beir imports."""
    s = str(s).strip()
    if len(s) > 1 and s[-1] == ".":
        s = s[:-1]
    return s.lower()


def parse_log(path):
    text = open(path, errors="ignore").read()

    def grab(pattern):
        m = re.search(pattern, text)
        return float(m.group(1)) if m else None

    return {
        "asr": grab(r"ASR Mean:\s*([\d.]+)"),
        "precision": grab(r"Precision mean:\s*([\d.]+)"),
        "recall": grab(r"Recall mean:\s*([\d.]+)"),
        "f1": grab(r"F1 mean:\s*([\d.]+)"),
    }


def parse_results(name):
    """Per-question outcomes: {id: [target answer appears, correct answer appears]}."""
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    out = {}
    for it in data:
        for r in list(it.values())[0]:
            out[r["id"]] = [
                clean_str(r["incorrect_answer"]) in clean_str(r["output_poison"]),
                clean_str(r["answer"]) in clean_str(r["output_poison"]),
            ]
    return out


def load_cache():
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH) as f:
                cache = json.load(f)
            if cache.get("version") == CACHE_VERSION:
                return cache
        except (json.JSONDecodeError, OSError):
            pass
    return {"version": CACHE_VERSION, "runs": {}}


def save_cache(cache):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    tmp = CACHE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f, indent=1)
    os.replace(tmp, CACHE_PATH)  # atomic, so a crash never leaves a half-written cache


def update_cache(rebuild=False, verbose=False):
    """Parse new or changed logs into the cache. Returns (cache, number of runs added or updated)."""
    cache = {"version": CACHE_VERSION, "runs": {}} if rebuild else load_cache()
    runs = cache["runs"]
    changed = 0
    for pattern in LOG_GLOBS:
        for path in glob.glob(pattern):
            name = os.path.splitext(os.path.basename(path))[0]
            m = NAME_RE.match(name)
            if not m:
                continue
            mtime = os.path.getmtime(path)
            if name in runs and runs[name]["log_mtime"] == mtime:
                continue  # unchanged since last crawl

            metrics = parse_log(path)
            if metrics["asr"] is None:
                continue  # crashed or still running; picked up on a later crawl once complete
            per_q = parse_results(name)

            mode = m.group("mode")
            stamp = m.group("stamp")
            when = datetime.strptime(stamp, "%Y%m%d-%H%M%S") if stamp else datetime.fromtimestamp(mtime)
            runs[name] = {
                "dataset": m.group("dataset"),
                "retriever": m.group("retriever"),
                "model": m.group("model").removeprefix("ollama-"),
                "n": int(m.group("n")) if m.group("n") else (5 if mode == "poisoned" else 0),
                "k": int(m.group("k")) if m.group("k") else 5,
                "timestamp": when.isoformat(timespec="seconds"),
                **metrics,
                "accuracy": (sum(ok for _, ok in per_q.values()) / len(per_q)) if per_q else None,
                "questions": len(per_q) if per_q else None,
                "per_question": per_q,
                "log": path,
                "log_mtime": mtime,
            }
            changed += 1
            if verbose:
                print(f"  cached {name}")

    # Drop entries whose log has been deleted
    for name in [n for n, r in runs.items() if not os.path.exists(r["log"])]:
        del runs[name]
        changed += 1
    if changed or not os.path.exists(CACHE_PATH):
        save_cache(cache)
    return cache, changed


def latest_runs(cache):
    """Most recent run per (dataset, retriever, model, n, k)."""
    latest = {}
    for name, r in cache["runs"].items():
        key = (r["dataset"], r["retriever"], r["model"], r["n"], r["k"])
        if key not in latest or r["timestamp"] > latest[key]["timestamp"]:
            latest[key] = {"name": name, **r}
    return latest


def flip_rate(baseline, poisoned):
    """Among questions whose baseline answer lacks the target, fraction the poisoned run turns into it."""
    b, p = baseline.get("per_question"), poisoned.get("per_question")
    if not b or not p:
        return None, None, None
    eligible = [q for q in b if q in p and not b[q][0]]
    flips = sum(p[q][0] for q in eligible)
    return (flips / len(eligible) if eligible else None), flips, len(eligible)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rebuild", action="store_true", help="discard the cache and re-parse every log")
    ap.add_argument("--list", action="store_true", help="list every cached run, oldest first")
    args = ap.parse_args()

    cache, changed = update_cache(rebuild=args.rebuild, verbose=True)
    runs = cache["runs"]
    print(f"{len(runs)} run(s) cached in {CACHE_PATH} ({changed} added or updated); "
          f"{len(latest_runs(cache))} distinct configuration(s).")

    if args.list:
        rows = sorted(runs.values(), key=lambda r: r["timestamp"])
        print(f"\n{'timestamp':19}  {'dataset':8} {'retriever':18} {'model':16} {'n':>2} {'k':>3}  {'ASR':>5} {'Acc':>5} {'F1':>5}")
        for r in rows:
            f = lambda x: "  –  " if x is None else f"{x:5.2f}"
            print(f"{r['timestamp'].replace('T', ' ')}  {r['dataset']:8} {r['retriever']:18} {r['model']:16} "
                  f"{r['n']:>2} {r['k']:>3}  {f(r['asr'])} {f(r['accuracy'])} {f(r['f1'])}")


if __name__ == "__main__":
    main()
