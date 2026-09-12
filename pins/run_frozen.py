"""Run named (arm, sizer, family) cells over the FROZEN window set, one JSONL row per cell-window.

Every arm of the 2x2 reads the same windows, pool and deadline rule from `pins/frozen_2x2.json`, so
the pairing the analysis depends on is a property of this driver rather than of whoever typed the
commands. Rows are appended the moment a run returns and completed keys are skipped on restart,
because the login node reaps a shell at ~12-15 CPU-min: run it in as many chunks as it takes.

    .venv/bin/python -m pins.run_frozen --cells least_laxity:by_size,least_laxity:as_requested \
        --out runs/frozen_sizer --limit 8
    .venv/bin/python -m pins.run_frozen --out runs/frozen_sizer --report
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from pins.elastisim_bench import run

CONFIG = Path(__file__).with_name("frozen_2x2.json")
METRICS = ("mean_wait_s", "p90_wait_s", "sla10_viol_pct", "sla5_viol_pct", "sla2_viol_pct",
           "mean_bsd", "util_win", "killed_pct", "jain_user_wait", "resize_events", "n",
           "calls", "sel_invalid", "fallbacks")


def cells(spec: str) -> list[tuple[str, str, str]]:
    """`arm:sizer[:family]`, comma separated. Sizer defaults to as_requested, family to none."""
    out = []
    for part in spec.split(","):
        bits = (part.strip().split(":") + ["", ""])[:3]
        out.append((bits[0], bits[1] or "as_requested", bits[2]))
    return out


def tag_of(arm: str, sizer: str, family: str) -> str:
    return f"fz_{arm}_{sizer}" + (f"_{family}" if family else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", default="", help="arm:sizer[:family] list")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--worlds", type=Path, default=Path("runs/holdout"))
    ap.add_argument("--model", default="qwen2.5:14b")
    ap.add_argument("--limit", type=int, default=0, help="stop after N windows per cell")
    ap.add_argument("--max-runs", type=int, default=0, help="stop after N runs this invocation")
    ap.add_argument("--labelled-only", action="store_true",
                    help="restrict to the 54 windows that carry counterfactual labels")
    ap.add_argument("--report", action="store_true", help="summarise what is already on disk")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    sink = a.out / "rows.jsonl"
    rows = [json.loads(l) for l in sink.read_text().splitlines() if l.strip()] if sink.exists() else []

    if a.report:
        return report(rows)

    cfg = json.loads(CONFIG.read_text())
    windows = [w["window"] for w in cfg["windows"]
               if w["has_oracle_labels"] or not a.labelled_only]
    if a.limit:
        windows = windows[:a.limit]
    done = {(r["window"], r["arm"], r["sizer"], r["family"]) for r in rows}
    n = 0
    for arm, sizer, family in cells(a.cells):
        for w in windows:
            if (w, arm, sizer, family) in done:
                continue
            res = run(a.worlds / w, arm, a.model, tag=tag_of(arm, sizer, family), quiet=True,
                      sizer=sizer, family=family)
            row = {"window": w, "arm": arm, "sizer": sizer, "family": family,
                   **{k: res.get(k) for k in METRICS}}
            with open(sink, "a") as f:
                f.write(json.dumps(row) + "\n")
            rows.append(row)
            n += 1
            print(f"  {w} {arm}+{sizer}{'/' + family if family else ''}: "
                  f"wait {res.get('mean_wait_s')}s sla10 {res.get('sla10_viol_pct')}% "
                  f"util {res.get('util_win')}", flush=True)
            if a.max_runs and n >= a.max_runs:
                print(f"chunk budget spent after {n} runs", flush=True)
                return report(rows)
    report(rows)


def report(rows: list[dict]) -> None:
    """Per-cell means, plus a PAIRED delta against the first cell on the windows both ran."""
    by: dict = {}
    for r in rows:
        by.setdefault((r["arm"], r["sizer"], r["family"]), {})[r["window"]] = r
    if not by:
        return print("no rows yet")
    keys = list(by)
    base = keys[0]
    print(f"\n{'cell':<34} {'n':>3} {'wait_s':>9} {'p90_wait':>9} {'sla10':>7} {'bsd':>6} "
          f"{'util':>6} {'d wait vs base':>15}")
    for k in keys:
        d = by[k]
        shared = sorted(set(d) & set(by[base]))
        dw = (statistics.mean(d[w]["mean_wait_s"] - by[base][w]["mean_wait_s"] for w in shared)
              if shared else float("nan"))
        name = f"{k[0]}+{k[1]}" + (f"/{k[2]}" if k[2] else "")
        print(f"{name:<34} {len(d):>3} {statistics.mean(x['mean_wait_s'] for x in d.values()):>9.0f} "
              f"{statistics.mean(x['p90_wait_s'] for x in d.values()):>9.0f} "
              f"{statistics.mean(x['sla10_viol_pct'] for x in d.values()):>7.2f} "
              f"{statistics.mean(x['mean_bsd'] for x in d.values()):>6.2f} "
              f"{statistics.mean(x['util_win'] for x in d.values()):>6.3f} "
              f"{dw:>+15.0f}  (n={len(shared)} paired)")
    print(f"base = {base[0]}+{base[1]}{'/' + base[2] if base[2] else ''}")


if __name__ == "__main__":
    main()
