"""Search for a policy-selection rule on TRAINING windows, then freeze it for held-out evaluation.

Two proposers over the same rule space and the same evaluation budget:
  --proposer llm     the model revises its own rule, shown its score and the best pair per window
  --proposer random  uniform samples from the grammar -- the control that says whether the model's
                     proposals are worth anything beyond the search itself

Compliance is not in question here: code executes the rule, so this measures CONTENT.
Usage:
  .venv/bin/python -m pins.rule_train --worlds 'runs/train/*' --iters 12 --proposer llm \
      --out rules/best_llm.json
"""
from __future__ import annotations

import argparse
import glob
import json
import random
import statistics as st
import subprocess
import sys
from pathlib import Path

from pins.elastisim_bench import (POLICY_MENU, RULE_FIELDS, RULE_MAX_BRANCHES, RULE_OPS,
                                  RULE_SYNTH, _packet_synth, _validate_rule)

R = Path(__file__).resolve().parent.parent
PY = str(R / ".venv/bin/python")
# Thresholds a proposer may pick from, per field: the ranges these quantities actually occupy.
GRID = {"offered_load": (0.25, 0.5, 1.0, 1.5, 2.0, 3.0), "queue_depth": (1, 5, 10, 25, 50, 100),
        "free_gpus": (0, 1, 4, 8, 16, 40), "prod_waiting": (0, 1, 2, 5),
        "declared_walltime": (0, 1, 5, 20)}


def score(rule: dict, worlds: list[Path], tag: str) -> tuple[float, dict]:
    """Mean wait over the training windows; per-window so the feedback can name the bad ones."""
    tmp = R / f"rules/_cand_{tag}.json"
    tmp.parent.mkdir(exist_ok=True)
    tmp.write_text(json.dumps(rule))
    per = {}
    for w in worlds:
        out = w / f"out/{tag}_job_statistics.csv"
        if out.exists():
            out.unlink()
        subprocess.run([PY, "-m", "pins.elastisim_bench", "run", "--world", str(w),
                        "--arm", "rule_synth", "--rule", str(tmp), "--tag", tag],
                       cwd=R, capture_output=True, timeout=3600)
        rows = [json.loads(l) for l in (w / "results.jsonl").read_text().splitlines() if l.strip()]
        cand = [r for r in rows if r.get("arm") == "rule_synth"]
        per[w.name] = cand[-1]["mean_wait_s"] if cand else float("inf")
    return st.mean(per.values()), per


def best_fixed(worlds: list[Path]) -> dict:
    """Per-window best deterministic pair, from runs already on disk. This is what to beat."""
    out = {}
    for w in worlds:
        rows = [json.loads(l) for l in (w / "results.jsonl").read_text().splitlines() if l.strip()]
        cand = [(r["mean_wait_s"], f"{r['arm']}+{r.get('sizer')}") for r in rows
                if r.get("arm") in POLICY_MENU["ordering"] and r.get("sizer") in POLICY_MENU["sizing"]]
        out[w.name] = min(cand) if cand else (float("inf"), "?")
    return out


def random_rule(rng: random.Random) -> dict:
    pick = lambda: {"ordering": rng.choice(list(POLICY_MENU["ordering"])),
                    "sizing": rng.choice(list(POLICY_MENU["sizing"]))}
    n = rng.randint(1, RULE_MAX_BRANCHES)
    branches = []
    for _ in range(n):
        f = rng.choice(list(RULE_FIELDS))
        branches.append({"if": [f, rng.choice(RULE_OPS), rng.choice(GRID[f])], "then": pick()})
    return {"branches": branches, "default": pick()}


def llm_rule(ctx, history, fixed) -> dict | None:
    from pins.correction import _ask
    fb = ""
    if history:
        fb = "\n\nYOUR PREVIOUS ATTEMPTS (mean waiting seconds, lower is better):\n"
        for h in history[-4:]:
            fb += f"  score {h['score']:.0f}: {json.dumps(h['rule'])}\n"
        fb += ("\nThe best FIXED pair per window scores:\n  " +
               "\n  ".join(f"{k}: {v[0]:.0f} with {v[1]}" for k, v in list(fixed.items())[:6]) +
               "\nYou must beat those. Change the rule; repeating a scored attempt is wasted.")
    return _ask(RULE_SYNTH, _packet_synth(ctx) + fb, ctx["model"], ctx["host"], ctx["cache"],
                "es-rule-train", num_predict=500)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", required=True, help="glob of TRAINING world dirs")
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--proposer", choices=["llm", "random"], default="llm")
    ap.add_argument("--model", default="qwen2.5:14b")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    worlds = [Path(w) for w in sorted(glob.glob(a.worlds)) if (Path(w) / "meta.json").exists()]
    if not worlds:
        sys.exit(f"no training worlds match {a.worlds}")
    fixed = best_fixed(worlds)
    print(f"{len(worlds)} training windows; best fixed pair mean = "
          f"{st.mean(v[0] for v in fixed.values()):.0f}", flush=True)

    from pins.llm_agent import HOST
    ctx = {"model": a.model, "host": HOST, "cache": {}, "pool_n": 80}
    rng = random.Random(a.seed)
    history, best = [], None
    for i in range(a.iters):
        rule = llm_rule(ctx, history, fixed) if a.proposer == "llm" else random_rule(rng)
        ok, why = _validate_rule(rule)
        if not ok:
            print(f"  iter {i}: REFUSED ({why})", flush=True)
            history.append({"rule": rule, "score": float("inf"), "invalid": why})
            continue
        sc, per = score(rule, worlds, f"tr{a.proposer}{i}")
        history.append({"rule": rule, "score": sc})
        if best is None or sc < best["score"]:
            best = {"rule": rule, "score": sc, "iter": i}
        print(f"  iter {i}: {sc:9.0f}  best {best['score']:9.0f}", flush=True)
    a.out.parent.mkdir(exist_ok=True)
    a.out.write_text(json.dumps(best["rule"], indent=1))
    (a.out.parent / (a.out.stem + "_history.json")).write_text(json.dumps(history, indent=1))
    print(f"BEST {best['score']:.0f} at iter {best['iter']} -> {a.out}")


if __name__ == "__main__":
    main()
