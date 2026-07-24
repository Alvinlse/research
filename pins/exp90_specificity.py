"""Exp 90 — no-exception specificity. Pre-reg: docs/superpowers/specs/2026-07-24-exp90-...md

False suggestion = an arm fires any change on a scene whose correct action is retain_market.
Ladder: market (0 by construction) / single-no-pkt (text-gated, ~0) / single-pkt / debate-pkt.
Primary: one-sided McNemar on `fired`, debate-pkt vs single-pkt (does the rebuttal fire less?).

  .venv/bin/python -m pins.exp90_specificity --model qwen2.5:14b --n 200 --seed 0 --max-delta 6
"""
from __future__ import annotations

import argparse
import json

from pins.correction import gather_corrections, referee_delta
from pins.exp88_analyse import mcnemar_exact_two_sided
from pins.exp88_budget_control import _jobs_of, run_case
from pins.exp89_analyse import mcnemar_one_sided
from pins.h2_eval import build_anchor
from pins.no_exception_scenes import sample_scenes

ARMS = ["market", "single-no-pkt", "single-pkt", "debate-pkt"]


def single_no_packet_fired(case, model: str, use_llm: bool) -> bool:
    """The pre-packet free-text single arm. Text-gated in correction.py, so on a no-text scene
    it makes no LLM call and returns no delta — measured, not assumed."""
    _floors, alloc, _ranking, _env = build_anchor(case)
    jobs, sup = _jobs_of(case, no_text=True)
    cache: dict = {}
    props = gather_corrections(jobs, sup, case.free_gpus, alloc, use_llm, model, cache)
    delta = referee_delta(alloc, props, case.free_gpus, use_llm, model, cache)
    return bool(delta.get("delta"))


def fired_discordant(res: dict, x: str, y: str) -> tuple[int, int]:
    """(#scenes where x fired and y did not, #scenes where y fired and x did not)."""
    def f(c, arm):
        return bool(res[c]["arms"][arm]["fired"])
    bx = sum(1 for c in res if f(c, x) and not f(c, y))
    cy = sum(1 for c in res if f(c, y) and not f(c, x))
    return bx, cy


def run(model: str, n: int, seed: int, max_delta: int,
        spread_min: int = 0, slack_lo: float = 0.15, slack_hi: float = 0.60) -> dict:
    scenes, meta = sample_scenes(n, seed, max_delta, spread_min, slack_lo, slack_hi)
    min_menu = min(m["menu_size"] for m in meta)     # non-triviality: every scene has a bait action
    results: dict = {}
    for i, case in enumerate(scenes):
        arms: dict = {}
        arms["market"] = {"fired": False, "rejected": False, "changes": {}}
        arms["single-no-pkt"] = {
            "fired": single_no_packet_fired(case, model, use_llm=True),
            "rejected": False, "changes": {}}
        for arm in ("single-pkt", "debate-pkt"):
            _final, _why, meta = run_case(case, model, True, arm, max_delta, 0, no_text=True)
            arms[arm] = {"fired": bool(meta["fired"]), "rejected": bool(meta["rejected"]),
                         "changes": meta["changes"]}
        results[case.id] = {"category": case.category, "arms": arms}
        marks = " ".join(f"{a}={'F' if arms[a]['fired'] else '.'}" for a in ARMS)
        print(f"{i + 1:>3}/{len(scenes)}  {case.id}  {marks}")
    return {"model": model, "n": len(scenes), "seed": seed, "arms": ARMS,
            "spread_min": spread_min, "slack": [slack_lo, slack_hi], "min_menu": min_menu,
            "results": results}


def analyse(blob: dict, easy_path: str | None = None) -> None:
    res = blob["results"]
    ids = list(res)
    easy = None
    if easy_path:
        try:
            with open(easy_path) as fh:
                easy = json.load(fh)["results"]
        except FileNotFoundError:
            print(f"(easy baseline {easy_path} not found — skipping hard-vs-easy)")
    if "min_menu" in blob:
        print(f"\nnon-triviality: every scene has menu>1 (min menu_size={blob['min_menu']}) "
              f"— a non-retain bait action always exists")
    if "spread_min" in blob:
        print(f"bait: spread_min={blob['spread_min']}  slack={blob.get('slack')}")
    print(f"\n=== per-arm false-suggestion rate (n={len(ids)}) ===")
    for arm in blob["arms"]:
        fired = sum(1 for c in ids if res[c]["arms"][arm]["fired"])
        harm = sum(1 for c in ids if res[c]["arms"][arm]["fired"]
                   and res[c]["arms"][arm]["rejected"])
        tag = ""
        if easy and arm in easy[next(iter(easy))]["arms"]:
            ez = sum(1 for c in easy if easy[c]["arms"][arm]["fired"])
            tag = f"   (easy Exp90: {ez}/{len(easy)})"
        print(f"  {arm:14s} fired {fired:>3d}/{len(ids)}   of which harmful {harm}{tag}")
    b, c = fired_discordant(res, "debate-pkt", "single-pkt")
    p_less = mcnemar_one_sided(c, b)          # H1: debate fires LESS than single-pkt
    print("\n=== PRIMARY: McNemar on `fired`, debate-pkt vs single-pkt ===")
    print(f"  debate-only fired={b}  single-only fired={c}")
    print(f"  H1 debate fires LESS: one-sided p={p_less:.4f}  "
          f"(two-sided {mcnemar_exact_two_sided(b, c):.4f})")
    print("  -> " + ("debate fires significantly less — rebuttal restores floor-silence"
                     if p_less < 0.05 else
                     "not significant at this n; direction " +
                     ("favors debate (fewer)" if c > b else "does not favor debate")))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5:14b")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-delta", type=int, default=6)
    ap.add_argument("--spread-min", type=int, default=0)
    ap.add_argument("--slack-lo", type=float, default=0.15)
    ap.add_argument("--slack-hi", type=float, default=0.60)
    ap.add_argument("--easy", default=None,
                    help="Exp 90 easy-scene results json for the hard-vs-easy print")
    ap.add_argument("--out", default="pins/results_exp90_qwen2514b.json")
    a = ap.parse_args()
    blob = run(a.model, a.n, a.seed, a.max_delta, a.spread_min, a.slack_lo, a.slack_hi)
    with open(a.out, "w") as fh:
        json.dump(blob, fh)
    analyse(blob, a.easy)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
