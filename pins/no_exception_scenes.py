"""Exp 90 sampler — real v2020 jobs -> no-exception scenes (no text, idle headroom only).

Correct action on every scene is retain_market: no job asked for more than its base, no note
carries a reason, so any grant/transfer/hold is unmotivated. Idle headroom (slack above the
seated bases) is the only surface an arm can meddle with. Scenes whose action menu is just
"change nothing" are discarded before scoring (non-vacuity gate).
"""
from __future__ import annotations

import argparse
import random

from pins.h2_eval import build_anchor
from pins.hardcases import HardCase, d, s
from pins.packet import candidate_actions
from pins.trace_replay import load_trace

CAP_CLIP = 8
_JS = (3, 4, 5)                # jobs per scene
_SLACK = (0.15, 0.6)           # idle headroom as a fraction of the seated bases


def _scene(idx: int, quanta: list[int], slack: float) -> tuple[HardCase, dict]:
    bases = [min(CAP_CLIP, max(1, q)) for q in quanta]
    base_sum = sum(bases)
    free = base_sum + max(1, round(base_sum * slack))   # guarantee free > base_sum: real idle headroom
    stmts = [d(f"r{j:02d}", "besteffort", "ontrack", b, 0, "") for j, b in enumerate(bases)]
    stmts.append(s(0, "none", ""))                # neutral supply; no reserve, no text
    case = HardCase(id=f"NE-{idx:04d}", category="no_exception", free_gpus=free, stmts=stmts,
                    predicate=lambda a, r: True, rationale="no-exception scene",
                    expect="rigid arm retains the market", must_cite=[])
    return case, {"id": case.id, "J": len(bases), "free": free, "base_sum": base_sum}


def sample_scenes(n: int, seed: int, max_delta: int = 6,
                  spread_min: int = 0, slack_lo: float = _SLACK[0], slack_hi: float = _SLACK[1]):
    """Return (scenes, meta) — n non-vacuous no-exception scenes, deterministic per seed.

    spread_min>0 adds the Exp 91 imbalance gate (base spread >= spread_min); slack_lo/slack_hi
    override the idle-headroom range. Defaults reproduce Exp 90 byte-identically.
    """
    rng = random.Random(seed)
    trace = load_trace()
    quanta_pool = [q for _a, _dur, q, _name in trace]
    scenes: list[HardCase] = []
    meta: list[dict] = []
    guard = 0
    while len(scenes) < n and guard < n * 200:
        guard += 1
        j = rng.choice(_JS)
        quanta = [rng.choice(quanta_pool) for _ in range(j)]
        bases = [min(CAP_CLIP, max(1, q)) for q in quanta]
        spread = max(bases) - min(bases)
        if spread < spread_min:                   # Exp 91 imbalance gate (no-op when spread_min=0)
            continue
        case, m = _scene(len(scenes), quanta, rng.uniform(slack_lo, slack_hi))
        floors, alloc, ranking, _ = build_anchor(case)
        menu = candidate_actions(alloc, case.free_gpus, ranking, floors, max_delta)
        if len(menu) <= 1:                        # non-vacuity gate: only retain_market exists
            continue
        m["menu_size"] = len(menu)
        m["spread"] = spread
        scenes.append(case)
        meta.append(m)
    if len(scenes) < n:
        raise RuntimeError(f"only found {len(scenes)}/{n} scenes "
                           f"(spread_min={spread_min}, slack=[{slack_lo},{slack_hi}])")
    return scenes, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-delta", type=int, default=6)
    ap.add_argument("--spread-min", type=int, default=0)
    ap.add_argument("--slack-lo", type=float, default=_SLACK[0])
    ap.add_argument("--slack-hi", type=float, default=_SLACK[1])
    a = ap.parse_args()
    scenes, meta = sample_scenes(a.n, a.seed, a.max_delta, a.spread_min, a.slack_lo, a.slack_hi)
    js = [m["J"] for m in meta]
    print(f"kept {len(scenes)} scenes  seed={a.seed}  max_delta={a.max_delta}  "
          f"spread_min={a.spread_min}  slack=[{a.slack_lo},{a.slack_hi}]")
    print(f"  jobs/scene: min {min(js)} max {max(js)} mean {sum(js)/len(js):.2f}")
    print(f"  free_gpus:  mean {sum(m['free'] for m in meta)/len(meta):.1f}")
    print(f"  headroom:   mean {sum(m['free']-m['base_sum'] for m in meta)/len(meta):.2f}")
    print(f"  spread:     mean {sum(m['spread'] for m in meta)/len(meta):.2f}")
    print(f"  menu_size:  mean {sum(m['menu_size'] for m in meta)/len(meta):.1f}")


if __name__ == "__main__":
    main()
