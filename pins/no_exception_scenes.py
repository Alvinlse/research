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


def sample_scenes(n: int, seed: int, max_delta: int = 6):
    """Return (scenes, meta) — n non-vacuous no-exception scenes, deterministic per seed."""
    rng = random.Random(seed)
    trace = load_trace()
    quanta_pool = [q for _a, _dur, q, _name in trace]
    scenes: list[HardCase] = []
    meta: list[dict] = []
    guard = 0
    while len(scenes) < n and guard < n * 50:
        guard += 1
        j = rng.choice(_JS)
        quanta = [rng.choice(quanta_pool) for _ in range(j)]
        case, m = _scene(len(scenes), quanta, rng.uniform(*_SLACK))
        floors, alloc, ranking, _ = build_anchor(case)
        menu = candidate_actions(alloc, case.free_gpus, ranking, floors, max_delta)
        if len(menu) <= 1:                        # non-vacuity gate: only retain_market exists
            continue
        m["menu_size"] = len(menu)
        scenes.append(case)
        meta.append(m)
    if len(scenes) < n:
        raise RuntimeError(f"only found {len(scenes)}/{n} non-vacuous scenes")
    return scenes, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-delta", type=int, default=6)
    a = ap.parse_args()
    scenes, meta = sample_scenes(a.n, a.seed, a.max_delta)
    js = [m["J"] for m in meta]
    print(f"kept {len(scenes)} scenes  seed={a.seed}  max_delta={a.max_delta}")
    print(f"  jobs/scene: min {min(js)} max {max(js)} mean {sum(js)/len(js):.2f}")
    print(f"  free_gpus:  mean {sum(m['free'] for m in meta)/len(meta):.1f}")
    print(f"  menu_size:  mean {sum(m['menu_size'] for m in meta)/len(meta):.1f}")


if __name__ == "__main__":
    main()
