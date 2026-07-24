# Exp 90 — No-exception specificity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure how often each LLM arm makes a false suggestion (fires a change) on real v2020 job-scenes that contain no exception, where the correct action is to retain the market allocation.

**Architecture:** A seeded sampler turns real trace jobs into `HardCase`-shaped no-exception scenes (no text, idle headroom as the only meddling surface). The existing `exp88_budget_control.run_case` scores each arm's `fired` flag unchanged; a thin driver runs the ladder (market / single-no-pkt / single-pkt / debate-pkt), dumps a Exp-88-shaped JSON, and runs a paired McNemar on the `fired` indicator.

**Tech Stack:** Python 3.10, `uv` venv at `.venv`, Ollama `qwen2.5:14b` (LLM arms only; all unit tests are offline). **No pytest** — this repo runs tests as `pins/test_*.py` modules with a `__main__` runner, invoked `.venv/bin/python -m pins.test_<name>` (see `pins/test_mechanism.py`).

## Global Constraints

- Run everything from `Research/` with `.venv/bin/python`. This is its own git repo — commit here.
- All arms at **temperature 0**; `max_delta = 6` held fixed across arms and the non-vacuity gate.
- Reuse, do not fork: `trace_replay.load_trace`, `h2_eval.build_anchor`, `packet.candidate_actions`, `packet.build_packet`, `exp88_budget_control.run_case`, `exp88_analyse.mcnemar_exact_two_sided`, `exp89_analyse.mcnemar_one_sided`, `correction.gather_corrections`/`referee_delta`.
- `no_text=True` is passed through `run_case` for every LLM arm — no scene ever carries authored text.
- Correct action = `retain_market` (packet action id 0). False suggestion = `meta["fired"]` = `bool(changes) or bool(hold_free)`.
- Login-node CPU reaper: the LLM run is one background sweep, no mid-run edits.

---

## File Structure

- `pins/no_exception_scenes.py` (NEW) — sampler: real jobs → non-vacuous no-exception `HardCase`s + a stats CLI.
- `pins/exp90_specificity.py` (NEW) — driver + analysis: run the ladder, dump JSON, print false rates + McNemar.
- `pins/test_exp90_scenes.py` (NEW) — offline tests for the sampler (`__main__` runner).
- `pins/test_exp90_analysis.py` (NEW) — offline tests for the `fired` discordant + arm scoring on a stub result (`__main__` runner).

Reused unchanged: `pins/h2_eval.py`, `pins/packet.py`, `pins/exp88_budget_control.py`, `pins/exp88_analyse.py`, `pins/exp89_analyse.py`, `pins/correction.py`, `pins/trace_replay.py`.

---

## Task 1: Scene sampler

**Files:**
- Create: `pins/no_exception_scenes.py`
- Test: `pins/test_exp90_scenes.py`

**Interfaces:**
- Consumes: `trace_replay.load_trace() -> list[tuple[arrival:int, dur:int, quanta:int, name:str]]`; `hardcases.HardCase`, `hardcases.d`, `hardcases.s`; `h2_eval.build_anchor(case) -> (floors, alloc, ranking, env)`; `packet.candidate_actions(alloc, free, ranking, floors, max_delta) -> list[dict]`.
- Produces: `sample_scenes(n: int, seed: int, max_delta: int = 6) -> tuple[list[HardCase], list[dict]]` returning the kept scenes and a parallel list of `{"id","J","free","base_sum","menu_size"}` metadata dicts. Ids are `NE-0000…`. Also `CAP_CLIP = 8`.

- [ ] **Step 1: Write the failing tests**

```python
# pins/test_exp90_scenes.py
"""Offline tests for the Exp 90 scene sampler — run: python -m pins.test_exp90_scenes"""
from pins.no_exception_scenes import sample_scenes, CAP_CLIP
from pins.h2_eval import build_anchor
from pins.hardcases import total


def test_deterministic_per_seed():
    a, _ = sample_scenes(20, seed=0)
    b, _ = sample_scenes(20, seed=0)
    assert [c.id for c in a] == [c.id for c in b]
    assert [c.free_gpus for c in a] == [c.free_gpus for c in b]


def test_no_text_and_inelastic():
    scenes, _ = sample_scenes(30, seed=1)
    for c in scenes:
        dem = [x for x in c.stmts if x["side"] == "demand"]
        assert 3 <= len(dem) <= 5
        assert all(x["requested_margin_gpus"] == 0 for x in dem)
        assert all(1 <= x["base_gpus"] <= CAP_CLIP for x in dem)


def test_correct_action_is_retain_market():
    # every base seats (free >= sum bases) and idle headroom is left over
    scenes, meta = sample_scenes(30, seed=2)
    for c, m in zip(scenes, meta):
        floors, alloc, ranking, _ = build_anchor(c)
        base_sum = sum(x["base_gpus"] for x in c.stmts if x["side"] == "demand")
        assert total(alloc) == base_sum          # all bases seated, no margin sold
        assert c.free_gpus > base_sum             # genuine idle headroom to meddle with


def test_non_vacuous_menu():
    from pins.packet import candidate_actions
    scenes, meta = sample_scenes(30, seed=3)
    for c, m in zip(scenes, meta):
        floors, alloc, ranking, _ = build_anchor(c)
        menu = candidate_actions(alloc, c.free_gpus, ranking, floors, 6)
        assert len(menu) > 1                      # more than just retain_market (id 0)
        assert m["menu_size"] == len(menu)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"running {len(tests)} scene tests\n")
    for t in tests:
        print(f"- {t.__name__}")
        t()
    print(f"\nall {len(tests)} tests passed.")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pins.test_exp90_scenes`
Expected: FAIL with `ModuleNotFoundError: No module named 'pins.no_exception_scenes'`

- [ ] **Step 3: Write the sampler**

```python
# pins/no_exception_scenes.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pins.test_exp90_scenes`
Expected: `all 4 tests passed.`

- [ ] **Step 5: Smoke the CLI**

Run: `.venv/bin/python -m pins.no_exception_scenes --n 200 --seed 0`
Expected: prints `kept 200 scenes …` with mean menu_size > 1.

- [ ] **Step 6: Commit**

```bash
git add pins/no_exception_scenes.py pins/test_exp90_scenes.py
git commit -m "Exp 90: no-exception scene sampler + tests"
```

---

## Task 2: Driver + analysis

**Files:**
- Create: `pins/exp90_specificity.py`
- Test: `pins/test_exp90_analysis.py`

**Interfaces:**
- Consumes: `no_exception_scenes.sample_scenes`; `exp88_budget_control.run_case(case, model, use_llm, arm, max_delta, temperature, no_text=False) -> (final, why, meta)` where `meta` has `fired: bool` and `rejected: bool`; `correction.gather_corrections(jobs, supply_note, free, alloc, use_llm, model, cache)` and `correction.referee_delta(alloc, props, free, use_llm, model, cache)`; `exp88_budget_control._jobs_of(case, no_text) -> (jobs, sup)`; `h2_eval.build_anchor`; `exp88_analyse.mcnemar_exact_two_sided(b, c)`; `exp89_analyse.mcnemar_one_sided(b, c)`.
- Produces: `single_no_packet_fired(case, model, use_llm) -> bool`; `fired_discordant(res, x, y) -> (bx, cy)`; writes `pins/results_exp90_qwen2514b.json` with shape `{"model","n","seed","arms":[...],"results":{id:{"category":str,"arms":{arm:{"fired":bool,"rejected":bool,"changes":dict}}}}}`.

- [ ] **Step 1: Write the failing tests**

```python
# pins/test_exp90_analysis.py
"""Offline tests for the Exp 90 driver — run: python -m pins.test_exp90_analysis"""
from pins.exp90_specificity import single_no_packet_fired, fired_discordant
from pins.no_exception_scenes import sample_scenes


def test_single_no_packet_is_silent_on_no_text():
    # correction.py is text-gated: empty notes -> no LLM call -> never fires
    scenes, _ = sample_scenes(5, seed=0)
    for c in scenes:
        assert single_no_packet_fired(c, "qwen2.5:14b", use_llm=True) is False


def test_fired_discordant_counts():
    res = {
        "s1": {"arms": {"A": {"fired": True},  "B": {"fired": False}}},  # A only
        "s2": {"arms": {"A": {"fired": False}, "B": {"fired": True}}},   # B only
        "s3": {"arms": {"A": {"fired": True},  "B": {"fired": True}}},   # both
        "s4": {"arms": {"A": {"fired": False}, "B": {"fired": False}}},  # neither
    }
    assert fired_discordant(res, "A", "B") == (1, 1)   # (A-only, B-only)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"running {len(tests)} analysis tests\n")
    for t in tests:
        print(f"- {t.__name__}")
        t()
    print(f"\nall {len(tests)} tests passed.")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pins.test_exp90_analysis`
Expected: FAIL with `ModuleNotFoundError: No module named 'pins.exp90_specificity'`

- [ ] **Step 3: Write the driver**

```python
# pins/exp90_specificity.py
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


def run(model: str, n: int, seed: int, max_delta: int) -> dict:
    scenes, _meta = sample_scenes(n, seed, max_delta)
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
    return {"model": model, "n": len(scenes), "seed": seed, "arms": ARMS, "results": results}


def analyse(blob: dict) -> None:
    res = blob["results"]
    ids = list(res)
    print(f"\n=== per-arm false-suggestion rate (n={len(ids)}) ===")
    for arm in blob["arms"]:
        fired = sum(1 for c in ids if res[c]["arms"][arm]["fired"])
        harm = sum(1 for c in ids if res[c]["arms"][arm]["fired"]
                   and res[c]["arms"][arm]["rejected"])
        print(f"  {arm:14s} fired {fired:>3d}/{len(ids)}   of which harmful {harm}")
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
    ap.add_argument("--out", default="pins/results_exp90_qwen2514b.json")
    a = ap.parse_args()
    blob = run(a.model, a.n, a.seed, a.max_delta)
    with open(a.out, "w") as fh:
        json.dump(blob, fh)
    analyse(blob)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pins.test_exp90_analysis`
Expected: `all 2 tests passed.` `test_single_no_packet_is_silent_on_no_text` makes no Ollama call because `gather_corrections` short-circuits on empty notes.

- [ ] **Step 5: Offline smoke of market/single-no-pkt path (no LLM arms)**

Run:
```bash
.venv/bin/python -c "
from pins.exp90_specificity import single_no_packet_fired
from pins.no_exception_scenes import sample_scenes
sc,_=sample_scenes(3,0)
print([single_no_packet_fired(c,'qwen2.5:14b',True) for c in sc])"
```
Expected: `[False, False, False]`

- [ ] **Step 6: Commit**

```bash
git add pins/exp90_specificity.py pins/test_exp90_analysis.py
git commit -m "Exp 90: specificity driver + fired McNemar analysis"
```

---

## Task 3: Run the experiment and record the result

**Files:**
- Create: `pins/exp90_specificity_14b.log` (run artifact), `pins/results_exp90_qwen2514b.json` (run artifact)
- Modify: `research_progress.md` (append the Exp 90 write-up)

**Interfaces:**
- Consumes: the driver from Task 2. Requires Ollama up at `http://localhost:11434` with `qwen2.5:14b` pulled.

- [ ] **Step 1: Confirm Ollama is serving the model**

Run: `curl -s http://localhost:11434/api/tags | grep -o 'qwen2.5:14b' | head -1`
Expected: prints `qwen2.5:14b`. If empty, the run cannot proceed — stop and report.

- [ ] **Step 2: Launch the run in the background (one sweep)**

Run:
```bash
PINS_NUM_CTX=8192 nohup .venv/bin/python -u -m pins.exp90_specificity \
  --model qwen2.5:14b --n 200 --seed 0 --max-delta 6 \
  > pins/exp90_specificity_14b.log 2>&1 &
```
Expected: a PID prints; `tail pins/exp90_specificity_14b.log` shows per-scene `NE-#### market=. single-no-pkt=. single-pkt=? debate-pkt=?` lines accruing. Do not edit files mid-run (reaper).

- [ ] **Step 3: Wait for completion, then read the analysis block**

Run: `tail -30 pins/exp90_specificity_14b.log`
Expected: the `=== per-arm false-suggestion rate ===` and `=== PRIMARY: McNemar ===` blocks, then `wrote pins/results_exp90_qwen2514b.json`.

- [ ] **Step 4: Sanity-check the invariants hold on the real run**

Run:
```bash
.venv/bin/python -c "
import json; r=json.load(open('pins/results_exp90_qwen2514b.json'))['results']
mk=sum(v['arms']['market']['fired'] for v in r.values())
sn=sum(v['arms']['single-no-pkt']['fired'] for v in r.values())
print('market fired', mk, ' single-no-pkt fired', sn, ' n', len(r))"
```
Expected: `market fired 0  single-no-pkt fired 0  n 200`. A non-zero `single-no-pkt` means text-gating leaked — report it, do not silently accept.

- [ ] **Step 5: Append the write-up to `research_progress.md`**

Add a `## Experiment 90 — …` section reporting: the three per-arm false rates, the primary McNemar (b, c, one-sided p), the harmful slice, which of the §7 decision-rule branches fired, and the reproduce command. Paste the raw analysis block from the log verbatim (user preference: show raw output).

- [ ] **Step 6: Commit**

```bash
git add pins/exp90_specificity_14b.log pins/results_exp90_qwen2514b.json research_progress.md
git commit -m "Exp 90: no-exception specificity run — <one-line verdict>"
```

---

## Self-Review

**Spec coverage:**
- §3 hypotheses → Task 2 `analyse` (one-sided McNemar debate<single) + Task 3 Step 4 (single-no-pkt≈0 verification). ✓
- §4 scene construction (J∈{3,4,5}, quanta clip 8, margin 0, slack 0.15–0.6, non-vacuity gate) → Task 1 `sample_scenes` + tests. ✓
- §5 arms ladder → Task 2 `ARMS` + `run`. ✓
- §6 metric `fired`, McNemar, harmful slice → Task 2 `analyse`. ✓
- §7 decision rule → Task 2 `analyse` verdict + Task 3 Step 5. ✓
- §8 N=200 → Task 3 Step 2. ✓
- §10 reuse list → honored (no forks). ✓
- §11 reproduce command → matches Task 3 Step 2. ✓

**Placeholder scan:** none — every step has runnable code/commands and expected output. The only `<…>` is the human-written verdict in the Task 3 commit message, which is intentional.

**Type consistency:** `sample_scenes -> (scenes, meta)` consumed with that arity in both tests and driver; `run_case(...)->(final,why,meta)` with `meta["fired"]`/`meta["rejected"]` matches `exp88_budget_control.py:178-181`; `fired_discordant -> (bx, cy)` fed to `mcnemar_one_sided(c, b)` in the intended direction (H1 = debate fires less). JSON `results[id]["arms"][arm]["fired"]` written in `run`, read in `analyse`, `fired_discordant`, and the Task 3 sanity check identically. ✓
