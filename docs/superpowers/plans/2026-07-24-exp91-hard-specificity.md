# Exp 91 — Hard-Specificity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade Exp 90's no-exception specificity test with efficiency bait (lopsided bases + tight slack) so that restraint is stress-tested on adversarial-but-still-retain scenes, then run it at `qwen2.5:14b`.

**Architecture:** Add two default-off params to the existing `sample_scenes` sampler — an imbalance gate (`spread_min`) and a tight-slack range (`slack_lo/slack_hi`) — and thread four matching CLI flags plus a hard-vs-easy report block through the reused `exp90_specificity.py` driver. Every arm, metric, and the McNemar primary are reused unchanged. With the new params unset, Exp 90 reproduces byte-identically.

**Tech Stack:** Python 3.10, `uv` project, offline pytest-free test modules (`python -m pins.test_*`), Ollama `qwen2.5:14b` at temp 0.

## Global Constraints

- Run everything from `Research/` with `.venv/bin/python`.
- The retain-market invariant is inviolable: jobs stay inelastic (`requested_margin_gpus = 0`), `no_text=True`, neutral supply. Do not add tiers, deadlines, reserves, or text.
- **Default-off compatibility:** with `spread_min=0` and `slack_lo=0.15, slack_hi=0.6`, `sample_scenes` MUST produce the identical scene ids/free_gpus as Exp 90.
- `CAP_CLIP = 8`, `max_delta = 6`, `J ∈ {3,4,5}` unchanged.
- One background LLM sweep only (login-node CPU reaper): no `pgrep` self-match gates, no edits mid-sweep.
- Test modules are run with `.venv/bin/python -m pins.<module>` (no pytest in this repo).

---

### Task 1: Sampler bait — imbalance gate + tight-slack range

**Files:**
- Modify: `pins/no_exception_scenes.py` (`sample_scenes`, `main`)
- Create: `pins/test_exp91_scenes.py`

**Interfaces:**
- Consumes: `_scene(idx, quanta, slack) -> (HardCase, dict)` (unchanged), `build_anchor`, `candidate_actions` (unchanged).
- Produces: `sample_scenes(n, seed, max_delta=6, spread_min=0, slack_lo=0.15, slack_hi=0.6) -> (scenes, meta)`. New keyword-only-safe params appended; positional calls `sample_scenes(n, seed)` and `sample_scenes(n, seed, max_delta)` unchanged. `meta[i]` gains `"spread"` (int = `max(bases)-min(bases)`).

- [ ] **Step 1: Write the failing tests**

Create `pins/test_exp91_scenes.py`:

```python
"""Offline tests for the Exp 91 hard-specificity sampler bait.
Run: .venv/bin/python -m pins.test_exp91_scenes"""
from pins.no_exception_scenes import sample_scenes, CAP_CLIP


def test_defaults_reproduce_exp90():
    # default params == Exp 90; explicit Exp 90 values must give identical scenes
    a, _ = sample_scenes(25, seed=0)
    b, _ = sample_scenes(25, seed=0, spread_min=0, slack_lo=0.15, slack_hi=0.6)
    assert [c.id for c in a] == [c.id for c in b]
    assert [c.free_gpus for c in a] == [c.free_gpus for c in b]


def test_imbalance_gate_enforced():
    scenes, meta = sample_scenes(30, seed=1, spread_min=4)
    for c, m in zip(scenes, meta):
        bases = [x["base_gpus"] for x in c.stmts if x["side"] == "demand"]
        assert max(bases) - min(bases) >= 4          # lopsided by construction
        assert m["spread"] == max(bases) - min(bases)
        assert m["spread"] <= CAP_CLIP - 1


def test_tight_slack_reduces_headroom():
    # tight slack must leave strictly less headroom than Exp 90's generous default, on average
    tight, mt = sample_scenes(40, seed=2, spread_min=4, slack_lo=0.05, slack_hi=0.30)
    loose, ml = sample_scenes(40, seed=2, spread_min=4, slack_lo=0.15, slack_hi=0.60)
    hr_t = sum(m["free"] - m["base_sum"] for m in mt) / len(mt)
    hr_l = sum(m["free"] - m["base_sum"] for m in ml) / len(ml)
    assert hr_t < hr_l
    for m in mt:                                     # invariant: headroom never zero
        assert m["free"] > m["base_sum"]


def test_invariant_still_holds_under_bait():
    scenes, _ = sample_scenes(30, seed=3, spread_min=4, slack_lo=0.05, slack_hi=0.30)
    for c in scenes:
        dem = [x for x in c.stmts if x["side"] == "demand"]
        assert all(x["requested_margin_gpus"] == 0 for x in dem)   # inelastic
        assert all(x["justification"] == "" for x in dem)          # no text
        assert all(x["side"] != "supply" or x["reserve_gpus"] == 0 for x in c.stmts)


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    print(f"running {len(tests)} exp91 scene tests\n")
    for t in tests:
        print(f"- {t.__name__}")
        t()
    print(f"\nall {len(tests)} tests passed.")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pins.test_exp91_scenes`
Expected: FAIL — `TypeError: sample_scenes() got an unexpected keyword argument 'spread_min'` (or `KeyError: 'spread'`).

- [ ] **Step 3: Add the params + imbalance gate to `sample_scenes`**

In `pins/no_exception_scenes.py`, change the signature and the draw loop. Current:

```python
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
```

Replace with:

```python
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
```

Note: `_JS`, `_SLACK`, `CAP_CLIP`, `_scene` already exist at module top and are unchanged.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pins.test_exp91_scenes`
Expected: `all 4 tests passed.`

- [ ] **Step 5: Verify Exp 90's own tests still pass (regression)**

Run: `.venv/bin/python -m pins.test_exp90_scenes`
Expected: `all 5 tests passed.` (defaults unchanged ⇒ Exp 90 sampler untouched.)

- [ ] **Step 6: Wire the new params into the sampler CLI (`main`)**

In `pins/no_exception_scenes.py`, `main()` currently parses `--n --seed --max-delta`. Add the three flags and pass them, and print spread stats:

```python
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
```

- [ ] **Step 7: Smoke the CLI in hard mode (offline, no LLM)**

Run: `.venv/bin/python -m pins.no_exception_scenes --n 200 --seed 0 --spread-min 4 --slack-lo 0.05 --slack-hi 0.30`
Expected: `kept 200 scenes` with `spread: mean` ≥ 4 and `headroom: mean` well below the Exp 90 baseline. If it raises `only found <200 scenes`, the `n*200` guard was too tight for `spread_min=4` — report the observed count in the run log (do not silently lower `spread_min`).

- [ ] **Step 8: Commit**

```bash
git add pins/no_exception_scenes.py pins/test_exp91_scenes.py
git commit -m "Exp 91: sampler bait — imbalance gate + tight-slack range (default-off)"
```

---

### Task 2: Driver flags + hard-vs-easy / non-triviality reporting

**Files:**
- Modify: `pins/exp90_specificity.py` (`run`, `analyse`, `main`)

**Interfaces:**
- Consumes: `sample_scenes(n, seed, max_delta, spread_min, slack_lo, slack_hi)` from Task 1; existing `run_case`, `single_no_packet_fired`, `mcnemar_one_sided`, `mcnemar_exact_two_sided`.
- Produces: `run(model, n, seed, max_delta, spread_min=0, slack_lo=0.15, slack_hi=0.6) -> blob`; blob gains `"spread_min"`, `"slack"`, and `"min_menu"`. `analyse(blob, easy_path=None)`.

- [ ] **Step 1: Thread the bait params through `run`**

In `pins/exp90_specificity.py`, change `run` to accept and forward the params and stash provenance + the non-triviality figure. Current head of `run`:

```python
def run(model: str, n: int, seed: int, max_delta: int) -> dict:
    scenes, _meta = sample_scenes(n, seed, max_delta)
```

Replace the signature and first lines with:

```python
def run(model: str, n: int, seed: int, max_delta: int,
        spread_min: int = 0, slack_lo: float = 0.15, slack_hi: float = 0.60) -> dict:
    scenes, meta = sample_scenes(n, seed, max_delta, spread_min, slack_lo, slack_hi)
    min_menu = min(m["menu_size"] for m in meta)     # non-triviality: every scene has a bait action
```

and at the `return` of `run`, add the provenance fields. Current:

```python
    return {"model": model, "n": len(scenes), "seed": seed, "arms": ARMS, "results": results}
```

Replace with:

```python
    return {"model": model, "n": len(scenes), "seed": seed, "arms": ARMS,
            "spread_min": spread_min, "slack": [slack_lo, slack_hi], "min_menu": min_menu,
            "results": results}
```

- [ ] **Step 2: Add hard-vs-easy + non-triviality to `analyse`**

In `pins/exp90_specificity.py`, extend `analyse` to optionally load an easy-scene results file and print the side-by-side, plus the non-triviality line. Current signature line:

```python
def analyse(blob: dict) -> None:
    res = blob["results"]
    ids = list(res)
    print(f"\n=== per-arm false-suggestion rate (n={len(ids)}) ===")
```

Replace with:

```python
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
```

Then, inside the existing per-arm loop, print the easy baseline next to each arm. Current loop body:

```python
    for arm in blob["arms"]:
        fired = sum(1 for c in ids if res[c]["arms"][arm]["fired"])
        harm = sum(1 for c in ids if res[c]["arms"][arm]["fired"]
                   and res[c]["arms"][arm]["rejected"])
        print(f"  {arm:14s} fired {fired:>3d}/{len(ids)}   of which harmful {harm}")
```

Replace with:

```python
    for arm in blob["arms"]:
        fired = sum(1 for c in ids if res[c]["arms"][arm]["fired"])
        harm = sum(1 for c in ids if res[c]["arms"][arm]["fired"]
                   and res[c]["arms"][arm]["rejected"])
        tag = ""
        if easy and arm in easy[next(iter(easy))]["arms"]:
            ez = sum(1 for c in easy if easy[c]["arms"][arm]["fired"])
            tag = f"   (easy Exp90: {ez}/{len(easy)})"
        print(f"  {arm:14s} fired {fired:>3d}/{len(ids)}   of which harmful {harm}{tag}")
```

- [ ] **Step 3: Add the CLI flags in `main`**

Current `main` of `pins/exp90_specificity.py`:

```python
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
```

Replace with:

```python
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
```

- [ ] **Step 4: Verify the analyse path offline with a synthetic blob**

Run:

```bash
.venv/bin/python -c "
from pins.exp90_specificity import analyse
blob = {'arms': ['market','single-no-pkt','single-pkt','debate-pkt'],
        'spread_min': 4, 'slack': [0.05,0.3], 'min_menu': 3,
        'results': {f'NE-{i:04d}': {'category':'no_exception','arms':{
            a: {'fired': (a=='single-pkt' and i<5), 'rejected': False, 'changes':{}}
            for a in ['market','single-no-pkt','single-pkt','debate-pkt']}} for i in range(20)}}
analyse(blob)
"
```

Expected: prints the non-triviality line (`min menu_size=3`), the bait line, `single-pkt fired 5/20`, all others `0/20`, and a PRIMARY McNemar block with `single-only fired=5 debate-only fired=0`, direction favoring debate. No crash.

- [ ] **Step 5: Commit**

```bash
git add pins/exp90_specificity.py
git commit -m "Exp 91: driver flags + hard-vs-easy / non-triviality reporting"
```

---

### Task 3: Launch the sweep and record the result

**Files:**
- Create: `pins/exp91_hard_14b.log` (run output), `pins/results_exp91_hard_qwen2514b.json` (driver output)
- Modify: `research_progress.md` (append the Exp 91 entry)

**Interfaces:**
- Consumes: the CLI from Tasks 1–2; the stored Exp 90 baseline `pins/results_exp90_qwen2514b.json`.
- Produces: nothing importable — this is the experiment run + write-up.

- [ ] **Step 1: Confirm Ollama is up and the model is present**

Run: `curl -s http://localhost:11434/api/tags | grep -o 'qwen2.5:14b' | head -1`
Expected: prints `qwen2.5:14b`. If empty, Ollama is down or the model is not pulled — stop and tell the user; do not silently fall back to a smaller model.

- [ ] **Step 2: Back up the Exp 90 baseline (do not overwrite it)**

Run: `test -f pins/results_exp90_qwen2514b.json && cp -n pins/results_exp90_qwen2514b.json pins/results_exp90_qwen2514b.bak.json; echo done`
Expected: `done`. Exp 91 writes a *separate* `--out`, so this is belt-and-suspenders against a wrong flag.

- [ ] **Step 3: Launch the single background sweep**

Run (ONE background sweep only — login-node reaper; no re-launch, no edits while it runs):

```bash
PINS_NUM_CTX=8192 .venv/bin/python -u -m pins.exp90_specificity \
  --model qwen2.5:14b --n 200 --seed 0 --max-delta 6 \
  --spread-min 4 --slack-lo 0.05 --slack-hi 0.30 \
  --easy pins/results_exp90_qwen2514b.json \
  --out pins/results_exp91_hard_qwen2514b.json | tee pins/exp91_hard_14b.log
```

Expected on completion: per-arm fired counts with the `(easy Exp90: 0/200)` tags, the PRIMARY McNemar block, and `wrote pins/results_exp91_hard_qwen2514b.json`.

- [ ] **Step 4: Sanity-check the output against the invariant**

Run: `.venv/bin/python -c "import json; b=json.load(open('pins/results_exp91_hard_qwen2514b.json')); print('n', b['n'], 'spread_min', b['spread_min'], 'min_menu', b['min_menu'], 'market_fired', sum(r['arms']['market']['fired'] for r in b['results'].values()))"`
Expected: `n 200`, `spread_min 4`, `min_menu` ≥ 2, and `market_fired 0` (the floor arm must never fire — a non-zero value means the harness is broken, not a finding).

- [ ] **Step 5: Append the Exp 91 entry to `research_progress.md`**

Add a `## Experiment 91 — HARD-SPECIFICITY` section following the Exp 90 format: the pre-reg link, the bait description (`spread_min=4`, `slack=[0.05,0.30]`), the observed keep-rate and spread/headroom stats, the per-arm fired table with the hard-vs-easy contrast, the PRIMARY McNemar result, the harmful slice, and the decision-rule reading (§8 of the spec) that the numbers land on. Paste the actual `pins/exp91_hard_14b.log` per-arm/McNemar lines verbatim (raw output, not only the summary).

- [ ] **Step 6: Commit**

```bash
git add pins/results_exp91_hard_qwen2514b.json pins/exp91_hard_14b.log research_progress.md
git commit -m "Exp 91: hard-specificity run at qwen2.5:14b — <one-line verdict>"
```

---

## Self-Review

**Spec coverage:**
- §3 imbalance gate + tight-slack → Task 1 (params, gate, both tests). ✓
- §3 default-off compatibility → Task 1 Step 5 (Exp 90 tests) + `test_defaults_reproduce_exp90`. ✓
- §4 primary McNemar (debate vs single) → reused unchanged in `analyse`; exercised in Task 2 Step 4. ✓
- §4 hard-vs-easy robustness claim → Task 2 (`--easy` + per-arm tag), Task 3 Step 3. ✓
- §4 `single-no-pkt` verification → reused arm, printed in `analyse`. ✓
- §5 metric `fired` + harmful slice → reused unchanged. ✓
- §5 non-triviality line → Task 2 (`min_menu`). ✓
- §7 N=200, one background sweep → Task 3 Steps 3, plus reaper caveat in Global Constraints. ✓
- §10 reproduce command → Task 3 Step 3 matches the spec verbatim. ✓

**Placeholder scan:** the only bracketed placeholder is `<one-line verdict>` in the final commit message (filled at run time) and the Task 3 Step 5 write-up, which is inherently result-dependent and spells out exactly which fields to record. No TBD/TODO/"handle edge cases" in code steps. ✓

**Type consistency:** `sample_scenes(..., spread_min, slack_lo, slack_hi)` signature identical in Task 1 (definition), Task 2 (`run` call), Task 3 (CLI). `meta["spread"]`, `blob["min_menu"]`, `blob["spread_min"]`, `blob["slack"]` written in Task 1/Task 2 and read in Task 2 `analyse`. `analyse(blob, easy_path)` matches its `main` call. ✓
