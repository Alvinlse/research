"""
Exp 51 Phase A — the referee WRITES ITS OWN MANUAL from experience (2026-07-16).

Extends the referee pivot (research_plan Focus / Goal 1): Exp 49/50 showed the r1:32b
referee is feasible but its judgment is *unaimed* under scarcity (won seed 3 by partial
prod grants, lost seed 2 by hedging besteffort ahead of incoming prod). Those lessons were
distilled into precedents BY HAND once; this module automates the loop — after every
training window the referee reviews its own decisions against the observed outcome (and
the floor's outcome on the same window, so "better/worse than the rigid rule" is explicit)
and proposes at most ONE precedent add/edit. The precedents are exactly the "manual" idea:
pay for a lesson once, reuse it forever — the scene cache already reuses IDENTICAL states;
a precedent generalises the ruling to the whole WHEN-class.

Division of labour (the design rule, adapted): the LLM writes the CONTENT of the manual;
deterministic code owns the MECHANICS — validation, id assignment, the entry cap, dedup,
rendering. The store is `manual_learned.json`; `referee_manual_learned.md` is rendered
from it with the same PROMPT-START marker `referee.py` loads, so Phase B (frozen eval)
is just `PINS_MANUAL=pins/referee_manual_learned.md trace_replay --referee ...`.

Anti-leak: training seeds start at 100 — disjoint from the eval seeds 0..31, and the
manual is FROZEN before any measured run (a manual that grows mid-eval makes seed k depend
on seeds <k and breaks the paired stats).

Run:  .venv/bin/python -m pins.manual_author --model deepseek-r1:32b --seeds 32
      (one background task; ~1-3 min of r1 per uncached referee call + 1 reflection/window)
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random

from pins.llm_agent import CTX_OPT, HOST, _parse, load_cache, save_cache
from pins.referee import make_policy_referee, set_manual
from pins.trace_replay import TRACES, load_trace, make_trace_workload
from pins.two_sided_sim import policy_none, simulate
from pins.uncertainty_sim import assign, load_uncertainty_distribution

HERE = os.path.dirname(os.path.abspath(__file__))
STORE = os.path.join(HERE, "manual_learned.json")          # source of truth (code-owned)
RENDER = os.path.join(HERE, "referee_manual_learned.md")   # what the referee/user reads
LOG = os.path.join(HERE, "manual_author_log.jsonl")        # one line per window: audit trail
MAX_ENTRIES = 12                                           # prompt budget: num_ctx 8192

SYSTEM_REFLECT = (
    "You are the REFEREE of a supercomputer GPU pool, reviewing your own allocations on one "
    "finished scheduling window. You see: the window's outcome under your rulings, the outcome "
    "of a rigid rule baseline on the SAME jobs, your decisions with their justifications, "
    "per-job results, and your current manual of precedents.\n"
    "Propose at most ONE change to the manual that would improve future rulings on windows "
    "like this: a new precedent, an edit to an existing one, or none if there is no "
    "transferable lesson. A precedent MUST be conditioned on observable cluster state (the "
    "WHEN clause) — the same request can deserve opposite rulings in different states, so an "
    "unconditioned rule is a bug. Learn from wins too: if a ruling worked, say when to repeat "
    "it. Prefer editing/sharpening an existing precedent over adding a near-duplicate.\n"
    "THE WHEN CLAUSE IS MATCHED PER DECISION, NOT PER WINDOW. Use ONLY these exact field "
    "names and values — they are the whole vocabulary the referee sees when it rules; any "
    "other field does not exist and your precedent will never match:\n"
    "  free_gpus            int  — GPUs free at that tick (see free_gpus_seen for this "
    "window's actual distribution)\n"
    "  tier                 'prod' | 'besteffort'                (per demand job)\n"
    "  deadline             'behind' | 'ontrack' | 'ahead'       (per demand job)\n"
    "  base_gpus            int                                  (per demand job)\n"
    "  requested_margin_gpus int                                 (per demand job)\n"
    "  requested_reserve_gpus int                                (supply side)\n"
    "  incoming_prod        'none' | 'few' | 'many'  — a CATEGORY, never a number: write "
    "incoming_prod != 'none', NOT incoming_prod > 0\n"
    "The window summary (pool_size_gpus, n_jobs) is CONSTANT across every decision — a WHEN "
    "clause referencing it matches always or never and teaches nothing.\n"
    "Your rule must be SATISFIABLE wherever it fires: check it against `free_gpus_seen` "
    "before proposing. A rule like 'reserve 3 GPUs' is dead if free_gpus is 0 or 1 in most "
    "decisions. Pick thresholds that actually occur, and prefer a rule that fires often.\n"
    "Respond with ONLY this JSON object:\n"
    '{"action": "add" | "edit" | "none", "id": "<Pn, required for edit>", '
    '"when": "<observable state condition>", "rule": "<the ruling to apply>", '
    '"why": "<one sentence: the outcome evidence from this window>"}'
)


# --------------------------------------------------------------------------- #
#  Manual store: code owns ids, cap, rendering                                  #
# --------------------------------------------------------------------------- #
def load_store() -> list[dict]:
    return json.load(open(STORE)) if os.path.exists(STORE) else []


def render(entries: list[dict]) -> str:
    """Deterministic md render; the block below PROMPT-START is what referee.py injects."""
    lines = ["# Referee manual — SELF-AUTHORED (Exp 51 Phase A; do not hand-edit, "
             "regenerate via pins/manual_author.py)", "",
             "<!-- PROMPT-START: everything below this line is sent to the referee -->"]
    if entries:
        lines += ["PRECEDENTS from past allocations (apply when the WHEN clause matches; "
                  "cite the precedent id in your justification):", ""]
        for e in entries:
            lines += [f"{e['id']}. WHEN {e['when']}:", f"    {e['rule']}",
                      f"    [learned: {e['why']}]", ""]
    return "\n".join(lines)


def save_store(entries: list[dict]) -> str:
    entries.sort(key=lambda e: int(e["id"][1:]))
    json.dump(entries, open(STORE, "w"), indent=1)
    text = render(entries)
    open(RENDER, "w").write(text)
    return text.split("PROMPT-START", 1)[1].split("-->", 1)[1].strip() if entries else ""


def apply_proposal(entries: list[dict], prop: dict | None) -> str:
    """Validate and apply one reflection proposal. Returns what happened (for the log)."""
    if not isinstance(prop, dict) or prop.get("action") not in ("add", "edit", "none"):
        return "rejected: unparseable"
    if prop["action"] == "none":
        return "none"
    when = str(prop.get("when", "")).strip()
    rule = str(prop.get("rule", "")).strip()
    why = str(prop.get("why", "")).strip()[:250]
    if not when or not rule:
        return "rejected: empty when/rule"
    if prop["action"] == "edit":
        for e in entries:
            if e["id"] == str(prop.get("id", "")).strip():
                e.update(when=when, rule=rule, why=why)
                return f"edit {e['id']}"
        return f"rejected: edit of unknown id {prop.get('id')!r}"
    if len(entries) >= MAX_ENTRIES:
        return f"rejected: manual full ({MAX_ENTRIES}); edit instead"
    nid = f"P{max((int(e['id'][1:]) for e in entries), default=0) + 1}"
    entries.append({"id": nid, "when": when, "rule": rule, "why": why})
    return f"add {nid}"


# --------------------------------------------------------------------------- #
#  One training window: floor + referee + reflection                            #
# --------------------------------------------------------------------------- #
def reflect(model: str, payload: dict) -> dict | None:
    import ollama
    resp = ollama.Client(host=HOST).chat(
        model=model, format="json",
        options={"temperature": 0, "num_predict": 4096, **CTX_OPT},
        messages=[{"role": "system", "content": SYSTEM_REFLECT},
                  {"role": "user", "content": json.dumps(payload, indent=1)}])
    return _parse(resp.message.content)


def run(model: str, n_seeds: int, seed_start: int, pool: int, statement_model: str) -> None:
    trace = load_trace(TRACES["v2020"][0])
    dist = load_uncertainty_distribution()
    cache = load_cache()
    entries = load_store()
    set_manual(save_store(entries))           # referee sees the manual as it grows
    horizon, n_jobs, spike_max, scale = 300, 16, 0.6, 3   # base world (Exp 28/29 recipe)

    for s in range(seed_start, seed_start + n_seeds):
        jobs, cap_map, tcap, _ = make_trace_workload(trace, n_jobs, s, horizon, None,
                                                     False, None, False, None, None,
                                                     TRACES["v2020"][1], 1)
        cap_map = {k: min(v, pool) for k, v in cap_map.items()}
        tcap = {k: min(v, pool) for k, v in tcap.items()}
        u_map, spike_map = assign(jobs, s, dist, spike_max)
        floor = simulate(jobs, policy_none, pool, horizon, u_map, spike_map, scale,
                         spike_max, cap_map, true_cap_map=tcap)
        decisions: list = []
        ref = simulate(jobs, make_policy_referee(True, model, cache, decisions, set(),
                                                 statement_model=statement_model),
                       pool, horizon, u_map, spike_map, scale, spike_max, cap_map,
                       true_cap_map=tcap)
        save_cache(cache)

        outcomes = [{"job": j.jid, "tier": j.tier, "arrival": j.arrival,
                     "deadline": j.deadline, "done_at": ref["done_at"].get(j.jid),
                     "violated": ref["done_at"].get(j.jid) is None
                                 or ref["done_at"][j.jid] > j.deadline}
                    for j in jobs]
        payload = {
            "window": {"pool_size_gpus": pool, "n_jobs": n_jobs, "seed": s},
            # the states a WHEN clause is actually matched against (P1 was written over the
            # constant pool size and fired in 5/925 scenes — condition on these instead)
            "free_gpus_seen": dict(sorted(collections.Counter(
                d["free_gpus"] for d in decisions).items())),
            "your_outcome": {k: round(ref[k], 3) for k in
                             ("sla", "prod_sla", "util", "fallback_rate")},
            "rigid_rule_baseline_outcome": {k: round(floor[k], 3) for k in
                                            ("sla", "prod_sla", "util")},
            "your_decisions": decisions[:12],
            "job_outcomes": outcomes,
            "current_manual": entries,
        }
        verdict = "reflection failed"
        try:
            verdict = apply_proposal(entries, reflect(model, payload))
        except Exception as e:
            verdict = f"reflection failed: {type(e).__name__}: {e}"
        set_manual(save_store(entries))
        with open(LOG, "a") as f:
            f.write(json.dumps({"seed": s, "verdict": verdict,
                                "d_prod_sla": round(ref["prod_sla"] - floor["prod_sla"], 3),
                                "fallback_rate": round(ref["fallback_rate"], 3),
                                "n_entries": len(entries)}) + "\n")
        print(f"seed {s}: dprodSLA {ref['prod_sla'] - floor['prod_sla']:+.3f} "
              f"fb {ref['fallback_rate']:.0%} -> {verdict} ({len(entries)} entries)")

    print(f"\nmanual: {len(entries)} entries -> {RENDER}")
    print("freeze it, then eval:  PINS_MANUAL=pins/referee_manual_learned.md "
          f".venv/bin/python -m pins.trace_replay --referee --llm --model <m> --seeds 32 --pools {pool}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-r1:32b")
    ap.add_argument("--seeds", type=int, default=32)
    ap.add_argument("--seed-start", type=int, default=100)   # disjoint from eval seeds 0..31
    ap.add_argument("--pool", type=int, default=8)
    ap.add_argument("--statement-model", default="qwen2.5:3b")  # submissions fixed, as in Exp 50
    args = ap.parse_args()
    run(args.model, args.seeds, args.seed_start, args.pool, args.statement_model)


if __name__ == "__main__":
    main()
