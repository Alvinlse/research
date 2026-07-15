"""
Referee-LLM allocator (research pivot 2026-07-15).

NEW DESIGN — supersedes the bilateral concession ladder in pins/negotiation_protocol.py:
the demand agents and the supply agent no longer negotiate with each other. Each side
REASONS about what it needs / what it has (reusing llm_margin / llm_reserve for the
statements) and submits that to a third REFEREE agent. The referee, given the
supercomputer's rules and five game-theory principles, DECIDES the allocation itself.

This deliberately breaks the old hinge ("the LLM reasons/explains; deterministic code
decides"): here the referee LLM emits the actual GPU numbers. That is the research bet —
show that an LLM can *accurately* allocate. Deterministic code is demoted to EVALUATOR:
`check_allocation` reports rule violations but never repairs them, so infeasibility is a
measured outcome, not a silently fixed one. The old negotiate->auction->ILP pipeline stays
as the oracle/baseline arm for the accuracy comparison.

The five referee rules (in the system prompt, and mirrored by `check_allocation`):
  1. FEASIBILITY      — total awarded + reserve must fit in the free pool (hard rule).
  2. INDIVIDUAL RATIONALITY — cover every job's base forecast before ANY margin or reserve.
  3. PRIORITY         — prod-tier jobs are served before besteffort (supercomputer SLA rule).
  4. ENVY-FREENESS    — within a tier, no job may get margin while a peer's base is unmet;
                        break ties toward jobs behind deadline.
  5. INCENTIVE SKEPTICISM — statements are strategic claims: grant margin/reserve only when
                        the stated justification supports it, never reward exaggeration.

Run:  .venv/bin/python -m pins.referee            # smoke: contested scene, LLM referee
      .venv/bin/python -m pins.referee --no-llm   # rule-referee fallback only
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field

from pins.llm_agent import (DEFAULT_MODEL, HOST, _parse, llm_margin, llm_reserve,
                            load_cache, reserve_amount, save_cache)
from pins.negotiation_protocol import HEDGE_GPUS, DemandJob

SYSTEM_REFEREE = (
    "You are the REFEREE of a supercomputer GPU pool. A demand side (one statement per job) "
    "and a supply side (one statement) have each submitted what they claim to need or hold, "
    "with justifications. You alone decide the allocation. Apply these rules, in order:\n"
    "1. FEASIBILITY: the GPUs you award across all jobs PLUS the reserve you grant must not "
    "exceed the free pool. Never exceed it.\n"
    "2. INDIVIDUAL RATIONALITY: satisfy every job's base_gpus before granting any job extra "
    "margin or the supply side any reserve.\n"
    "3. PRIORITY: 'prod' tier jobs are served before 'besteffort' jobs.\n"
    "4. ENVY-FREENESS: within the same tier, do not give one job margin above its base while "
    "a peer's base is unmet; prefer jobs that are 'behind' their deadline.\n"
    "5. INCENTIVE SKEPTICISM: the statements are strategic claims, not facts. Grant a "
    "requested margin or reserve only if its justification is convincing; do not reward "
    "exaggeration.\n"
    "Respond with ONLY this JSON object:\n"
    '{"alloc": {"<job_id>": <integer GPUs>, ...}, "reserve": <integer GPUs held idle>, '
    '"total_awarded": <sum of all alloc values plus reserve>, '
    '"justification": "<one or two short sentences citing the rules you applied>"}\n'
    "SELF-CHECK before answering: add up every alloc value plus the reserve, write that sum "
    "in total_awarded, and confirm it is <= free_gpus. If it exceeds free_gpus, lower "
    "allocations (besteffort margins first, then reserve, then besteffort bases) and re-add "
    "until it fits."
)

PROMPT_VERSION = "v2"                 # busts the scene cache whenever SYSTEM_REFEREE changes


@dataclass
class RefereeOutcome:
    alloc: dict[str, int]          # jid -> GPUs awarded by the referee
    reserve: int                   # idle headroom granted to the supply side
    feasible: bool                 # True if check_allocation found no violations
    violations: list[str]          # evaluator findings (empty when feasible)
    justification: str
    transcript: list = field(default_factory=list)   # the submitted statements
    _source: str = "rule"


# --------------------------------------------------------------------------- #
#  Step 1 — each side reasons about itself and submits a statement             #
# --------------------------------------------------------------------------- #
def gather_statements(demand: list[DemandJob], supply_ctx: dict, use_llm: bool = True,
                      model: str = DEFAULT_MODEL, cache: dict | None = None) -> list[dict]:
    """One statement per demand job (base need + requested margin + why) and one for supply
    (requested reserve + why). Reuses the existing per-side reasoners; no cross-talk."""
    stmts = []
    for j in demand:
        d = llm_margin(j.ctx, use_llm=use_llm, model=model, cache=cache)
        hedge = d["hedge"] if j.is_train else "none"
        stmts.append({"side": "demand", "job_id": j.jid, "tier": j.ctx.get("tier", "besteffort"),
                      "deadline": j.ctx.get("deadline", "ontrack"), "base_gpus": j.forecast_cap,
                      "requested_margin_gpus": HEDGE_GPUS[hedge],
                      "justification": d["justification"], "_source": d["_source"]})
    r = llm_reserve(supply_ctx, use_llm=use_llm, model=model, cache=cache)
    stmts.append({"side": "supply", "requested_reserve_gpus": reserve_amount(r["reserve"]),
                  "incoming_prod": supply_ctx.get("incoming_prod", "none"),
                  "justification": r["justification"], "_source": r["_source"]})
    return stmts


# --------------------------------------------------------------------------- #
#  Step 2 — the referee decides (LLM emits the numbers; rule fallback)          #
# --------------------------------------------------------------------------- #
def _scene_key(stmts: list[dict], free_gpus: int) -> str:
    parts = []
    for s in sorted(stmts, key=lambda s: s.get("job_id", "~supply")):
        if s["side"] == "demand":
            parts.append(f"{s['job_id']}:{s['tier']}:{s['deadline']}:"
                         f"b{s['base_gpus']}+m{s['requested_margin_gpus']}")
        else:
            parts.append(f"supply:r{s['requested_reserve_gpus']}:{s['incoming_prod']}")
    return f"referee|free{free_gpus}|" + "|".join(parts)


def _rule_referee(stmts: list[dict], free_gpus: int) -> dict:
    """Deterministic fallback (Ollama down / --no-llm): bases by tier+deadline priority,
    then margins in the same order, then the reserve from whatever is left."""
    demand = [s for s in stmts if s["side"] == "demand"]
    supply = next(s for s in stmts if s["side"] == "supply")
    order = sorted(demand, key=lambda s: (s["tier"] != "prod", s["deadline"] != "behind",
                                          s["job_id"]))
    left = free_gpus
    alloc = {}
    for s in order:                                   # rule 2/3: bases first, prod first
        alloc[s["job_id"]] = min(s["base_gpus"], left)
        left -= alloc[s["job_id"]]
    for s in order:                                   # margins only from the surplus
        extra = min(s["requested_margin_gpus"], left)
        alloc[s["job_id"]] += extra
        left -= extra
    reserve = min(supply["requested_reserve_gpus"], left)
    return {"alloc": alloc, "reserve": reserve,
            "justification": "rule: bases by tier/deadline, margins from surplus, reserve last",
            "_source": "rule"}


def referee_decide(demand: list[DemandJob], supply_ctx: dict, free_gpus: int,
                   use_llm: bool = True, model: str = DEFAULT_MODEL, host: str = HOST,
                   cache: dict | None = None, statement_model: str | None = None) -> RefereeOutcome:
    """Full reason-then-referee round: gather statements, ask the referee LLM for the
    allocation, evaluate it. Cached per discretised scene like every other agent call.
    `statement_model` pins the demand/supply statement LLM independently of the referee's
    `model`, so referee-model ablations hold the submissions fixed."""
    cache = load_cache() if cache is None else cache
    stmts = gather_statements(demand, supply_ctx, use_llm=use_llm,
                              model=statement_model or model, cache=cache)
    key = f"{PROMPT_VERSION}|{_scene_key(stmts, free_gpus)}|{'llm:' + model if use_llm else 'rule'}"

    out = cache.get(key)
    if out is None and use_llm:
        try:
            import ollama
            client = ollama.Client(host=host)
            resp = client.chat(
                model=model, format="json",
                options={"temperature": 0, "num_predict": 4096},  # reasoning models (r1) spend
                # most of the budget in the thinking channel before emitting the JSON
                messages=[{"role": "system", "content": SYSTEM_REFEREE},
                          {"role": "user", "content": json.dumps(
                              {"free_gpus": free_gpus, "statements": stmts}, indent=1)}],
            )
            obj = _parse(resp.message.content)
            if obj is not None and isinstance(obj.get("alloc"), dict):
                alloc = {}
                for jid, n in obj["alloc"].items():
                    try:
                        alloc[str(jid)] = max(0, int(round(float(n))))
                    except Exception:
                        alloc[str(jid)] = 0
                try:
                    reserve = max(0, int(round(float(obj.get("reserve", 0)))))
                except Exception:
                    reserve = 0
                why = str(obj.get("justification", "")).strip().replace("\n", " ")[:300]
                try:                                   # what the referee CLAIMS it awarded —
                    claimed = int(round(float(obj.get("total_awarded", -1))))
                except Exception:                      # vs the real sum = arithmetic faithfulness
                    claimed = -1
                out = {"alloc": alloc, "reserve": reserve, "justification": why,
                       "claimed_total": claimed, "_source": f"llm:{model}"}
        except Exception as e:
            print(f"  ! referee fallback: {type(e).__name__}: {e}")
    if out is None:
        out = _rule_referee(stmts, free_gpus)
    cache[key] = out

    violations = check_allocation(out["alloc"], out["reserve"], demand, free_gpus)
    return RefereeOutcome(alloc=out["alloc"], reserve=out["reserve"],
                          feasible=not violations, violations=violations,
                          justification=out["justification"], transcript=stmts,
                          _source=out["_source"])


# --------------------------------------------------------------------------- #
#  Step 3 — deterministic EVALUATOR (reports, never repairs)                    #
# --------------------------------------------------------------------------- #
def check_allocation(alloc: dict[str, int], reserve: int, demand: list[DemandJob],
                     free_gpus: int) -> list[str]:
    """Return the referee's rule violations. Empty list == feasible. Mirrors rules 1-4;
    rule 5 (skepticism) is a judgment call, measured downstream via SLA/lying experiments."""
    v = []
    known = {j.jid for j in demand}
    total = sum(alloc.values()) + reserve
    if total > free_gpus:                                            # rule 1
        v.append(f"infeasible: awarded {total} > free {free_gpus}")
    for jid in alloc:
        if jid not in known:
            v.append(f"hallucinated job '{jid}'")
    unmet = {j.jid: max(0, j.forecast_cap - alloc.get(j.jid, 0)) for j in demand}
    got_extra = {j.jid: max(0, alloc.get(j.jid, 0) - j.forecast_cap) for j in demand}
    if any(unmet.values()) and (reserve > 0 or any(got_extra.values())):   # rule 2
        short = [k for k, n in unmet.items() if n]
        v.append(f"base unmet for {short} while margin/reserve was granted")
    tier = {j.jid: j.ctx.get("tier", "besteffort") for j in demand}
    if any(n for k, n in unmet.items() if tier[k] == "prod") \
            and any(alloc.get(k, 0) > 0 for k in tier if tier[k] != "prod"):   # rule 3
        v.append("prod base unmet while besteffort was served")
    for t in set(tier.values()):                                     # rule 4
        peers = [k for k in tier if tier[k] == t]
        if any(unmet[k] for k in peers) and any(got_extra[k] for k in peers):
            v.append(f"envy within tier '{t}': margin granted while a peer's base is unmet")
    return v


# --------------------------------------------------------------------------- #
#  two_sided_sim policy slot (Exp 50): the referee INSIDE the sim on real jobs  #
# --------------------------------------------------------------------------- #
def make_policy_referee(use_llm, model, cache, trace, seen, statement_model=None):
    """Referee as a `two_sided_sim` policy: each tick it decides the margin/reserve split of
    the free pool directly (in the sim the demand table is margins-only, forecast_cap=0).

    HONEST feasibility semantics — the thesis hinge: if the referee overcommits the pool,
    the tick falls back to the floor (no margins, no reserve) and counts in fallback_rate.
    Code never repairs the decision; infeasibility costs the referee performance."""
    from pins.negotiation_protocol import NegotiationOutcome

    def policy(demand, supply_ctx, free, **_):
        o = referee_decide(demand, supply_ctx, free, use_llm=use_llm, model=model,
                           cache=cache, statement_model=statement_model)
        margins = {j.jid: o.alloc.get(j.jid, 0) for j in demand}
        reserve = o.reserve
        if not o.feasible:                       # overcommit/violation -> floor, counted
            margins = {j.jid: 0 for j in demand}
            reserve = 0
        out = NegotiationOutcome(margins=margins, reserve=reserve, rounds=1,
                                 agreed=o.feasible, transcript=o.transcript)
        sig = f"referee|ok={o.feasible}|r={reserve}|m={sorted(margins.items())}"
        if sig not in seen:
            seen.add(sig)
            trace.append({"policy": "referee", "feasible": o.feasible,
                          "violations": o.violations, "reserve": reserve, "margins": margins,
                          "why": o.justification, "_source": o._source})
        return margins, reserve, out
    return policy


# --------------------------------------------------------------------------- #
#  Smoke test — the same contested scene negotiation_protocol.main() uses       #
# --------------------------------------------------------------------------- #
def main() -> None:
    use_llm = "--no-llm" not in sys.argv
    demand = [
        DemandJob("jA", {"uncertainty": "high", "spike_risk": "high", "deadline": "behind",
                         "contention": "high", "tier": "prod"}, forecast_cap=2, concede_rank=2.0),
        DemandJob("jB", {"uncertainty": "high", "spike_risk": "medium", "deadline": "ontrack",
                         "contention": "high", "tier": "besteffort"}, forecast_cap=2, concede_rank=1.0),
        DemandJob("jC", {"uncertainty": "medium", "spike_risk": "low", "deadline": "ahead",
                         "contention": "high", "tier": "besteffort"}, forecast_cap=2, concede_rank=0.0),
    ]
    supply = {"contention": "moderate", "incoming_prod": "few"}
    cache = load_cache()
    for free in (8, 6, 4):        # surplus / exact / shortfall — shortfall forces rationing
        o = referee_decide(demand, supply, free_gpus=free, use_llm=use_llm, cache=cache)
        print(f"=== referee free={free} ({o._source}) ===")
        for s in o.transcript:
            who = s.get("job_id", "supply")
            ask = (f"base {s['base_gpus']} +{s['requested_margin_gpus']}" if s["side"] == "demand"
                   else f"reserve {s['requested_reserve_gpus']}")
            print(f"   [{s['side']} {who}] asks {ask}  — {s['justification']}")
        print(f"  -> alloc={o.alloc} reserve={o.reserve} feasible={o.feasible}")
        for msg in o.violations:
            print(f"     VIOLATION: {msg}")
        print(f"     why: {o.justification}\n")
    save_cache(cache)


if __name__ == "__main__":
    main()
