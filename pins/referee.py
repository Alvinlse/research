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

import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass, field

from pins.llm_agent import (metered_client, CTX_OPT, DEFAULT_MODEL, HOST, MARGIN_HEDGES, RESERVE_LEVELS, _parse,
                            llm_margin, llm_reserve, load_cache, reserve_amount, save_cache)
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

# Exp 58/E1 (research_plan Phase 6): the referee gains MEMORY of its own last executed
# ruling plus a qualitative change-cost note. Appended to SYSTEM_REFEREE only when
# `prev_alloc` is passed, so every existing tier's prompt — and cache key — is untouched.
RULE6_PREV = (
    "\n6. CONTINUITY: you are also given previous_allocation — the allocation that "
    "actually executed last epoch. Changing a job's allocation costs part of its progress "
    "(checkpoint/restart), so KEEP an allocation unless the statements justify changing "
    "it. Do not keep an allocation that rules 1-5 now argue against."
)

# Exp 63 (revision 2): the referee also decides ADMISSION — which queued jobs start now and
# which keep waiting. Appended only when `waiting` is passed, so every pre-admission tier's
# prompt — and cache key — is untouched (the RULE6_PREV pattern).
RULE7_ADMIT = (
    "\n7. ADMISSION: you are also given waiting_jobs — queued jobs that have not started. "
    "After margins and reserve, admitted jobs draw their base_gpus from what remains of the "
    "free pool, in the order you list them; a job that only partly fits runs SLOWER, and "
    "spreading a scarce pool thinly slows everyone. List jobs to hold back this epoch in "
    '"defer" (they wait at zero GPUs; only never-started jobs can be deferred) and the '
    'admitted ones in "admit_order", most important first. Never defer a prod job while '
    "admitting a besteffort one. Deferral is rationing, not punishment: defer only when the "
    "remaining pool cannot usefully cover the job's base."
)

def _HYBRID(model: str) -> bool:      # models whose API accepts think=; the rest 400 on it
    return model.startswith(("deepseek-r1", "qwen3"))

_MANUAL = ""       # precedent block appended to SYSTEM_REFEREE (Exp 51 manual arm)
_MANUAL_TAG = ""   # cache-key component: manual hash, so manual/vanilla rulings never mix


def set_manual(text: str) -> None:
    """Install the precedent block (raw text, no markers). Empty string turns the arm off."""
    global _MANUAL, _MANUAL_TAG
    _MANUAL = text.strip()
    _MANUAL_TAG = f"|man:{hashlib.sha1(_MANUAL.encode()).hexdigest()[:8]}" if _MANUAL else ""


def load_manual(path: str) -> None:
    """Load a manual file: everything below its PROMPT-START marker is the block."""
    text = open(path).read()
    set_manual(text.split("PROMPT-START", 1)[1].split("-->", 1)[1])


if os.environ.get("PINS_MANUAL"):     # e.g. PINS_MANUAL=pins/referee_manual_learned.md
    load_manual(os.environ["PINS_MANUAL"])


@dataclass
class RefereeOutcome:
    alloc: dict[str, int]          # jid -> GPUs awarded by the referee
    reserve: int                   # idle headroom granted to the supply side
    feasible: bool                 # True if check_allocation found no violations
    violations: list[str]          # evaluator findings (empty when feasible)
    justification: str
    transcript: list = field(default_factory=list)   # the submitted statements
    _source: str = "rule"
    defer: frozenset = frozenset()               # Exp 63: waiting jids held back this epoch
    priority: dict | None = None                 # Exp 63: admit order (higher = served first)


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
#  Step 1b — CROSS-TALK: each side sees the other and revises, still partisan   #
# --------------------------------------------------------------------------- #
SYSTEM_REBUT_DEMAND = (
    "You are the ADVOCATE for ONE job competing for a shared GPU pool. You have already stated "
    "your opening position. You can now see the supply side's position and your competitors'. "
    "Argue your job's case to the referee who will decide.\n"
    "You may keep or lower your margin request; you may NOT raise it, and your base need is "
    "fixed. Conceding margin you cannot justify makes your remaining claim more credible — the "
    "referee is explicitly skeptical of unsupported asks. But do not concede a margin your job "
    "genuinely needs just because others want it.\n"
    'Reply JSON: {"hedge": "none|some|heavy", "justification": "<one sentence to the referee>"}')

SYSTEM_REBUT_SUPPLY = (
    "You are the SUPPLY agent for a shared GPU cluster, holding headroom against incoming "
    "high-priority work. You have already stated your opening reserve. You can now see every "
    "job's stated need and justification. Argue your case to the referee who will decide.\n"
    "You may keep or lower your reserve; you may NOT raise it. If the demand on the table is "
    "more urgent than the work you are holding headroom for, releasing it is the right call — "
    "but do not release headroom that genuinely protects incoming prod work.\n"
    'Reply JSON: {"reserve": "none|light|heavy", "justification": "<one sentence to the '
    'referee>"}')


def _opposing_view(stmts: list[dict], me: str | None) -> list[dict]:
    """What one side is allowed to see: everyone else's position + why, never private ctx."""
    view = []
    for s in stmts:
        if s["side"] == "demand" and s["job_id"] != me:
            view.append({"who": f"job {s['job_id']}", "tier": s["tier"],
                         "deadline": s["deadline"], "needs_base": s["base_gpus"],
                         "wants_margin": s["requested_margin_gpus"],
                         "argues": s["justification"]})
        elif s["side"] == "supply":
            view.append({"who": "supply", "wants_reserve": s["requested_reserve_gpus"],
                         "incoming_prod": s["incoming_prod"], "argues": s["justification"]})
    return view


def _rebut_call(system: str, payload: dict, field: str, levels: list[str], model: str,
                host: str, cache: dict) -> tuple[str | None, str]:
    """One partisan revision. Returns (level, justification); level None => keep opening."""
    key = ("rebut|" + hashlib.sha1(
        (system[:40] + json.dumps(payload, sort_keys=True) + model).encode()).hexdigest())
    if key in cache:
        c = cache[key]
        return c["level"], c["why"]
    level, why = None, ""
    try:
        import ollama
        resp = metered_client(host).chat(
            model=model, format="json", think=False,
            options={"temperature": 0, "num_predict": 120, **CTX_OPT},
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": json.dumps(payload, indent=1)}])
        obj = _parse(resp.message.content)
        if obj is not None:
            lv = str(obj.get(field, "")).strip().lower()
            if lv in levels:
                level = lv
            why = str(obj.get("justification", "")).strip().replace("\n", " ")[:200]
    except Exception as e:
        print(f"  ! rebut fallback: {type(e).__name__}: {e}")
    cache[key] = {"level": level, "why": why}
    return level, why


def rebut(stmts: list[dict], free_gpus: int, use_llm: bool = True, model: str = DEFAULT_MODEL,
          host: str = HOST, cache: dict | None = None) -> list[dict]:
    """DEBATE round: each side reads the opposing positions and revises its own ask, still
    arguing only its own corner. Returns statements in the same schema gather_statements emits,
    so referee_decide consumes them unchanged (that keeps the debate/no-debate A/B honest).

    Monotone by construction: asks may only shrink. That guarantees termination in one round and
    means the round can only free capacity, never manufacture contention. `_r0_gpus` records the
    opening ask so downstream can measure the concession rate -- if nobody ever moves the debate
    is decoration, and if everyone always caves it is sycophancy (both are theatre, not signal).
    """
    if not use_llm:
        return stmts
    cache = load_cache() if cache is None else cache
    out = []
    for s in stmts:
        s = dict(s)
        if s["side"] == "demand":
            lv, why = _rebut_call(
                SYSTEM_REBUT_DEMAND,
                {"free_gpus": free_gpus, "my_job": s["job_id"], "my_tier": s["tier"],
                 "my_deadline": s["deadline"], "my_base_gpus": s["base_gpus"],
                 "my_opening_margin_gpus": s["requested_margin_gpus"],
                 "my_opening_argument": s["justification"],
                 "others": _opposing_view(stmts, s["job_id"])},
                "hedge", MARGIN_HEDGES, model, host, cache)
            s["_r0_gpus"] = s["requested_margin_gpus"]
            if lv is not None:                       # monotone: never let the round inflate an ask
                s["requested_margin_gpus"] = min(HEDGE_GPUS[lv], s["requested_margin_gpus"])
                s["justification"] = why or s["justification"]
        else:
            lv, why = _rebut_call(
                SYSTEM_REBUT_SUPPLY,
                {"free_gpus": free_gpus, "my_opening_reserve_gpus": s["requested_reserve_gpus"],
                 "incoming_prod": s["incoming_prod"], "my_opening_argument": s["justification"],
                 "others": _opposing_view(stmts, None)},
                "reserve", RESERVE_LEVELS, model, host, cache)
            s["_r0_gpus"] = s["requested_reserve_gpus"]
            if lv is not None:
                s["requested_reserve_gpus"] = min(reserve_amount(lv),
                                                  s["requested_reserve_gpus"])
                s["justification"] = why or s["justification"]
        s["_debated"] = True
        out.append(s)
    return out


# --------------------------------------------------------------------------- #
#  Step 2 — the referee decides (LLM emits the numbers; rule fallback)          #
# --------------------------------------------------------------------------- #
def _roles(stmts: list[dict]) -> list[dict]:
    """Demand statements in canonical ROLE order — the order a delta-triggered ruling is
    indexed by. Identity (job_id) is deliberately excluded: two scenes with the same tiers,
    deadlines and asks are the same *situation*, whatever the jobs happen to be called."""
    return sorted((s for s in stmts if s["side"] == "demand"),
                  key=lambda s: (s["tier"], s["deadline"], -s["base_gpus"],
                                 -s["requested_margin_gpus"]))


def _alloc_to_slots(alloc: dict, stmts: list[dict]) -> dict:
    """jid-keyed award -> slot-keyed, so a delta ruling can be reused by other jobs."""
    return {f"s{i}": int(alloc.get(s["job_id"], 0)) for i, s in enumerate(_roles(stmts))}


def _alloc_from_slots(slots: dict, stmts: list[dict]) -> dict:
    """slot-keyed award -> jid-keyed for THIS scene's jobs (inverse of _alloc_to_slots)."""
    return {s["job_id"]: int(slots.get(f"s{i}", 0)) for i, s in enumerate(_roles(stmts))}


def _scene_key(stmts: list[dict], free_gpus: int, trigger: str = "bucket") -> str:
    parts = []
    if trigger == "delta":
        # Exp 57: role-indexed, identity-free. `bucket` (default) keys every distinct job_id
        # separately, so a structurally identical scene re-fires the LLM for a new job — the
        # ~4.7x invocation gap vs the negotiated arm. Here slot i is "the i-th job in role
        # order", so the cached ruling transfers to any scene with the same shape.
        for i, s in enumerate(_roles(stmts)):
            parts.append(f"s{i}:{s['tier']}:{s['deadline']}:"
                         f"b{s['base_gpus']}+m{s['requested_margin_gpus']}")
        sup = [s for s in stmts if s["side"] != "demand"]
        for s in sup:
            parts.append(f"supply:r{s['requested_reserve_gpus']}:{s['incoming_prod']}")
        key = f"referee-d|free{free_gpus}|" + "|".join(parts)
        if any(s.get("_debated") for s in stmts):
            key += "|dbt" + hashlib.sha1(
                "|".join(s["justification"] for s in stmts).encode()).hexdigest()[:12]
        return key
    for s in sorted(stmts, key=lambda s: s.get("job_id", "~supply")):
        if s["side"] == "demand":
            parts.append(f"{s['job_id']}:{s['tier']}:{s['deadline']}:"
                         f"b{s['base_gpus']}+m{s['requested_margin_gpus']}")
        else:
            parts.append(f"supply:r{s['requested_reserve_gpus']}:{s['incoming_prod']}")
    key = f"referee|free{free_gpus}|" + "|".join(parts)
    if any(s.get("_debated") for s in stmts):
        # A debate round rewrites the ARGUMENTS even when the numbers land back where they
        # opened, and the referee reads those arguments -- so the discretised numbers alone
        # would collide with the no-debate arm's cached ruling. Hash the text to separate them.
        key += "|dbt" + hashlib.sha1(
            "|".join(s["justification"] for s in stmts).encode()).hexdigest()[:12]
    return key


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
                   cache: dict | None = None, statement_model: str | None = None,
                   think: bool = True, stmts: list[dict] | None = None,
                   trigger: str = "bucket", no_argue: bool = False,
                   prev_alloc: dict | None = None,
                   waiting: list[dict] | None = None) -> RefereeOutcome:
    """Full reason-then-referee round: gather statements, ask the referee LLM for the
    allocation, evaluate it. Cached per discretised scene like every other agent call.
    `statement_model` pins the demand/supply statement LLM independently of the referee's
    `model`, so referee-model ablations hold the submissions fixed.
    `stmts` overrides the gathering step with hand-authored submissions — the hard-case suite
    (pins/hardcases.py) uses it to put an exceptional fact in a job's own justification."""
    cache = load_cache() if cache is None else cache
    if stmts is None:
        stmts = gather_statements(demand, supply_ctx, use_llm=use_llm,
                                  model=statement_model or model, cache=cache)
    if no_argue:
        # Exp 57f ablation: the referee sees the NUMBERS but none of the advocacy — isolates
        # what the two-sided statements contribute beyond structured quantities. Blanked
        # AFTER gathering so the demand/supply reasoning cost stays identical.
        stmts = [{**s, "justification": ""} for s in stmts]
    key = (f"{PROMPT_VERSION}{_MANUAL_TAG}|{_scene_key(stmts, free_gpus, trigger)}"
           f"|{'llm:' + model if use_llm else 'rule'}{'' if think else '|nothink'}"
           f"{'|noarg' if no_argue else ''}")
    if prev_alloc is not None:
        # v3-prev arm: the ruling depends on the executed history, so the history is part
        # of the scene identity — two ticks with equal statements but different pasts must
        # never share a cached ruling (the Exp 56/57g cache-collision lesson).
        key += "|prev:" + hashlib.sha1(
            json.dumps(sorted(prev_alloc.items())).encode()).hexdigest()[:10]
    if waiting is not None:
        # Exp 63: the ruling now includes WHO STARTS, so the waiting set is part of the scene
        # identity. jid included (defer names jobs); waited_ticks excluded (continuous — it
        # would kill every cache hit) on the discretise-the-asks-not-the-arguments precedent.
        key += "|adm:" + hashlib.sha1("|".join(
            f"{w['jid']}:{w['tier']}:{w['deadline']}:b{w['base_gpus']}"
            for w in sorted(waiting, key=lambda w: w["jid"])).encode()).hexdigest()[:10]
    if statement_model and statement_model != "qwen2.5:3b":
        # The scene key discretises the ASKS but not the arguments, and the referee reads the
        # arguments -- so a stronger advocate that lands on the same numbers would otherwise
        # reuse the 3b-advocate ruling. Keep advocate tiers on separate keys.
        key += f"|adv:{statement_model}"

    out = cache.get(key)
    if out is not None and trigger == "delta" and "alloc_slots" in out:
        out = dict(out, alloc=_alloc_from_slots(out["alloc_slots"], stmts))
    if out is None and use_llm:
        try:
            import ollama
            client = metered_client(host)
            resp = client.chat(
                # only hybrid reasoners accept the thinking channel; ollama now 400s on the
                # rest (it used to ignore the flag, which is how Exp 51-53 ran think=True on
                # 14b). Gate the API call, NOT `think` -- the cache key must stay unsuffixed
                # so those tiers keep replaying.
                model=model, format="json", think=think and _HYBRID(model),
                options={"temperature": 0, "num_predict": 4096, **CTX_OPT},  # reasoning models (r1) spend
                # most of the budget in the thinking channel before emitting the JSON
                messages=[{"role": "system",
                           "content": SYSTEM_REFEREE
                                      + (RULE6_PREV if prev_alloc is not None else "")
                                      + (RULE7_ADMIT if waiting is not None else "")
                                      + ("\n\n" + _MANUAL if _MANUAL else "")},
                          {"role": "user", "content": json.dumps(
                              {"free_gpus": free_gpus, "statements": stmts}
                              | ({"previous_allocation": prev_alloc}
                                 if prev_alloc is not None else {})
                              | ({"waiting_jobs": waiting}
                                 if waiting is not None else {}), indent=1)}],
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
                if waiting is not None:
                    # the referee's OWN admission call, unrepaired: unknown jids pass through
                    # harmlessly (the sim matches by jid); an empty defer = admit everyone.
                    out["defer"] = [str(x) for x in obj.get("defer", [])
                                    if isinstance(obj.get("defer"), list)]
                    order = obj.get("admit_order")
                    out["admit_order"] = [str(x) for x in order] if isinstance(order, list) else []
        except Exception as e:
            print(f"  ! referee fallback: {type(e).__name__}: {e}")
    if out is None:
        out = _rule_referee(stmts, free_gpus)
        if waiting is not None:                        # deterministic admission fallback
            from pins.negotiation_protocol import admission_plan
            prio, dfr = admission_plan(waiting, free_gpus, reserve=out["reserve"])
            out["defer"] = sorted(dfr)
            out["admit_order"] = sorted(prio, key=prio.get, reverse=True)
    if trigger == "delta":
        cache[key] = dict(out, alloc_slots=_alloc_to_slots(out["alloc"], stmts))
    else:
        cache[key] = out

    violations = check_allocation(out["alloc"], out["reserve"], demand, free_gpus,
                                  waiting_jids=frozenset(w["jid"] for w in waiting or ()))
    defer = frozenset(out.get("defer") or ())
    priority = None
    if waiting is not None:
        order = out.get("admit_order") or []
        priority = {jid: float(len(order) - i) for i, jid in enumerate(order)}
        # rule-3 analogue for admission, REPORTED never repaired: prod held back while a
        # besteffort peer starts is a violation the referee pays for, not one code fixes.
        tiers = {w["jid"]: w["tier"] for w in waiting}
        if any(tiers.get(j) == "prod" for j in defer) \
                and any(t != "prod" and j not in defer for j, t in tiers.items()):
            violations = violations + ["prod deferred while besteffort admitted"]
    return RefereeOutcome(alloc=out["alloc"], reserve=out["reserve"],
                          feasible=not violations, violations=violations,
                          justification=out["justification"], transcript=stmts,
                          _source=out["_source"], defer=defer, priority=priority)


# --------------------------------------------------------------------------- #
#  Step 3 — deterministic EVALUATOR (reports, never repairs)                    #
# --------------------------------------------------------------------------- #
def check_allocation(alloc: dict[str, int], reserve: int, demand: list[DemandJob],
                     free_gpus: int, waiting_jids: frozenset = frozenset()) -> list[str]:
    """Return the referee's rule violations. Empty list == feasible. Mirrors rules 1-4;
    rule 5 (skepticism) is a judgment call, measured downstream via SLA/lying experiments.
    `waiting_jids` (Exp 63): under rule 7 an award to a WAITING job's base is legitimate,
    not a hallucination — and it still counts toward the rule-1 total, as admitted bases
    genuinely draw from the free pool."""
    v = []
    known = {j.jid for j in demand} | set(waiting_jids)
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
# Exp 57 controller shell: per-(seed,arm) tick counters, read+reset by trace_replay the
# same way take_tokens() is. fast = ticks governed by an adapted standing ruling (no LLM);
# llm = ticks that invoked referee_decide.
SHELL_STATS = {"fast": 0, "llm": 0, "debate": 0}
QSTATS: dict = {}          # plan §12: last policy's quality-cache counters (reuse/false-reuse)


def take_shell_stats() -> dict:
    out = dict(SHELL_STATS)
    SHELL_STATS.update(fast=0, llm=0, debate=0)
    return out


def take_qcache_stats() -> dict:
    out = dict(QSTATS) or {"qcache_reuse": 0.0, "qcache_false": 0.0, "qcache_entries": 0.0}
    QSTATS.clear()
    return out


def make_policy_referee(use_llm, model, cache, trace, seen, statement_model=None, think=True,
                        debate=False, trigger="bucket", theta=None, stale="fresh",
                        extend=False, no_argue=False, prev_input=False, fast_negotiate=False,
                        hard_trigger=False, admit=False, gamma=None, qcache=None):
    assert not (debate and no_argue), "--debate rewrites the arguments --no-argue removes"
    assert not hard_trigger or debate, "--hard-trigger gates the debate round"
    assert gamma is None or debate, "--gamma gates the debate round"
    """Referee as a `two_sided_sim` policy: each tick it decides the margin/reserve split of
    the free pool directly (in the sim the demand table is margins-only, forecast_cap=0).

    HONEST feasibility semantics — the thesis hinge: if the referee overcommits the pool,
    the tick falls back to the floor (no margins, no reserve) and counts in fallback_rate.
    Code never repairs the decision; infeasibility costs the referee performance.

    `debate` inserts the cross-talk round (Step 1b) between the statements and the ruling: both
    sides read each other, then revise. The referee stage is byte-identical either way, so the
    debate/no-debate contrast isolates the round itself.

    Controller shell (Exp 57, from the LLMSched pipeline; default OFF so every prior tier
    replays unchanged):
      `theta` — Δ-trigger threshold. Each tick Δ = 0.4·|Δfree|/prev + 0.4·jobset-symdiff
        + 0.2·new-prod-behind. Δ<=theta -> FAST MODE: deterministically re-execute the
        STANDING ruling (keep awards of continuing jobs, drop departed, NEW jobs get 0 —
        their arrival raises Δ and normally fires the trigger). The LLM authored the standing
        ruling; fast mode only executes it, so decision authority stays with the referee.
        An adapted ruling that no longer fits the live pool re-fires the trigger instead of
        being repaired.
      `stale` — 'one' = pipelined epochs: the referee decides on the PREVIOUS tick's
        statements (the snapshot a concurrent 180 s planner would have), and the evaluator
        validates the ruling against the LIVE tick. 'fresh' (default) = decide on this tick.
      `fast_negotiate` — fast mode calls the cheap bounded-concession `negotiate()` protocol
        (pins/negotiation_protocol.py, same mechanism as the 'negotiated' arm) on the LIVE
        tick instead of replaying/extending the standing ruling. The referee still owns every
        tick past theta (novel/risky scenes); routine ticks go to the cheap auction instead of
        a frozen memory of the referee's last word. Mutually exclusive with `extend` (which
        only makes sense for the replay mechanism); requires theta.

    `hard_trigger` (E3) — gate the DELIBERATION, not the ruling. The rebuttal round fires only
    on a hard trigger (prod arrival, free-GPU bucket crossing, a job newly behind deadline, or
    a fallback last tick); routine ticks submit the opening statements straight to the referee,
    who still rules EVERY tick on live numbers. This is the layer 57g got wrong: it froze the
    ruling (measured load-bearing, arm B +3.08*) to save the cheap statements, whereas the
    expensive part is the per-job rebuttal call. Un-debated ticks also drop the |dbt hash from
    the scene key, so they share cached rulings with the plain referee arm — identical inputs,
    identical ruling, no extra inference. Requires `debate`."""
    from pins.negotiation_protocol import NegotiationOutcome, negotiate
    assert not (fast_negotiate and extend), "--fast-negotiate replaces the --extend replay path"
    assert not fast_negotiate or theta is not None, "--fast-negotiate needs --theta"

    st = {"free": None, "jids": None, "behind": frozenset(),
          "standing": None,            # (alloc dict, reserve) of the last ruling
          "prev_in": None,             # (demand, supply_ctx, free) of the previous tick
          "h_free": None, "h_jids": None,      # E3 hard-trigger state, kept SEPARATE from the
          "h_behind": frozenset(),             # theta shell's so the two gates can compose
          "h_fell": False,
          "tick": 0}                    # §12: age clock for the quality cache
    qc = None
    if qcache is not None:
        from pins.qcache import QualityCache
        qc = QualityCache(threshold=qcache)
        QSTATS.clear()

    def _delta(demand, free):
        jids = frozenset(j.jid for j in demand)
        behind = frozenset(j.jid for j in demand
                           if j.ctx.get("tier") == "prod" and j.ctx.get("deadline") == "behind")
        if st["free"] is None:
            d = 1.0                                        # cold start always triggers
        else:
            # Pool sizes here are 0-8 GPUs, so a RAW |dfree|/prev term is near-binary (any
            # 1-GPU move fires). Bucket it the way the agents themselves see supply
            # (empty / scarce / roomy): only a bucket CROSSING counts as a supply change.
            bucket = lambda f: 0 if f == 0 else (1 if f <= 2 else 2)
            du = 1.0 if bucket(free) != bucket(st["free"]) else 0.0
            # with `extend` the standing ruling GOVERNS routine arrivals, so only departures
            # count toward the job term; risky arrivals (new prod-behind) still fire via the
            # risk term (0.2 > the 0.15 default theta) — the referee is consulted for risky
            # novelty, never for routine churn.
            moved = (st["jids"] - jids) if extend else (jids ^ st["jids"])
            dj = min(1.0, len(moved) / max(len(st["jids"]), 1))
            risk = 1.0 if behind - st["behind"] else 0.0
            d = 0.4 * du + 0.4 * dj + 0.2 * risk
        st["free"], st["jids"], st["behind"] = free, jids, behind
        return d

    def _gamma(demand, free, waiting) -> float:
        """Elevated plan §8: the SERIOUSNESS score, the continuous generalisation of `_hard`.

        Gamma_t = mean of five normalised risk terms; deliberation fires above threshold theta
        (or on any hard trigger, per the plan). Three terms are read straight off the scene
        (SLA risk, uncertainty, starvation). The plan's D_ambiguity is defined on bid/ask
        closeness, which this arm has no explicit market for — the proxy here is how nearly the
        free pool ties the contested want, which is the same quantity the closeness measures:
        a scene that clears by one GPU is the scene where a prediction error flips the outcome.
        C_churn is the share of contesting jobs that were absent from the last executed
        ruling — job-set turnover, the churn signal this scene actually carries."""
        n = max(len(demand), 1)
        r_sla = sum(1 for j in demand if j.ctx.get("deadline") == "behind") / n
        unc = sum({"high": 1.0, "medium": 0.5}.get(j.ctx.get("uncertainty"), 0.0)
                  for j in demand) / n
        # near-tie proxy for D_ambiguity: every contesting job wants at least one margin GPU,
        # so |free - n| is the clearing slack. A scene that clears by one GPU is exactly where
        # a prediction error flips the award.
        ambiguity = math.exp(-abs(free - n) / 2.0)
        starv = ((sum(1 for w in (waiting or []) if w.get("waited_ticks", 0) >= 10)
                  / max(len(waiting), 1)) if waiting else 0.0)
        prev = st.get("executed") or {}
        churn = (sum(1 for j in demand if j.jid not in prev) / n) if prev else 0.0
        return (r_sla + ambiguity + unc + starv + churn) / 5.0

    def _hard(demand, free) -> bool:
        """E3: does this scene deserve a deliberation round? Fires on the events the plan
        pre-registered as hard triggers; routine arrival/departure churn does not."""
        jids = frozenset(j.jid for j in demand)
        behind = frozenset(j.jid for j in demand if j.ctx.get("deadline") == "behind")
        bucket = lambda f: 0 if f == 0 else (1 if f <= 2 else 2)   # the agents' own supply view
        fire = (st["h_jids"] is None                               # cold start: always deliberate
                or st["h_fell"]                                    # last ruling was infeasible
                or bucket(free) != bucket(st["h_free"])            # contested-capacity crossing
                or bool(behind - st["h_behind"])                   # a job newly behind deadline
                or any(j.jid not in st["h_jids"] and j.ctx.get("tier") == "prod"
                       for j in demand))                           # prod arrival
        st["h_jids"], st["h_free"], st["h_behind"] = jids, free, behind
        return fire

    def policy(demand, supply_ctx, free, waiting=None, **_):
        waiting = waiting if admit else None       # admission is an opt-in arm (+admit tier)
        fast = (theta is not None and st["standing"] is not None
                and _delta(demand, free) <= theta)
        if theta is not None and not fast:
            _delta(demand, free) if st["free"] is None else None   # keep state warm on cold start
        if fast:
            if fast_negotiate:
                no = negotiate(demand, supply_ctx, free, use_llm=use_llm,
                               model=statement_model or model, cache=cache)
                margins, sreserve, rounds, transcript = dict(no.margins), no.reserve, no.rounds, no.transcript
            else:
                alloc, sreserve = st["standing"]
                margins = {j.jid: alloc.get(j.jid, 0) for j in demand}
                if extend:
                    # Award arrivals by the ruling's own revealed policy: per-tier exemplar
                    # award (mean of what the ruling gave this tier), granted in the referee's
                    # rule 3/4 order — prod before besteffort, behind before ontrack. No
                    # exemplar (ruling never awarded this tier) -> 0: extended, not invented.
                    by_tier: dict[str, list[int]] = {}
                    for j in demand:
                        if j.jid in alloc and alloc[j.jid] > 0:
                            by_tier.setdefault(j.ctx.get("tier", "besteffort"), []).append(alloc[j.jid])
                    new = [j for j in demand if j.jid not in alloc]
                    for j in sorted(new, key=lambda j: (j.ctx.get("tier") != "prod",
                                                        j.ctx.get("deadline") != "behind", j.jid)):
                        ex = by_tier.get(j.ctx.get("tier", "besteffort"))
                        margins[j.jid] = round(sum(ex) / len(ex)) if ex else 0
                rounds, transcript = 0, []
            if not check_allocation(margins, sreserve, demand, free):   # still fits live pool
                SHELL_STATS["fast"] += 1
                st["prev_in"] = (demand, supply_ctx, free)
                st["executed"] = dict(margins) | {"_reserve": sreserve}
                fout = NegotiationOutcome(
                    margins=margins, reserve=sreserve, rounds=rounds, agreed=True,
                    transcript=transcript)
                if waiting:
                    # fast ticks admit by the deterministic rule; the referee owns admission
                    # only on the novel/risky scenes it is consulted for — same division of
                    # labour as the margins.
                    from pins.negotiation_protocol import admission_plan
                    fout.priority, fout.defer = admission_plan(waiting, free, reserve=sreserve)
                return margins, sreserve, fout
            # infeasible (standing ruling no longer fits, or negotiate() couldn't clear) ->
            # fall through and re-invoke the referee

        # plan §12: quality-aware similarity reuse. Retrieve -> adapt by category -> RE-VALIDATE
        # (never repair) -> execute. A rejected candidate is counted as a false reuse and the
        # tick falls through to a fresh ruling, so the safety layer is never bypassed.
        if qc is not None:
            prev_exec = {k: v for k, v in (st.get("executed") or {}).items() if k != "_reserve"}
            want = sum(1 for _ in demand)
            z = qc.state(demand, free, want, prev_exec)
            hit = qc.retrieve(z, st["tick"])
            if hit is not None:
                entry, _score = hit
                cand = qc.adapt(entry, demand, free)
                bad = check_allocation(cand, entry.reserve, demand, free)
                live_q = qc.quality(cand, entry.reserve, demand, free, bool(bad), prev_exec)
                qc.note_reuse(entry.q, live_q)
                if not bad:
                    SHELL_STATS["fast"] += 1
                    st["prev_in"] = (demand, supply_ctx, free)
                    st["executed"] = dict(cand) | {"_reserve": entry.reserve}
                    st["tick"] += 1
                    return cand, entry.reserve, NegotiationOutcome(
                        margins=cand, reserve=entry.reserve, rounds=0, agreed=True,
                        transcript=[{"round": 0, "actor": "qcache",
                                     "why": f"reused a ruling of quality {entry.q:.2f}"}])

        dec_demand, dec_supply, dec_free = (
            st["prev_in"] if (stale == "one" and st["prev_in"] is not None)
            else (demand, supply_ctx, free))
        stmts = None
        if debate:
            stmts = gather_statements(dec_demand, dec_supply, use_llm=use_llm,
                                      model=statement_model or model, cache=cache)
            if gamma is not None:            # plan §8: score gate, OR any hard trigger
                fire = _gamma(dec_demand, dec_free, waiting) > gamma or _hard(dec_demand, dec_free)
            else:
                fire = not hard_trigger or _hard(dec_demand, dec_free)
            if fire:
                stmts = rebut(stmts, dec_free, use_llm=use_llm, model=statement_model or model,
                              cache=cache)
                SHELL_STATS["debate"] += 1
        o = referee_decide(dec_demand, dec_supply, dec_free, use_llm=use_llm, model=model,
                           cache=cache, statement_model=statement_model, think=think,
                           stmts=stmts, trigger=trigger, no_argue=no_argue,
                           prev_alloc=(st.get("executed") or {}) if prev_input else None,
                           waiting=waiting)
        SHELL_STATS["llm"] += 1
        margins = {j.jid: o.alloc.get(j.jid, 0) for j in demand}
        if stale == "one":               # stale ruling -> the evaluator re-checks vs LIVE state
            violations = check_allocation(margins, o.reserve, demand, free)
            feasible = not violations
        else:
            violations, feasible = o.violations, o.feasible
        reserve = o.reserve
        st["h_fell"] = not feasible              # a fallback re-arms the deliberation trigger
        if not feasible:                         # overcommit/violation -> floor, counted
            margins = {j.jid: 0 for j in demand}
            reserve = 0
        else:
            st["standing"] = (dict(o.alloc), reserve)
        st["prev_in"] = (demand, supply_ctx, free)
        if qc is not None:               # §12: store only what VALIDATED and executed
            prev_exec = {k: v for k, v in (st.get("executed") or {}).items() if k != "_reserve"}
            if feasible:
                qc.store(qc.state(demand, free, len(demand), prev_exec), margins, reserve,
                         qc.quality(margins, reserve, demand, free, False, prev_exec),
                         st["tick"], demand)
            st["tick"] += 1
            QSTATS.update(qc.stats())
        st["executed"] = dict(margins) | {"_reserve": reserve}   # E1: what actually ran
        out = NegotiationOutcome(margins=margins, reserve=reserve, rounds=1,
                                 agreed=feasible, transcript=o.transcript,
                                 # infeasible ruling -> floor tick: no admission control either
                                 priority=o.priority if feasible else None,
                                 defer=o.defer if feasible else None)
        sig = f"referee|free={free}|ok={feasible}|r={reserve}|m={sorted(margins.items())}"
        if sig not in seen:
            seen.add(sig)
            trace.append({"policy": "referee", "free_gpus": free, "feasible": feasible,
                          "violations": violations, "reserve": reserve,
                          "llm_reserve": o.reserve, "margins": margins,
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
