"""
Bounded two-sided NEGOTIATION protocol (build task #2).

Until now the demand agent (margin/hedge, Exp 16-17, pins/uncertainty_sim.py) and the supply
agent (headroom reserve, Exp 14, pins/supply_sim.py) set their levers **in isolation**, each in
its own simulator. The two-sided thesis (Research/CLAUDE.md §3-5) needs them to INTERACT over the
SAME free GPUs, with deliberately asymmetric objectives:

  * demand pushes the reserve R -> 0 (use all capacity now, get my safety margin);
  * supply pushes R > 0 (hold idle headroom for an incoming, non-preemptable prod job).

This module is the bounded protocol that resolves that tension. It is NOT open-ended chat — the
risks the design must answer (Research/CLAUDE.md §3) are termination and a side caving
sycophantically, so the protocol is a finite ladder of monotone concessions with a mandatory
heuristic fallback. The LLMs (pins/llm_agent.llm_margin / llm_reserve) only pick a categorical
LEVEL + justification; this code owns the GPU arithmetic and the convergence.

Properties (provable, for the paper):
  * TERMINATION — each job's hedge has 3 levels and the reserve has 3 levels; every concession
    strictly lowers one level and levels never rise, so the loop runs at most 3*n_jobs + 3 steps.
  * SAFETY — on non-convergence it returns the fully-conceded state (margins 0, reserve 0): the
    point-forecast demand with no headroom, i.e. the cheap heuristic. "Negotiation can only help
    or be neutral, never stall" (Research/CLAUDE.md §4).

`single_llm_plan` is the Open-Q #5 control: one pins/llm_agent.llm_joint call decides BOTH levers
at once (no negotiation), returning the SAME outcome shape so a simulator swaps them by a flag.

Run:  .venv/bin/python -m pins.negotiation_protocol      # smoke: a contested case + a fallback case
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pins.llm_agent import (RESERVE_LEVELS, llm_joint, llm_margin, llm_reserve,
                            reserve_amount)

# Demand-side hedge ladder, in GPU terms (code owns the magnitude; the LLM only picks the level).
HEDGES = ["none", "some", "heavy"]
HEDGE_GPUS = {"none": 0, "some": 1, "heavy": 2}


@dataclass
class DemandJob:
    """One job on the demand side of the table for this clearing point."""
    jid: str
    ctx: dict                 # the bridged ctx for llm_margin (uncertainty/spike_risk/deadline/…)
    forecast_cap: int         # the job's point-forecast GPU need this round (cap0)
    is_train: bool = True     # margin is only useful in the spike-prone train phase
    concede_rank: float = 0.0  # lower = concedes its margin FIRST (low priority / much slack)


@dataclass
class NegotiationOutcome:
    margins: dict[str, int]    # jid -> agreed margin GPUs (0 for non-train / fully conceded)
    reserve: int               # agreed headroom GPUs held idle for incoming prod
    rounds: int                # concession steps taken (0 = agreed immediately)
    agreed: bool               # True if want <= avail was reached; False -> heuristic fallback
    transcript: list = field(default_factory=list)


def default_margin_gpus(job: DemandJob, hedge: str) -> int:
    """Level -> margin GPUs. Override with an uncertainty-sized version (margin_uncertainty)."""
    return HEDGE_GPUS[hedge] if job.is_train else 0


def negotiate(demand: list[DemandJob], supply_ctx: dict, free_gpus: int,
              margin_gpus=default_margin_gpus, use_llm: bool = False,
              model: str = "qwen2.5:3b", cache: dict | None = None,
              max_rounds: int | None = None) -> NegotiationOutcome:
    """Resolve demand margin vs supply reserve over the CONTESTED SLICE by bounded concession.

    `free_gpus` is the whole free pool; the contested slice is what remains once every job's
    non-negotiable base forecast is met (`free_gpus - sum(forecast_cap)`). The negotiation moves
    only the safety margins and the reserve within that slice.

    Round 0: each job's demand agent picks a hedge (llm_margin); the supply agent picks a reserve
    level (llm_reserve). If the requested margins fit under slice-minus-reserve, done. Otherwise
    concede one level at a time — whichever side's concession closes the want/avail gap more — until
    it fits or both sides are exhausted. Returns the agreed (or fully-conceded) plan + a transcript."""
    cache = {} if cache is None else cache
    n = len(demand)
    cap = max_rounds if max_rounds is not None else 3 * n + 3

    # --- round 0: both sides state their opening position --------------------------------------
    hedge: dict[str, str] = {}
    transcript: list = []
    for j in demand:
        d = llm_margin(j.ctx, use_llm=use_llm, model=model, cache=cache)
        h = d["hedge"] if j.is_train else "none"
        hedge[j.jid] = h
        transcript.append({"round": 0, "actor": "demand", "jid": j.jid, "level": h,
                           "why": d["justification"], "_source": d["_source"]})
    rd = llm_reserve(supply_ctx, use_llm=use_llm, model=model, cache=cache)
    reserve_level = rd["reserve"]
    transcript.append({"round": 0, "actor": "supply", "level": reserve_level,
                       "why": rd["justification"], "_source": rd["_source"]})

    # Only the slice ABOVE base demand is contested. Each job's forecast_cap is its non-negotiable
    # point-forecast need — the auction allocates it regardless — so the negotiation resolves only
    # the demand-side safety MARGINS vs the supply RESERVE over the free GPUs left once base demand
    # is met. (Negotiating over the whole pool made `want` exceed `avail` whenever base demand alone
    # filled the pool, forcing the fallback under any contention; the LLM's levels never survived.)
    base = sum(j.forecast_cap for j in demand)

    def margins() -> dict[str, int]:
        return {j.jid: margin_gpus(j, hedge[j.jid]) for j in demand}

    def want() -> int:
        return sum(margin_gpus(j, hedge[j.jid]) for j in demand)   # contested asks = margins only

    def avail() -> int:
        return free_gpus - base - reserve_amount(reserve_level)    # the contested slice, minus reserve

    # jobs that can still give up margin, ordered by who should concede first (low prio / slack)
    by_concede = sorted(demand, key=lambda j: (j.concede_rank, j.jid))

    rounds = 0
    while want() > avail() and rounds < cap:
        gap = want() - avail()
        # candidate concession 1: supply lowers the reserve one level (frees GPUs -> avail up)
        cur_r = RESERVE_LEVELS.index(reserve_level)
        supply_gain = 0
        if cur_r > 0:
            supply_gain = reserve_amount(reserve_level) - reserve_amount(RESERVE_LEVELS[cur_r - 1])
        # candidate concession 2: the next eligible demand job drops a hedge level (want down)
        demand_job = next((j for j in by_concede
                           if HEDGES.index(hedge[j.jid]) > 0), None)
        demand_gain = 0
        if demand_job is not None:
            cur_h = HEDGES.index(hedge[demand_job.jid])
            demand_gain = (margin_gpus(demand_job, hedge[demand_job.jid])
                           - margin_gpus(demand_job, HEDGES[cur_h - 1]))

        if supply_gain == 0 and demand_gain == 0:
            break                                          # both sides exhausted -> no agreement
        # take the bigger gap-closer; tie -> supply yields first (its headroom is speculative too)
        take_supply = supply_gain >= demand_gain and supply_gain > 0
        rounds += 1
        if take_supply:
            reserve_level = RESERVE_LEVELS[cur_r - 1]
            transcript.append({"round": rounds, "actor": "supply", "action": "concede",
                               "level": reserve_level, "gap": gap,
                               "why": "lower headroom to free GPUs for waiting demand"})
        else:
            cur_h = HEDGES.index(hedge[demand_job.jid])
            hedge[demand_job.jid] = HEDGES[cur_h - 1]
            transcript.append({"round": rounds, "actor": "demand", "jid": demand_job.jid,
                               "action": "concede", "level": hedge[demand_job.jid], "gap": gap,
                               "why": "drop speculative safety margin so others can run"})

    agreed = want() <= avail()
    return NegotiationOutcome(margins=margins(), reserve=reserve_amount(reserve_level),
                              rounds=rounds, agreed=agreed, transcript=transcript)


# --------------------------------------------------------------------------- #
#  Single-LLM baseline (Open-Q #5): one agent decides both levers, no rounds     #
# --------------------------------------------------------------------------- #
def aggregate_joint_ctx(demand: list[DemandJob], supply_ctx: dict) -> dict:
    """Collapse the demand table into the worst-case buckets a single agent would weigh, plus the
    supply picture. Worst-case (max) because the lone agent must protect the most at-risk job."""
    order = {"low": 0, "medium": 1, "high": 2}
    dorder = {"ahead": 0, "ontrack": 1, "behind": 2}
    train = [j for j in demand if j.is_train] or demand
    if not train:                                          # no margin candidates this tick
        return {"uncertainty": "low", "spike_risk": "low", "deadline": "ahead",
                "contention": supply_ctx["contention"], "incoming_prod": supply_ctx["incoming_prod"]}
    unc = max((j.ctx.get("uncertainty", "low") for j in train), key=lambda b: order.get(b, 0))
    spk = max((j.ctx.get("spike_risk", "low") for j in train), key=lambda b: order.get(b, 0))
    dl = max((j.ctx.get("deadline", "ahead") for j in train), key=lambda b: dorder.get(b, 0))
    return {"uncertainty": unc, "spike_risk": spk, "deadline": dl,
            "contention": supply_ctx["contention"], "incoming_prod": supply_ctx["incoming_prod"]}


def single_llm_plan(demand: list[DemandJob], supply_ctx: dict, free_gpus: int,
                    margin_gpus=default_margin_gpus, use_llm: bool = False,
                    model: str = "qwen2.5:3b", cache: dict | None = None) -> NegotiationOutcome:
    """The must-have control: ONE llm_joint call sets a uniform margin level (applied to every
    train job) and the reserve, in a single shot. Same NegotiationOutcome shape as `negotiate`."""
    cache = {} if cache is None else cache
    jctx = aggregate_joint_ctx(demand, supply_ctx)
    d = llm_joint(jctx, use_llm=use_llm, model=model, cache=cache)
    hedge = d["margin"]
    margins = {j.jid: margin_gpus(j, hedge if j.is_train else "none") for j in demand}
    return NegotiationOutcome(
        margins=margins, reserve=reserve_amount(d["reserve"]), rounds=1, agreed=True,
        transcript=[{"round": 0, "actor": "joint", "state": "|".join(jctx.values()),
                     "margin": hedge, "reserve": d["reserve"],
                     "why": d["justification"], "_source": d["_source"]}])


def _print_outcome(title: str, o: NegotiationOutcome) -> None:
    print(f"=== {title} ===")
    print(f"  agreed={o.agreed}  rounds={o.rounds}  reserve={o.reserve}  margins={o.margins}")
    for t in o.transcript:
        actor = t.get("actor")
        if actor == "joint":
            print(f"   [r{t['round']} joint] margin={t['margin']} reserve={t['reserve']} "
                  f"({t['_source']})  {t['why']}")
        elif actor == "demand":
            who = t.get("jid", "?")
            act = t.get("action", "open")
            print(f"   [r{t['round']} demand {who}] {act}->{t['level']}  {t['why']}")
        else:
            act = t.get("action", "open")
            print(f"   [r{t['round']} supply] {act}->{t['level']}  {t['why']}")
    print()


def main() -> None:
    # Contended case: 4 free GPUs, three train jobs each forecasting 2 -> base want 6 > 4, and the
    # supply side wants to reserve for incoming prod. The ladder must concede to fit.
    demand = [
        DemandJob("jA", {"uncertainty": "high", "spike_risk": "high", "deadline": "behind",
                         "contention": "high", "tier": "prod"}, forecast_cap=2, concede_rank=2.0),
        DemandJob("jB", {"uncertainty": "high", "spike_risk": "medium", "deadline": "ontrack",
                         "contention": "high", "tier": "besteffort"}, forecast_cap=2, concede_rank=1.0),
        DemandJob("jC", {"uncertainty": "medium", "spike_risk": "low", "deadline": "ahead",
                         "contention": "high", "tier": "besteffort"}, forecast_cap=2, concede_rank=0.0),
    ]
    supply = {"contention": "moderate", "incoming_prod": "few"}
    # Resolvable conflict: base forecast = 6, free = 8 -> contested slice = 2. jA opens a heavy (+2)
    # margin and supply opens a light (+1) reserve -> want 2 > avail (2-1)=1. The ladder lets supply
    # yield its speculative headroom so the at-risk prod job KEEPS its margin (agreed, margin survives).
    _print_outcome("negotiate (resolvable, free=8, slice=2)", negotiate(demand, supply, free_gpus=8))
    _print_outcome("single-LLM baseline (same scene)", single_llm_plan(demand, supply, free_gpus=8))

    # Fallback case: free = 6 = base forecast exactly -> contested slice = 0; no margin can fit even
    # fully conceded, so it returns the safe heuristic (margins 0, reserve 0) and the auction simply
    # rations the base demand. (Under the old whole-pool rule this fired under almost any contention.)
    _print_outcome("negotiate (no slice, free=6 -> fallback)",
                   negotiate(demand, supply, free_gpus=6))


if __name__ == "__main__":
    main()
