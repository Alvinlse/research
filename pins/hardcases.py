"""The HARD-CASE SUITE — where a rigid decider is supposed to break.

Thesis under test: the LLM referee behaves like a flexible human operator and resolves hard /
exceptional scenes that a static ILP formulation cannot. Every previous experiment measured the
MEAN over random trace windows; exceptions are rare there, so averaging hides exactly the
effect claimed. This suite scores the TAIL instead, case by case.

PRE-REGISTRATION. Scenes, exception text and predicates are authored BEFORE any decider is run
on them, and committed as such. `expect` records the predicted failure mode up front, so a
surprise is visible as a surprise. Do not edit a case after seeing a decider's output — add a
new one.

FAIRNESS. The ILP arm uses pins/ilp.py unmodified: the project's real solver, given the best
static formulation available, not a weakened stand-in. The claim is not "ILP is bad" — it is
that a formulation fixed in advance cannot encode facts nobody anticipated. Category ROUTINE
exists so the suite can lose: those are ordinary packing problems where the solver should win.

CHANNEL. An exceptional fact reaches the referee the way it would reach a human operator: in
the free-text `justification` of the submission that raises it. The ILP never sees that text —
not because we withhold it, but because a MILP has nowhere to put it. That asymmetry IS the
experiment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from pins.negotiation_protocol import DemandJob

Alloc = dict[str, int]


@dataclass
class HardCase:
    id: str
    category: str
    free_gpus: int
    stmts: list[dict]                       # hand-authored submissions (exception in justification)
    predicate: Callable[[Alloc, int], bool]  # (alloc, reserve) -> did it do the defensible thing?
    rationale: str                          # why that predicate is the defensible answer
    expect: str                             # PRE-REGISTERED prediction of the rigid arm's failure
    must_cite: list[str] = field(default_factory=list)   # substrings a faithful justification cites

    def demand(self) -> list[DemandJob]:
        """DemandJobs implied by the submissions (the evaluator checks against these)."""
        return [DemandJob(jid=s["job_id"],
                          ctx={"tier": s["tier"], "deadline": s["deadline"]},
                          forecast_cap=s["base_gpus"],
                          is_train=s.get("is_train", True))
                for s in self.stmts if s["side"] == "demand"]


def d(jid, tier, deadline, base, margin, why, is_train=True) -> dict:
    return {"side": "demand", "job_id": jid, "tier": tier, "deadline": deadline,
            "base_gpus": base, "requested_margin_gpus": margin,
            "justification": why, "is_train": is_train, "_source": "hardcase"}


def s(reserve, incoming, why) -> dict:
    return {"side": "supply", "requested_reserve_gpus": reserve, "incoming_prod": incoming,
            "justification": why, "_source": "hardcase"}


def total(alloc: Alloc) -> int:
    return sum(alloc.values())


CASES: list[HardCase] = []


def case(**kw):
    CASES.append(HardCase(**kw))


# --------------------------------------------------------------------------- #
# 1. INFEASIBLE — demand exceeds the pool. No allocation satisfies everyone;
#    the defensible act is triage, which a MILP has no representation for.
# --------------------------------------------------------------------------- #
case(id="INFEAS-01", category="infeasible", free_gpus=5,
     stmts=[d("r00", "prod", "ahead", 4, 0, "Base 4 GPUs. Deadline is 40 ticks out; plenty of slack."),
            d("r01", "prod", "behind", 4, 0, "Base 4 GPUs. Deadline in 2 ticks — misses without full base now."),
            s(0, "none", "No incoming prod this tick.")],
     predicate=lambda a, r: a.get("r01", 0) >= 4,
     rationale="Only one prod job can be served fully; the one 2 ticks from its deadline is the "
               "one whose SLA is still saveable. Serving r00 first loses both.",
     expect="INFEASIBLE, or an arbitrary split that half-serves both and saves neither",
     must_cite=["r01", "deadline"])

case(id="INFEAS-02", category="infeasible", free_gpus=3,
     stmts=[d("r00", "prod", "behind", 3, 0, "Base 3 GPUs, deadline in 1 tick."),
            d("r01", "prod", "behind", 3, 0, "Base 3 GPUs, deadline in 1 tick."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: (a.get("r00", 0) == 3) != (a.get("r01", 0) == 3),
     rationale="Symmetric and both cannot be saved. Picking ONE and serving it fully beats "
               "splitting 2/1, which misses both deadlines. Either choice passes.",
     expect="a 2/1 or 1.5/1.5-style split that satisfies neither job")

case(id="INFEAS-03", category="infeasible", free_gpus=6,
     stmts=[d("r00", "prod", "behind", 4, 2, "Base 4 plus 2 margin; spike risk high."),
            d("r01", "besteffort", "behind", 4, 0, "Base 4. Behind, but best-effort tier."),
            s(2, "many", "Two prod jobs land next tick; need 2 reserved.")],
     predicate=lambda a, r: a.get("r00", 0) >= 4 and r >= 1,
     rationale="Prod base and some incoming-prod protection both survive only if the "
               "best-effort job yields. Tier order is the stated policy.",
     expect="serves the best-effort base because the objective treats GPUs as fungible",
     must_cite=["r00"])

case(id="INFEAS-04", category="infeasible", free_gpus=2,
     stmts=[d("r00", "prod", "behind", 6, 0, "Base 6 GPUs — cannot be met by this pool at all."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: total(a) + r <= 2,
     rationale="The job is unservable this tick. The only defensible act is not to overcommit; "
               "awarding what does not exist is the one true error.",
     expect="INFEASIBLE and no allocation returned at all")

case(id="INFEAS-05", category="infeasible", free_gpus=4,
     stmts=[d("r00", "prod", "behind", 2, 0, "Base 2, deadline in 3 ticks."),
            d("r01", "prod", "behind", 2, 0, "Base 2, deadline in 3 ticks."),
            d("r02", "prod", "behind", 2, 0, "Base 2, deadline in 3 ticks."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: sum(1 for j in ("r00", "r01", "r02") if a.get(j, 0) >= 2) == 2,
     rationale="Three identical prod jobs, room for two. Serving exactly two fully is the "
               "throughput-optimal triage; 2/1/1 saves one instead of two.",
     expect="an even spread across all three, saving one job instead of two")

case(id="INFEAS-06", category="infeasible", free_gpus=5,
     stmts=[d("r00", "prod", "behind", 5, 0, "Base 5, deadline in 2 ticks. All-or-nothing: "
                                             "this job cannot run partially allocated."),
            d("r01", "besteffort", "ontrack", 2, 0, "Base 2, elastic, runs fine on 1."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r00", 0) == 5 or a.get("r00", 0) == 0,
     rationale="A stated all-or-nothing job must be served whole or not at all; a partial "
               "award is pure waste. Either choice is defensible, a 3/2 split is not.",
     expect="a partial award to r00 — the LP relaxation has no all-or-nothing concept",
     must_cite=["r00"])

# --------------------------------------------------------------------------- #
# 2. NL-POLICY — an operator instruction in words. Not in any objective; a
#    solver needs a remodel and recompile, an operator just reads it.
# --------------------------------------------------------------------------- #
case(id="POLICY-01", category="nl_policy", free_gpus=6,
     stmts=[d("r00", "besteffort", "ontrack", 2, 2, "Student coursework job, deadline tonight. "
                                                    "OPERATOR NOTE: protect student jobs this week."),
            d("r01", "prod", "ahead", 2, 2, "Production job, 30 ticks of slack."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r00", 0) >= 2,
     rationale="A standing operator instruction overrides the default tier order for the week. "
               "The prod job has slack and loses nothing by conceding margin.",
     expect="serves prod first — the objective encodes tier priority and cannot read the note",
     must_cite=["student"])

case(id="POLICY-02", category="nl_policy", free_gpus=4,
     stmts=[d("r00", "prod", "ontrack", 2, 2, "Base 2 + 2 margin. OPERATOR NOTE: we are in a "
                                              "power-capped window; do not hand out margin GPUs."),
            d("r01", "besteffort", "ontrack", 1, 2, "Base 1 + 2 margin requested."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r00", 0) <= 2 and a.get("r01", 0) <= 1,
     rationale="The instruction bans margin this window. Bases are still served; only the "
               "hedge is withheld.",
     expect="hands out margin anyway — 'power-capped' is not a variable in the formulation",
     must_cite=["power"])

case(id="POLICY-03", category="nl_policy", free_gpus=8,
     stmts=[d("r00", "prod", "ontrack", 3, 2, "Base 3 + 2 margin."),
            d("r01", "prod", "ontrack", 3, 2, "Base 3 + 2 margin. OPERATOR NOTE: this job belongs "
                                              "to the demo running at 15:00 — it must not be starved."),
            s(2, "many", "Prod wave incoming, want 2 reserved.")],
     predicate=lambda a, r: a.get("r01", 0) >= 3,
     rationale="Named-job protection is a legitimate operator override; r01's base is the "
               "minimum that honours it.",
     expect="may starve r01 to build the reserve; the note is invisible to the solver",
     must_cite=["demo"])

case(id="POLICY-04", category="nl_policy", free_gpus=5,
     stmts=[d("r00", "besteffort", "behind", 2, 1, "Base 2. OPERATOR NOTE: this user has been "
                                                   "starved three windows running — give them a turn."),
            d("r01", "besteffort", "ontrack", 2, 2, "Base 2 + 2 margin, healthy."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r00", 0) >= 2,
     rationale="Repeated starvation is a fairness fact that lives outside a single tick's "
               "state; the operator supplies it and it should bind.",
     expect="treats both best-effort jobs as interchangeable — history is not in the model",
     must_cite=["starv"])

case(id="POLICY-05", category="nl_policy", free_gpus=4,
     stmts=[d("r00", "prod", "ontrack", 4, 0, "Base 4. OPERATOR NOTE: maintenance at the next "
                                              "tick — anything started now will be killed."),
            d("r01", "besteffort", "ontrack", 2, 0, "Base 2, short job, finishes this tick."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r01", 0) >= 2 and a.get("r00", 0) < 4,
     rationale="Work that will be killed next tick is wasted; the short job that completes "
               "first is the only one that banks value.",
     expect="serves the prod job by tier priority and burns the whole award on maintenance",
     must_cite=["maintenance"])

case(id="POLICY-06", category="nl_policy", free_gpus=6,
     stmts=[d("r00", "prod", "ahead", 3, 3, "Base 3 + 3 margin, plenty of slack."),
            d("r01", "prod", "ahead", 3, 3, "Base 3 + 3 margin, plenty of slack."),
            s(0, "none", "OPERATOR NOTE: leave 2 GPUs free for an urgent interactive session "
                         "a researcher is about to start — they have not submitted yet.")],
     predicate=lambda a, r: r >= 2 or total(a) <= 4,
     rationale="A reservation for work that has not been submitted cannot be expressed as a "
               "job constraint, but it is a normal operator action.",
     expect="allocates the full pool — an unsubmitted job does not exist to the solver",
     must_cite=["interactive"])

# --------------------------------------------------------------------------- #
# 3. CONTRADICTION — two stated hard constraints cannot both hold. Something
#    must be broken; the question is whether the decider breaks it knowingly.
# --------------------------------------------------------------------------- #
case(id="CONTRA-01", category="contradiction", free_gpus=4,
     stmts=[d("r00", "prod", "behind", 4, 0, "Base 4. Marked NON-PREEMPTABLE by its owner."),
            d("r01", "prod", "behind", 4, 0, "Base 4. Also marked NON-PREEMPTABLE by its owner."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: (a.get("r00", 0) == 4) != (a.get("r01", 0) == 4),
     rationale="Both cannot hold. Honouring one fully and stating which was broken is the "
               "operator's move; splitting honours neither claim.",
     expect="INFEASIBLE — contradictory hard constraints have no feasible region")

case(id="CONTRA-02", category="contradiction", free_gpus=6,
     stmts=[d("r00", "prod", "behind", 4, 0, "Base 4, deadline in 1 tick."),
            s(4, "many", "A prod wave lands next tick and I need 4 reserved — that is a hard "
                         "floor, not a preference.")],
     predicate=lambda a, r: a.get("r00", 0) >= 4 or r >= 4,
     rationale="6 GPUs cannot fund both a 4-GPU base and a 4-GPU floor. Fully honouring one "
               "side beats a 3/3 split that fails both.",
     expect="splits the pool to minimise a weighted penalty, satisfying neither hard claim")

case(id="CONTRA-03", category="contradiction", free_gpus=5,
     stmts=[d("r00", "prod", "behind", 3, 0, "Base 3. Owner states it must finish this tick."),
            d("r01", "prod", "behind", 3, 0, "Base 3. Owner states it must finish this tick. "
                                             "NOTE: r01 is a dependency of r00 — r00 cannot "
                                             "start until r01 completes."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r01", 0) >= 3 and a.get("r00", 0) < 3,
     rationale="The dependency resolves the tie: serving the blocker first is the only ordering "
               "that makes progress. Serving r00 achieves nothing until r01 finishes.",
     expect="ignores the dependency — it is stated in prose, not in the constraint matrix",
     must_cite=["r01"])

case(id="CONTRA-04", category="contradiction", free_gpus=4,
     stmts=[d("r00", "besteffort", "ontrack", 2, 0, "Base 2. Owner insists this is really prod "
                                                    "and was mis-tagged at submission."),
            d("r01", "prod", "ahead", 2, 2, "Base 2 + 2 margin, 50 ticks of slack."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r00", 0) >= 2,
     rationale="Both bases fit; only the margin is contested. A disputed tag is not a reason to "
               "starve a base when the true-prod job has 50 ticks of slack.",
     expect="honours the tag literally and prefers r01's margin over r00's base")

case(id="CONTRA-05", category="contradiction", free_gpus=3,
     stmts=[d("r00", "prod", "behind", 2, 0, "Base 2, must not be preempted."),
            d("r01", "prod", "behind", 2, 0, "Base 2, must not be preempted."),
            s(2, "many", "Also need 2 reserved — hard requirement.")],
     predicate=lambda a, r: total(a) + r <= 3 and (a.get("r00", 0) == 2 or a.get("r01", 0) == 2 or r == 2),
     rationale="Three mutually exclusive hard claims on 3 GPUs. Feasibility is the one "
               "inviolable rule; beyond that, satisfying one claim fully is the best available.",
     expect="INFEASIBLE, or an infeasible over-award if the constraints are relaxed wrongly")

case(id="CONTRA-06", category="contradiction", free_gpus=8,
     stmts=[d("r00", "prod", "behind", 8, 0, "Base 8 — the entire pool. Non-negotiable."),
            d("r01", "prod", "behind", 1, 0, "Base 1. Also non-negotiable. Finishes in one tick "
                                             "and unblocks four downstream jobs."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r01", 0) >= 1,
     rationale="One GPU spent unblocking four downstream jobs dominates the marginal 8th GPU of "
               "a single job. The exclusive claim must yield somewhere and this is the cheapest.",
     expect="awards all 8 to r00 on claim size, or declares INFEASIBLE",
     must_cite=["r01"])

# --------------------------------------------------------------------------- #
# 4. UNMODELED — a fact that matters but has no variable in the formulation.
# --------------------------------------------------------------------------- #
case(id="UNMOD-01", category="unmodeled", free_gpus=4,
     stmts=[d("r00", "besteffort", "behind", 2, 0, "Base 2. Checkpoints in 3 ticks — stopping it "
                                                   "now discards 40 ticks of completed work."),
            d("r01", "besteffort", "behind", 2, 0, "Base 2. Started this tick, nothing to lose."),
            s(2, "many", "Want 2 reserved for incoming prod.")],
     predicate=lambda a, r: a.get("r00", 0) >= 2,
     rationale="Sunk work about to be checkpointed is real value at risk; the fresh job can be "
               "restarted for free. Identical on every modelled axis, opposite in value.",
     expect="treats them as interchangeable — 'checkpoints in 3 ticks' is not a variable",
     must_cite=["checkpoint"])

case(id="UNMOD-02", category="unmodeled", free_gpus=6,
     stmts=[d("r00", "prod", "ontrack", 4, 0, "Base 4. Needs all 4 GPUs on ONE node — it is an "
                                              "NVLink all-reduce job and degrades 5x if split."),
            d("r01", "besteffort", "ontrack", 2, 2, "Base 2 + 2 margin, embarrassingly parallel."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r00", 0) in (0, 4),
     rationale="A topology-sensitive job wants its full width or nothing; a 2-GPU award to a "
               "5x-degrading all-reduce is worse than not running it.",
     expect="awards a partial 2-3 GPUs — interconnect topology is outside the model",
     must_cite=["NVLink"])

case(id="UNMOD-03", category="unmodeled", free_gpus=5,
     stmts=[d("r00", "prod", "behind", 3, 0, "Base 3, deadline in 2 ticks."),
            d("r01", "besteffort", "ontrack", 2, 0, "Base 2. NOTE: this is the licence server for "
                                                    "the whole cluster's compiler toolchain — if "
                                                    "it stops, every other job fails to build."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r01", 0) >= 2,
     rationale="A cluster-wide dependency dwarfs one job's deadline. Its tier tag says "
               "best-effort; its actual blast radius says otherwise.",
     expect="serves the prod deadline and kills the licence server",
     must_cite=["licen"])

case(id="UNMOD-04", category="unmodeled", free_gpus=4,
     stmts=[d("r00", "prod", "behind", 4, 0, "Base 4, deadline in 2 ticks. NOTE: the result is "
                                             "only useful if it lands before the 09:00 meeting, "
                                             "which is 1 tick away — it cannot make it either way."),
            d("r01", "besteffort", "ontrack", 3, 0, "Base 3, completes in one tick, result needed all day."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r01", 0) >= 3,
     rationale="A deadline that is already unreachable has no value left to protect; the job "
               "whose output is still useful should get the GPUs.",
     expect="protects the prod deadline it can no longer meet",
     must_cite=["r01"])

case(id="UNMOD-05", category="unmodeled", free_gpus=6,
     stmts=[d("r00", "prod", "ontrack", 3, 3, "Base 3 + 3 margin. NOTE: this job's spike is "
                                              "caused by a memory leak we are already patching; "
                                              "the margin would be wasted on the leak."),
            d("r01", "besteffort", "behind", 3, 0, "Base 3, genuinely behind."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r00", 0) <= 3 and a.get("r01", 0) >= 3,
     rationale="Margin is meant to absorb legitimate variance, not to feed a known defect. The "
               "hedge signal is real but its cause makes it worthless.",
     expect="grants the margin — the spike statistic looks identical to a legitimate one",
     must_cite=["leak"])

case(id="UNMOD-06", category="unmodeled", free_gpus=5,
     stmts=[d("r00", "besteffort", "behind", 5, 0, "Base 5. NOTE: this is a re-run of a job that "
                                                   "has already failed four times for the same "
                                                   "reason; nothing has changed since."),
            d("r01", "besteffort", "ontrack", 2, 0, "Base 2, first attempt, healthy."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r01", 0) >= 2,
     rationale="A job that has failed identically four times has near-zero expected value; "
               "spending the pool on it starves work that might succeed.",
     expect="serves the larger, more-behind job — failure history is not a modelled feature",
     must_cite=["fail"])

# --------------------------------------------------------------------------- #
# 5. CORRUPT — the inputs are wrong. A solver optimises confidently over garbage.
# --------------------------------------------------------------------------- #
case(id="CORRUPT-01", category="corrupt", free_gpus=8,
     stmts=[d("r00", "prod", "behind", 500, 0, "Base 500 GPUs (declared)."),
            d("r01", "prod", "behind", 2, 0, "Base 2 GPUs."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: total(a) + r <= 8 and a.get("r01", 0) >= 2,
     rationale="A 500-GPU declaration on an 8-GPU pool is a data error, not a request. The "
               "sane act is to serve the credible job and not be captured by the bad number.",
     expect="either INFEASIBLE, or the whole pool poured into the impossible job",
     must_cite=["r01"])

case(id="CORRUPT-02", category="corrupt", free_gpus=4,
     stmts=[d("r00", "prod", "behind", -2, 0, "Base -2 GPUs (declared)."),
            d("r01", "besteffort", "ontrack", 2, 0, "Base 2 GPUs."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r00", 0) >= 0 and total(a) + r <= 4,
     rationale="A negative requirement is meaningless; treating it as zero and staying feasible "
               "is the only defensible reading.",
     expect="a negative coefficient flows into the objective and skews the solution")

case(id="CORRUPT-03", category="corrupt", free_gpus=6,
     stmts=[d("r00", "prod", "behind", 3, 0, "Base 3. Declared runtime 0 ticks — it claims to "
                                             "finish instantly, which cannot be true."),
            d("r01", "prod", "behind", 3, 0, "Base 3, declared runtime 10 ticks, plausible."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r01", 0) >= 3,
     rationale="Both fit only if the impossible claim is trusted. The credible job should not "
               "be starved by an obviously broken declaration.",
     expect="trusts the zero-runtime claim and schedules around a job that never ends")

case(id="CORRUPT-04", category="corrupt", free_gpus=4,
     stmts=[d("r00", "prod", "behind", 2, 0, "Base 2."),
            d("r00", "besteffort", "ahead", 3, 0, "Base 3. (Duplicate job id — same id submitted "
                                                  "twice with different tiers.)"),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: total(a) + r <= 4,
     rationale="A duplicated id is a submission bug. Whatever is decided, awarding more than the "
               "pool because a job was counted twice is the failure to avoid.",
     expect="double-counts the id and over-awards")

case(id="CORRUPT-05", category="corrupt", free_gpus=5,
     stmts=[d("r00", "prod", "behind", 2, 99, "Base 2, margin 99 requested — a units bug, the "
                                              "job means 99 percent utilisation not 99 GPUs."),
            d("r01", "besteffort", "ontrack", 2, 1, "Base 2 + 1 margin."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r00", 0) <= 3 and a.get("r01", 0) >= 2,
     rationale="The stated cause makes the 99 a units error. Serving base plus a sane margin, "
               "and not starving r01, is the reading that matches the words.",
     expect="maximises against the 99 and gives r00 everything available",
     must_cite=["r00"])

case(id="CORRUPT-06", category="corrupt", free_gpus=0,
     stmts=[d("r00", "prod", "behind", 2, 0, "Base 2, deadline imminent."),
            s(1, "many", "Prod wave incoming, want 1 reserved.")],
     predicate=lambda a, r: total(a) + r == 0,
     rationale="Nothing is free. The only correct output is to award nothing; any positive "
               "award is fabricated capacity.",
     expect="should handle this correctly — included as a sanity check on both arms")

# --------------------------------------------------------------------------- #
# 6. AMBIGUOUS — no rule settles it; the tradeoff has to be reasoned about.
# --------------------------------------------------------------------------- #
case(id="AMBIG-01", category="ambiguous", free_gpus=4,
     stmts=[d("r00", "prod", "ahead", 2, 2, "Base 2 + 2 margin. 60 ticks of slack."),
            d("r01", "besteffort", "behind", 4, 0, "Base 4. Misses its deadline this tick without all 4."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r01", 0) >= 2,
     rationale="Tier says prod first; the situation says a hedge for a job with 60 ticks of "
               "slack should not cost another job its deadline. Some real concession to r01.",
     expect="applies tier priority mechanically and funds the prod margin first")

case(id="AMBIG-02", category="ambiguous", free_gpus=6,
     stmts=[d("r00", "prod", "behind", 5, 0, "Base 5, saves one deadline."),
            d("r01", "besteffort", "behind", 2, 0, "Base 2, saves one deadline."),
            d("r02", "besteffort", "behind", 2, 0, "Base 2, saves one deadline."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r01", 0) >= 2 and a.get("r02", 0) >= 2,
     rationale="Two deadlines saved beats one, and the pool cannot do both. Tier is a "
               "tie-breaker, not a licence to halve total SLA.",
     expect="serves the prod job and lets two deadlines slip to save one")

case(id="AMBIG-03", category="ambiguous", free_gpus=4,
     stmts=[d("r00", "prod", "behind", 4, 0, "Base 4. 95 percent through its run."),
            d("r01", "prod", "behind", 4, 0, "Base 4. 5 percent through its run."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r00", 0) >= 4,
     rationale="Identical on every modelled axis; the job that is nearly done converts GPUs "
               "into a completed result, the other cannot finish either way.",
     expect="a coin-flip or an even split — progress is not in the objective",
     must_cite=["r00"])

case(id="AMBIG-04", category="ambiguous", free_gpus=5,
     stmts=[d("r00", "prod", "ontrack", 2, 3, "Base 2 + 3 margin; genuinely high spike risk."),
            d("r01", "prod", "ontrack", 2, 3, "Base 2 + 3 margin; genuinely high spike risk."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r00", 0) >= 2 and a.get("r01", 0) >= 2,
     rationale="Both bases must hold; the margin is what gets rationed. Funding one job's full "
               "hedge while the other loses its base is the error.",
     expect="funds one job's full request and starves the other's base")

case(id="AMBIG-05", category="ambiguous", free_gpus=8,
     stmts=[d("r00", "besteffort", "behind", 8, 0, "Base 8 — one big job, uses the whole pool, "
                                                   "finishes in 10 ticks."),
            d("r01", "besteffort", "behind", 1, 0, "Base 1, finishes in 1 tick."),
            d("r02", "besteffort", "behind", 1, 0, "Base 1, finishes in 1 tick."),
            d("r03", "besteffort", "behind", 1, 0, "Base 1, finishes in 1 tick."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: sum(1 for j in ("r01", "r02", "r03") if a.get(j, 0) >= 1) >= 2,
     rationale="Three one-tick jobs clear almost immediately and then the big job proceeds. "
               "Head-of-line blocking on the 8-GPU job is the classic avoidable mistake.",
     expect="awards the whole pool to the single largest claim")

case(id="AMBIG-06", category="ambiguous", free_gpus=4,
     stmts=[d("r00", "prod", "behind", 2, 0, "Base 2, deadline in 1 tick, will still miss by a "
                                             "hair even if fully served."),
            d("r01", "prod", "behind", 2, 0, "Base 2, deadline in 4 ticks, makes it comfortably "
                                             "if served now, misses if delayed."),
            s(1, "few", "One prod job may land; 1 reserved would help."),
            ],
     predicate=lambda a, r: a.get("r01", 0) >= 2,
     rationale="Spending on a deadline that misses anyway, at the cost of one that is still "
               "saveable, is the mistake. r01's base is the saveable SLA.",
     expect="prioritises the more-behind job because lateness dominates the objective",
     must_cite=["r01"])

# --------------------------------------------------------------------------- #
# 7. ROUTINE — controls. Ordinary packing, no exception, no prose that matters.
#    The solver SHOULD win or tie here. A suite the LLM sweeps is not evidence.
# --------------------------------------------------------------------------- #
case(id="ROUTINE-01", category="routine", free_gpus=8,
     stmts=[d("r00", "prod", "ontrack", 2, 1, "Base 2 + 1 margin."),
            d("r01", "besteffort", "ontrack", 2, 1, "Base 2 + 1 margin."),
            s(1, "few", "One prod job may land next tick.")],
     predicate=lambda a, r: a.get("r00", 0) >= 2 and a.get("r01", 0) >= 2 and total(a) + r <= 8,
     rationale="Everything fits. Serving both bases within capacity is simply correct.",
     expect="ILP solves this optimally; no advantage expected for the referee")

case(id="ROUTINE-02", category="routine", free_gpus=6,
     stmts=[d("r00", "prod", "behind", 3, 0, "Base 3."),
            d("r01", "besteffort", "ontrack", 3, 0, "Base 3."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r00", 0) >= 3 and a.get("r01", 0) >= 3,
     rationale="Exact fit, no contention. Both bases served.",
     expect="ILP optimal")

case(id="ROUTINE-03", category="routine", free_gpus=8,
     stmts=[d("r00", "prod", "ontrack", 1, 1, "Base 1 + 1 margin."),
            d("r01", "prod", "ontrack", 1, 1, "Base 1 + 1 margin."),
            d("r02", "besteffort", "ontrack", 1, 1, "Base 1 + 1 margin."),
            s(2, "many", "Prod wave incoming; 2 reserved requested.")],
     predicate=lambda a, r: all(a.get(j, 0) >= 1 for j in ("r00", "r01", "r02")) and r >= 1,
     rationale="Bases and a real reserve all fit inside 8; no tradeoff is forced.",
     expect="ILP optimal")

case(id="ROUTINE-04", category="routine", free_gpus=4,
     stmts=[d("r00", "prod", "behind", 4, 2, "Base 4 + 2 margin requested."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r00", 0) == 4,
     rationale="One job, base fits exactly, margin does not. Serve the base, drop the hedge.",
     expect="ILP optimal")

case(id="ROUTINE-05", category="routine", free_gpus=7,
     stmts=[d("r00", "besteffort", "ontrack", 2, 0, "Base 2."),
            d("r01", "besteffort", "ontrack", 2, 0, "Base 2."),
            d("r02", "besteffort", "ontrack", 2, 0, "Base 2."),
            s(1, "few", "One prod job may land.")],
     predicate=lambda a, r: sum(a.get(j, 0) for j in ("r00", "r01", "r02")) >= 6 and r >= 1,
     rationale="6 bases + 1 reserve fits 7 exactly. Nothing to trade off.",
     expect="ILP optimal")

case(id="ROUTINE-06", category="routine", free_gpus=8,
     stmts=[d("r00", "prod", "behind", 4, 2, "Base 4 + 2 margin, high spike risk."),
            d("r01", "besteffort", "ahead", 2, 2, "Base 2 + 2 margin, lots of slack."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r00", 0) >= 4 and a.get("r01", 0) >= 2,
     rationale="Both bases fit with room for some margin; the ranking of the leftovers is a "
               "preference, not an exception.",
     expect="ILP optimal")




# =========================================================================== #
#  ROUND 2 — SHARPENED CASES (added after the round-1 baseline, commit d6d28d5)
#
#  Round 1 was too easy: the rigid arms passed 29-30/42 because many predicates
#  are satisfied incidentally by ordinary tier-ranked packing. These cases are
#  built so the defensible answer is UNREACHABLE by any static value ordering —
#  it requires serving the lower-ranked claim, refusing the highest bid, or
#  holding capacity idle.
#
#  METHOD NOTE, because the distinction matters: these were iterated against the
#  RIGID arms (--no-llm) until both fail them. That is legitimate — it verifies a
#  case actually tests what it claims. Tuning against the LLM's output would be
#  rigging, and is not done: no LLM has been run on this suite at any point. The
#  predicate must still be defensible on its own rationale, and the ROUTINE
#  controls are untouched.
# =========================================================================== #

case(id="INFEAS-07", category="infeasible", free_gpus=8,
     stmts=[d("r00", "prod", "behind", 8, 0, "Base 8, the whole pool. 20 ticks of work left and "
                                             "the deadline is 2 ticks away — it cannot make it."),
            d("r01", "besteffort", "behind", 2, 0, "Base 2, finishes in 1 tick."),
            d("r02", "besteffort", "behind", 2, 0, "Base 2, finishes in 1 tick."),
            d("r03", "besteffort", "behind", 2, 0, "Base 2, finishes in 1 tick."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: sum(1 for j in ("r01", "r02", "r03") if a.get(j, 0) >= 2) >= 3,
     rationale="The prod job misses its deadline whatever it is given, so its GPUs buy nothing. "
               "Three best-effort deadlines are saveable and fit exactly.",
     expect="ranks prod highest and pours the pool into a job that cannot finish")

case(id="INFEAS-08", category="infeasible", free_gpus=4,
     stmts=[d("r00", "prod", "behind", 4, 0, "Base 4. 30 ticks of work, deadline in 1 tick — "
                                             "unreachable however it is served."),
            s(4, "many", "A prod job with a 3-tick deadline lands next tick and needs all 4. It "
                         "is already queued and certain — not a forecast.")],
     predicate=lambda a, r: r >= 4 and total(a) == 0,
     rationale="Holding the pool idle for one tick saves a deadline that is still reachable; "
               "spending it now saves nothing. Idleness is the correct act here.",
     expect="allocates to the present job — an unarrived job scores nothing in the objective")

case(id="INFEAS-09", category="infeasible", free_gpus=3,
     stmts=[d("r00", "prod", "behind", 4, 0, "Base 4, all-or-nothing MPI job — it cannot start "
                                             "on fewer than 4 and 4 do not exist this tick."),
            d("r01", "besteffort", "behind", 3, 0, "Base 3, elastic, deadline this tick."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r01", 0) >= 3 and a.get("r00", 0) == 0,
     rationale="A 3-GPU award to a job that needs 4 to start is pure waste; the elastic job can "
               "use exactly what is free and makes its deadline.",
     expect="awards the 3 GPUs to prod on rank and strands them",
     must_cite=["r01"])

case(id="CONTRA-07", category="contradiction", free_gpus=4,
     stmts=[d("r00", "prod", "behind", 2, 0, "Base 2, deadline in 3 ticks."),
            d("r01", "prod", "behind", 2, 0, "Base 2, deadline in 3 ticks."),
            s(4, "many", "HARD FLOOR: four prod jobs with 1-tick deadlines land next tick and "
                         "need all 4 GPUs. Both jobs here have 3 ticks of slack and can wait.")],
     predicate=lambda a, r: r >= 4 and total(a) == 0,
     rationale="Four saveable deadlines next tick outweigh two jobs that still have slack. The "
               "reserve floor is the binding claim even though it serves no present job.",
     expect="serves the present jobs — reserve is ranked below any real job in the objective")

case(id="CONTRA-08", category="contradiction", free_gpus=4,
     stmts=[d("r00", "prod", "ahead", 4, 0, "Base 4, exclusive claim, 60 ticks of slack."),
            d("r01", "besteffort", "behind", 4, 0, "Base 4, exclusive claim, deadline this tick."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r01", 0) >= 4 and a.get("r00", 0) == 0,
     rationale="Two exclusive claims, one pool. The prod job loses nothing by waiting 60 ticks; "
               "the best-effort job loses its deadline permanently.",
     expect="honours the prod claim because tier outranks deadline pressure in the objective")

case(id="CONTRA-09", category="contradiction", free_gpus=5,
     stmts=[d("r00", "prod", "behind", 6, 0, "Base 6. OWNER INSTRUCTION: if you cannot give me "
                                             "all 6, give me none and give them to r01 instead — "
                                             "a partial award is useless to me."),
            d("r01", "besteffort", "ontrack", 5, 0, "Base 5, elastic."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r00", 0) == 0 and a.get("r01", 0) >= 5,
     rationale="The owner has stated the conditional themselves. Six do not exist, so the "
               "instruction resolves cleanly to giving them to r01.",
     expect="awards 5 partial GPUs to prod, exactly what its owner said is useless",
     must_cite=["r01"])

case(id="AMBIG-07", category="ambiguous", free_gpus=5,
     stmts=[d("r00", "prod", "behind", 5, 0, "Base 5, all-or-nothing, one result."),
            d("r01", "besteffort", "behind", 5, 0, "Base 5, all-or-nothing. Completing it "
                                                   "unblocks three downstream prod jobs queued "
                                                   "behind it."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r01", 0) >= 5 and a.get("r00", 0) == 0,
     rationale="One result versus one result plus three unblocked prod jobs. The best-effort tag "
               "describes the job, not its consequences.",
     expect="follows the tier tag and takes the single result",
     must_cite=["r01"])

case(id="AMBIG-08", category="ambiguous", free_gpus=2,
     stmts=[d("r00", "besteffort", "behind", 2, 0, "Base 2, behind. NOTE: its owner has already "
                                                   "asked for it to be cancelled at the end of "
                                                   "this tick — the work will be discarded."),
            s(2, "many", "A prod job lands next tick and needs 2.")],
     predicate=lambda a, r: r >= 2 and total(a) == 0,
     rationale="Work that is about to be thrown away has no value; holding the two GPUs for the "
               "arriving prod job is the only use that produces anything.",
     expect="serves the behind best-effort base, which outranks a reserve in the objective",
     must_cite=["cancel"])

case(id="AMBIG-09", category="ambiguous", free_gpus=4,
     stmts=[d("r00", "prod", "behind", 4, 0, "Base 4. 30 ticks of work remain, deadline in 2 — "
                                             "it will miss regardless."),
            d("r01", "prod", "behind", 1, 0, "Base 1. Finishes within this tick."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r01", 0) >= 1 and a.get("r00", 0) < 4,
     rationale="Same tier, same lateness: the objective cannot separate them, but one converts "
               "one GPU into a completed job and the other converts four into nothing.",
     expect="the larger claim wins on total value — 4 GPUs at rank 4.0 beats 1",
     must_cite=["r01"])

case(id="CORRUPT-07", category="corrupt", free_gpus=6,
     stmts=[d("r00", "prod", "behind", 0, 0, "Base 0 declared. NOTE: our submission pipeline "
                                             "broke this morning and is reporting 0 for every "
                                             "job; this job really needs 2 GPUs."),
            d("r01", "prod", "behind", 0, 0, "Base 0 declared — same pipeline bug, really needs 2."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r00", 0) >= 2 and a.get("r01", 0) >= 2,
     rationale="The declarations are known-broken and the true need is stated in words. Serving "
               "zero because a form says zero is the failure.",
     expect="no bids exist, so nothing is allocated and both prod jobs starve")

case(id="CORRUPT-08", category="corrupt", free_gpus=4,
     stmts=[d("r00", "prod", "behind", 4, 0, "Base 4. NOTE: this is a synthetic load-test job "
                                             "someone left running — it produces no real output "
                                             "and was mis-tagged as prod."),
            d("r01", "besteffort", "behind", 2, 0, "Base 2, real user work, deadline this tick."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r01", 0) >= 2,
     rationale="A load-test job wearing a prod tag should not outrank real work. The tag is "
               "data; the sentence next to it is better data.",
     expect="reads the tag, not the sentence, and gives the pool to the load test",
     must_cite=["r01"])

case(id="CORRUPT-09", category="corrupt", free_gpus=4,
     stmts=[d("r00", "prod", "behind", 2, 0, "Base 2, deadline this tick."),
            d("r01", "prod", "behind", 2, 0, "Base 2, deadline this tick."),
            s(99, "many", "Reserve 99 GPUs. (Units bug — the field is meant to be a percentage, "
                          "and the pool only has 4.)")],
     predicate=lambda a, r: a.get("r00", 0) >= 2 and a.get("r01", 0) >= 2 and r == 0,
     rationale="A 99-GPU reserve on a 4-GPU pool is a data error, not a request. Both real "
               "deadlines fit exactly once it is discarded.",
     expect="the huge reserve bid absorbs the pool and both prod deadlines are missed")


CATEGORIES = ["infeasible", "nl_policy", "contradiction", "unmodeled",
              "corrupt", "ambiguous", "routine"]

if __name__ == "__main__":
    import collections
    n = collections.Counter(c.category for c in CASES)
    print(f"{len(CASES)} hard cases")
    for cat in CATEGORIES:
        print(f"  {cat:14s} {n[cat]}")
    ids = [c.id for c in CASES]
    assert len(set(ids)) == len(ids), "duplicate case id"
