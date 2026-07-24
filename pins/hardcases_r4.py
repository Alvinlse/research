"""ROUND 4 of the hard-case suite — 50 primary + 8 control cases to POWER the debate-structure
test (Exp 89). Pre-registration: docs/superpowers/specs/2026-07-24-exp89-round3-extension-design.md

WHY THIS ROUND EXISTS. Exp 88 budget-controlled the only surviving pro-LLM result: on the
round-3 PRIMARY 31, debate-pkt 14/31 beat the budget-matched single-pkt-boN 10/31 (b=5 c=1,
McNemar p=0.22 — inconclusive by design at n=31). The point estimate says debate's edge is
STRUCTURE, not budget. This round adds 50 primary cases so the POOLED test (r3 31 + r4 50 = 81)
reaches ~0.83 power, with the new 50 also reported alone as the clean confirmatory number.

THE AUTHORING RULE (round-3's rule, sharpened — see the spec for the full argument):

  1. TEXT-DEPENDENCE. Deleting the exception sentence leaves a scene whose NUMBERS imply a
     DIFFERENT, perfectly reasonable answer. The no-text arm is then a real ablation.

  2. SYSTEMATIC-FIRST-READING (new). The case is built so a single pass is pulled toward a
     specific, CONSISTENT wrong reading — a salient surface cue points one way, a subtler
     governing fact in the prose points the other. This is the mechanistic target of the test:
       - best-of-N fixes STOCHASTIC error (it votes away noise) but CANNOT fix a systematic
         first reading — all N samples agree on the wrong answer and the vote confirms it.
       - a rebuttal round CAN fix a systematic first reading — the second perspective challenges
         the cue the first pass locked onto.
     So these cases are exactly where structure should beat matched-budget sampling.

PRE-REGISTRATION DISCIPLINE (see spec §6):
  - Authored BLIND to Exp 88's per-case outcomes (never opened results_exp88_*.json).
  - Iterated against the RIGID arms ONLY. Acceptance: every PRIMARY case has BOTH ilp and rule
    FAIL its predicate; every CONTROL has BOTH PASS. No LLM/debate/boN run during authoring.
  - Composition proportional to r3 PRIMARY (blind to which categories won):
    nl_policy 13, unmodeled 13, corrupt 8, contradiction 6, ambiguous 5, infeasible 5 = 50.
    Controls: placebo/confirm = 8.
  - The static ILP objective and rule floor are the ones in hardcase_eval.py, never tuned here.

Validate with:  .venv/bin/python -m pins.hardcase_eval --no-llm --suite r4
"""
from __future__ import annotations

from pins.hardcases import CATEGORIES, HardCase, d, s, total
from pins.hardcases_r3 import _satisfiable

CASES_R4: list[HardCase] = []


def case(**kw):
    CASES_R4.append(HardCase(**kw))


# =========================================================================== #
#  PRIMARY — 50 text-dependent, systematic-first-reading cases.
# =========================================================================== #

# --------------------------------------------------------------------------- #
# NL-POLICY — an operator rule that lives only in words. 13 cases. The surface
# number invites a consistent misread; a subtler clause governs.
# --------------------------------------------------------------------------- #

case(id="P4-POLICY-01", category="nl_policy", free_gpus=6,
     stmts=[d("r00", "prod", "behind", 2, 0, "Base 2, deadline this tick."),
            d("r01", "prod", "ontrack", 2, 0, "Base 2, 20 ticks of slack."),
            s(0, "none", "OPERATOR NOTE: this rack is fed by two substations rated 2 GPUs each. "
                         "Substation B is offline for maintenance until 16:00, so the real "
                         "ceiling right now is 2 GPUs total, not four.")],
     predicate=lambda a, r: total(a) + r <= 2 and a.get("r00", 0) >= 2,
     rationale="The note gives a surface product — two substations of two — and then removes "
               "one. The true cap is 2, and the behind job takes it; the slack job waits.",
     expect="reads '2 each, two substations' as 4, serves both bases, and trips the live cap",
     must_cite=["substation"])

case(id="P4-POLICY-02", category="nl_policy", free_gpus=8,
     stmts=[d("r00", "prod", "ontrack", 2, 2, "Base 2 + 2 margin."),
            d("r01", "prod", "ontrack", 2, 2, "Base 2 + 2 margin."),
            s(0, "none", "GREEN-COMPUTE DIRECTIVE until noon: cut DISCRETIONARY GPU use in half. "
                         "Committed bases are not discretionary; the requested margins are.")],
     predicate=lambda a, r: (a.get("r00", 0) >= 2 and a.get("r01", 0) >= 2
                             and 4 <= total(a) + r <= 6),
     rationale="Halve the discretionary part only: the 4 margin GPUs become 2, the 4 base GPUs "
               "stay. 'Cut in half' read across the whole request would starve committed work.",
     expect="either fills all 8 (ignores the note) or halves everything to 4, cutting bases",
     must_cite=["discretionary"])

case(id="P4-POLICY-03", category="nl_policy", free_gpus=6,
     stmts=[d("r00", "prod", "behind", 4, 0, "Base 4, deadline this tick. FAIRSHARE BANNER: your "
                                             "group has used 3 of its 4-GPU daily cap on the "
                                             "other rack; 1 remains."),
            d("r01", "besteffort", "ontrack", 2, 0, "Base 2, elastic."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r00", 0) <= 1 and a.get("r01", 0) >= 2,
     rationale="The 4 in the request and the 4 in the cap are different fours — the group has 1 "
               "GPU of quota left. Reading the request width instead of the remaining quota "
               "blows the daily cap and starves the other job.",
     expect="serves the prod base 4 on tier and ignores a quota stated only in the banner",
     must_cite=["cap"])

case(id="P4-POLICY-04", category="nl_policy", free_gpus=4,
     stmts=[d("r00", "prod", "behind", 4, 0, "Base 4. 30 ticks of work remain."),
            d("r01", "besteffort", "ontrack", 2, 0, "Base 2. One tick of work left, then it exits."),
            s(0, "none", "OPERATOR NOTE: this node begins a hard maintenance drain in 2 ticks — "
                         "anything still running is killed, no checkpoint. Only start work that "
                         "completes before the drain.")],
     predicate=lambda a, r: a.get("r01", 0) >= 2 and a.get("r00", 0) == 0,
     rationale="The drain makes 'behind and high tier' irrelevant: the long job cannot finish "
               "and will be killed mid-run, wasting the GPUs. The short job completes and banks "
               "its result.",
     expect="serves the behind prod job into a drain that kills it two ticks later",
     must_cite=["drain"])

case(id="P4-POLICY-05", category="nl_policy", free_gpus=4,
     stmts=[d("r00", "prod", "behind", 4, 0, "Base 4, deadline this tick. Dataset is "
                                             "export-controlled and may run only on the air-gapped "
                                             "GPUs — of which this pool has none."),
            d("r01", "besteffort", "behind", 3, 0, "Base 3, deadline this tick, no restriction."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r01", 0) >= 3 and a.get("r00", 0) == 0,
     rationale="The top-ranked job legally cannot run on these GPUs at all. Awarding it capacity "
               "it may not use strands the pool while a runnable deadline goes unmet.",
     expect="awards the base 4 to the highest-value job and strands the whole pool",
     must_cite=["export"])

case(id="P4-POLICY-06", category="nl_policy", free_gpus=4,
     stmts=[d("r00", "prod", "behind", 4, 0, "Base 4, deadline in 1 tick."),
            d("r01", "besteffort", "ontrack", 4, 0, "Base 4. OPERATOR OVERRIDE: this incident "
                                                    "mitigation job outranks everything on the "
                                                    "cluster until the outage is closed — serve it "
                                                    "first regardless of tier tag."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r01", 0) >= 4 and a.get("r00", 0) == 0,
     rationale="A stated operator override inverts the tier order for one job. The best-effort "
               "tag is stale relative to the incident; the words carry the real priority.",
     expect="serves the prod-behind job — the tier tag is the only priority it can see",
     must_cite=["override"])

case(id="P4-POLICY-07", category="nl_policy", free_gpus=4,
     stmts=[d("r00", "prod", "ontrack", 4, 0, "Base 4. PARTITION POLICY: the prod tag grants "
                                              "priority only 06:00-22:00; outside those hours "
                                              "every job on this partition is best-effort. Local "
                                              "time is 02:00."),
            d("r01", "besteffort", "behind", 4, 0, "Base 4, deadline this tick."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r01", 0) >= 4 and a.get("r00", 0) == 0,
     rationale="At 02:00 the partition policy demotes the prod tag to best-effort, so the two "
               "jobs rank by deadline — and only r01 has one. The tag no longer means what it says.",
     expect="ranks the prod tag above best-effort and serves r00, per the static tier order",
     must_cite=["22:00"])

case(id="P4-POLICY-09", category="nl_policy", free_gpus=6,
     stmts=[d("r00", "prod", "behind", 4, 0, "Base 4, deadline this tick."),
            d("r01", "prod", "behind", 2, 0, "Base 2, deadline this tick."),
            s(0, "none", "FAIR-USE RULE effective today: a per-job ceiling of 2 GPUs applies to "
                         "every job on this pool — larger jobs run trimmed to 2. Total pool "
                         "capacity is otherwise unaffected.")],
     predicate=lambda a, r: a.get("r00", 0) <= 2 and a.get("r01", 0) >= 2,
     rationale="A per-job ceiling of 2 caps the large job regardless of its width or tier; the "
               "smaller job is unaffected. Reading only the requested widths blows the ceiling on "
               "r00 and starves r01.",
     expect="gives the base-4 prod job its full width — no per-job cap exists in the objective",
     must_cite=["per-job"])

case(id="P4-POLICY-08", category="nl_policy", free_gpus=8,
     stmts=[d("r00", "prod", "ontrack", 6, 0, "Base 6."),
            d("r01", "besteffort", "ontrack", 2, 0, "Base 2, elastic."),
            s(0, "none", "OPERATOR NOTE: breaker limit is 5 GPUs PER PHASE and this rack is a "
                         "single phase — no job and no total may exceed 5 concurrent GPUs.")],
     predicate=lambda a, r: a.get("r00", 0) <= 5 and total(a) + r <= 5,
     rationale="'5 per phase, single phase' is a 5-GPU ceiling, not a 10-GPU one. The base 6 "
               "cannot be honoured; it clips to 5 and leaves nothing for the pool.",
     expect="reads 'per phase' as one of several phases and serves the full base 6",
     must_cite=["phase"])

case(id="P4-POLICY-10", category="nl_policy", free_gpus=6,
     stmts=[d("r00", "prod", "behind", 3, 0, "Base 3, deadline this tick."),
            d("r01", "prod", "behind", 3, 0, "Base 3, deadline this tick."),
            s(0, "none", "THERMAL NOTE: the two jobs share a cooling zone rated for 3 GPUs of "
                         "sustained draw. Run them together at 6 and both throttle to half speed "
                         "and miss; run one at a time and it finishes clean.")],
     predicate=lambda a, r: (a.get("r00", 0) >= 3) != (a.get("r01", 0) >= 3),
     rationale="Six GPUs are free but the shared cooling zone only supports three at speed. "
               "Serving both — which the capacity allows — throttles both into missing; serving "
               "one saves one.",
     expect="both bases fit in the pool, so it serves both and throttles both to a miss",
     must_cite=["cooling"])

case(id="P4-POLICY-11", category="nl_policy", free_gpus=4,
     stmts=[d("r00", "besteffort", "ontrack", 2, 0, "Base 2. CONTRACT NOTE: customer Aoba has a "
                                                    "signed floor of 2 guaranteed GPUs at all "
                                                    "times, independent of tier — this is their "
                                                    "job."),
            d("r01", "prod", "ontrack", 4, 0, "Base 4."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r00", 0) >= 2 and a.get("r01", 0) <= 2,
     rationale="A contractual floor overrides the tier ranking for two GPUs. The best-effort tag "
               "does not describe this job's real guarantee; the contract does.",
     expect="ranks prod over best-effort and gives the prod job all four",
     must_cite=["contract"])

case(id="P4-POLICY-12", category="nl_policy", free_gpus=4,
     stmts=[d("r00", "prod", "behind", 2, 0, "Base 2, deadline this tick."),
            d("r01", "prod", "behind", 2, 0, "Base 2, deadline this tick. ANTI-AFFINITY: r01 and "
                                             "r00 must not share a node — they saturate the same "
                                             "NVLink and corrupt each other's collectives. This "
                                             "pool is one node."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: (a.get("r00", 0) >= 2) != (a.get("r01", 0) >= 2),
     rationale="Both bases fit the node numerically, but a stated anti-affinity forbids running "
               "them together. One runs and finishes; running both corrupts both.",
     expect="packs both bases onto the one node and corrupts both collectives",
     must_cite=["affinity"])

case(id="P4-POLICY-13", category="nl_policy", free_gpus=6,
     stmts=[d("r00", "besteffort", "behind", 3, 0, "Base 3, deadline this tick."),
            s(3, "none", "CLUSTER FREEZE for the accreditation benchmark begins now: hold the pool "
                         "empty and admit no new best-effort work until the benchmark run is "
                         "logged. This is a dated compliance deliverable.")],
     predicate=lambda a, r: r >= 3 and total(a) == 0,
     rationale="A dated compliance freeze holds the pool against an ordinary best-effort job that "
               "the numbers alone would happily serve. The instruction exists only as words.",
     expect="serves the present job — an empty reserve ranks below any real bid",
     must_cite=["freeze"])

# --------------------------------------------------------------------------- #
# UNMODELED — real value with no slot in the objective. 13 cases. The tier/size
# cue is salient and points away from the fact that should decide.
# --------------------------------------------------------------------------- #

case(id="P4-UNMOD-01", category="unmodeled", free_gpus=2,
     stmts=[d("r00", "prod", "ontrack", 2, 0, "Base 2. Routine nightly aggregation; the report is "
                                              "read next week."),
            d("r01", "besteffort", "ontrack", 2, 0, "Base 2. A live investor demo starts in five "
                                                    "minutes and renders on this job — people are "
                                                    "already in the room."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r01", 0) >= 2 and a.get("r00", 0) == 0,
     rationale="One job has a hard human deadline in minutes and no tier tag to prove it; the "
               "other is high-tier and entirely slack. The value is in the room, not the tag.",
     expect="serves the prod job on tier and lets the demo fail",
     must_cite=["demo"])

case(id="P4-UNMOD-02", category="unmodeled", free_gpus=2,
     stmts=[d("r00", "prod", "ontrack", 2, 0, "Base 2. Its compute grant rolls over to next "
                                              "quarter — no penalty if it waits."),
            d("r01", "besteffort", "ontrack", 2, 0, "Base 2. Its grant is use-it-or-lose-it and "
                                                    "expires at midnight tonight; unspent hours "
                                                    "are forfeit."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r01", 0) >= 2 and a.get("r00", 0) == 0,
     rationale="Deferring the prod job costs nothing; deferring the best-effort one forfeits its "
               "funding. Expiry of a budget is nowhere in a scheduler's objective.",
     expect="serves the higher tier and lets a funded allocation expire unused",
     must_cite=["expire"])

case(id="P4-UNMOD-03", category="unmodeled", free_gpus=3,
     stmts=[d("r00", "prod", "behind", 3, 0, "Base 3. Fresh start; 40 ticks of work ahead."),
            d("r01", "besteffort", "behind", 3, 0, "Base 3. At epoch 99 of 100 — one tick from "
                                                   "done, and it checkpoints on completion only."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r01", 0) >= 3 and a.get("r00", 0) == 0,
     rationale="One tick finishes a nearly-complete run and banks 100 epochs; the same tick barely "
               "dents a fresh job. Sunk progress is real value the submission record cannot show.",
     expect="serves the behind prod job and defers the run that was one tick from finishing",
     must_cite=["epoch"])

case(id="P4-UNMOD-04", category="unmodeled", free_gpus=2,
     stmts=[d("r00", "prod", "ontrack", 2, 0, "Base 2. Standalone; its output feeds nothing else."),
            d("r01", "besteffort", "ontrack", 2, 0, "Base 2. Twelve downstream jobs are blocked "
                                                    "waiting on this one's output and cannot start "
                                                    "until it finishes."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r01", 0) >= 2 and a.get("r00", 0) == 0,
     rationale="Finishing the best-effort job unblocks twelve others; finishing the prod job "
               "unblocks none. The dependency fan-out is invisible to a per-job objective.",
     expect="serves the prod job — downstream fan-out is not a term it optimises",
     must_cite=["downstream"])

case(id="P4-UNMOD-05", category="unmodeled", free_gpus=2,
     stmts=[d("r00", "prod", "ontrack", 2, 0, "Base 2. 50 ticks of work remain."),
            d("r01", "prod", "ontrack", 2, 0, "Base 2. Finishes in a single tick, then releases "
                                              "its GPUs back to the queue where nine jobs wait."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r01", 0) >= 2 and a.get("r00", 0) == 0,
     rationale="Serving the one-tick job clears it and frees capacity for a full queue next tick; "
               "serving the 50-tick job locks the GPUs up. Throughput over the queue is unmodeled.",
     expect="a tie on every scored axis, broken by index rather than by makespan",
     must_cite=["releases"])

case(id="P4-UNMOD-06", category="unmodeled", free_gpus=2,
     stmts=[d("r00", "prod", "ontrack", 2, 0, "Base 2. Its 400 GB dataset must be staged from cold "
                                              "archive first — a six-hour transfer before it can "
                                              "even start."),
            d("r01", "besteffort", "ontrack", 2, 0, "Base 2. Its dataset is already resident on "
                                                    "this node's local NVMe; starts immediately."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r01", 0) >= 2 and a.get("r00", 0) == 0,
     rationale="Handing GPUs to the prod job leaves them idle for six hours of staging while a "
               "ready job waits. Data-transfer latency is not a variable in the objective.",
     expect="serves the prod job and idles the GPUs through a six-hour stage-in",
     must_cite=["stage"])

case(id="P4-UNMOD-07", category="unmodeled", free_gpus=4,
     stmts=[d("r00", "prod", "behind", 4, 0, "Base 4, deadline this tick. Pure inference sweep — "
                                             "restarting it later costs nothing but time."),
            d("r01", "besteffort", "ontrack", 4, 0, "Base 4. Holds a non-transferable hardware "
                                                    "license dongle that is checked out to this "
                                                    "node until 18:00; if it yields now the dongle "
                                                    "is stranded and no job can use it until then."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r01", 0) >= 4 and a.get("r00", 0) == 0,
     rationale="Preempting the licensed job strands a scarce dongle for hours; the prod job loses "
               "only a resumable restart. The dongle lock exists nowhere in the model.",
     expect="serves the behind prod deadline and strands the licence for the afternoon",
     must_cite=["dongle"])

case(id="P4-UNMOD-08", category="unmodeled", free_gpus=2,
     stmts=[d("r00", "prod", "behind", 2, 0, "Base 2, deadline this tick. Its results are compared "
                                             "against a reviewer's rerun; either order is fine."),
            d("r01", "besteffort", "behind", 2, 0, "Base 2, deadline this tick. This is the live "
                                                   "leaderboard scoring job for a competition whose "
                                                   "submission window shuts in one tick and never "
                                                   "reopens."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r01", 0) >= 2 and a.get("r00", 0) == 0,
     rationale="Both are behind, but only one faces a window that never reopens; the other can be "
               "rerun. Irreversibility is the deciding fact and it is stated only in prose.",
     expect="serves the prod tier and misses a one-time competition window",
     must_cite=["window"])

case(id="P4-UNMOD-09", category="unmodeled", free_gpus=2,
     stmts=[d("r00", "prod", "ontrack", 2, 0, "Base 2. Runs entirely on-prem, no external cost."),
            d("r01", "besteffort", "ontrack", 2, 0, "Base 2. If it does not run here it fails over "
                                                    "to metered cloud GPUs that bill the grant at "
                                                    "$40/hr; running it locally is free."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r01", 0) >= 2 and a.get("r00", 0) == 0,
     rationale="Deferring the best-effort job spends real money on cloud failover; deferring the "
               "prod job spends nothing. Budget burn on another cluster is off the scheduler's "
               "books.",
     expect="serves the prod tier and pushes the other job onto a metered cloud bill",
     must_cite=["cloud"])

case(id="P4-UNMOD-10", category="unmodeled", free_gpus=4,
     stmts=[d("r00", "prod", "behind", 4, 0, "Base 4, deadline this tick. Batch model training; "
                                             "the checkpoint it would produce is one of forty "
                                             "routine ones."),
            d("r01", "besteffort", "ontrack", 4, 0, "Base 4. A PhD student's thesis defense is at "
                                                    "09:00 tomorrow and this generates the only "
                                                    "figure still missing from the slides."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r01", 0) >= 4 and a.get("r00", 0) == 0,
     rationale="A once-in-a-career human deadline sits under a best-effort tag; the prod job loses "
               "one interchangeable checkpoint. Neither the stakes nor the tag's staleness is "
               "modelled.",
     expect="serves the behind prod job and misses the thesis figure overnight",
     must_cite=["thesis"])

case(id="P4-UNMOD-11", category="unmodeled", free_gpus=3,
     stmts=[d("r00", "prod", "behind", 3, 0, "Base 3, deadline this tick. Deterministic; produces "
                                             "the same output whenever it runs."),
            d("r01", "besteffort", "behind", 3, 0, "Base 3, deadline this tick. Reads a live "
                                                   "market data feed that is only available during "
                                                   "today's trading session — after close the "
                                                   "input is gone for good."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r01", 0) >= 3 and a.get("r00", 0) == 0,
     rationale="The prod job's inputs are permanent; the best-effort job's inputs vanish at the "
               "closing bell. A perishable input is not a deadline the model represents.",
     expect="serves the prod tier and loses a perishable live feed the other job needed",
     must_cite=["feed"])

case(id="P4-UNMOD-12", category="unmodeled", free_gpus=4,
     stmts=[d("r00", "prod", "behind", 4, 0, "Base 4, deadline this tick. Note from the owner: it "
                                             "has already blown its real deadline by a day — the "
                                             "result is now worthless, we just forgot to cancel."),
            d("r01", "besteffort", "behind", 4, 0, "Base 4, deadline this tick, result still "
                                                   "needed."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r01", 0) >= 4 and a.get("r00", 0) == 0,
     rationale="The high-tier job's own owner says its output no longer has value; the flag it "
               "carries is stale. Serving it burns the pool on a result nobody will use.",
     expect="protects the prod tier flag and serves a job whose value has already lapsed",
     must_cite=["worthless"])

case(id="P4-UNMOD-13", category="unmodeled", free_gpus=2,
     stmts=[d("r00", "besteffort", "ontrack", 2, 0, "Base 2. Cold start; model weights load from "
                                                    "disk either way."),
            d("r01", "besteffort", "ontrack", 2, 0, "Base 2. Its 70B weights are already sharded "
                                                    "across this node's GPU memory from last tick; "
                                                    "moving it costs a twelve-tick reload."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r01", 0) >= 2 and a.get("r00", 0) == 0,
     rationale="Identical on tier, deadline and size; one is already warm in this node's memory. "
               "Resident state is real value that the submission record does not carry.",
     expect="an arbitrary tie-break — the two jobs look identical to the solver",
     must_cite=["sharded"])

# --------------------------------------------------------------------------- #
# CORRUPT — the declared numbers are wrong and the prose says how. 8 cases.
# --------------------------------------------------------------------------- #

case(id="P4-CORRUPT-01", category="corrupt", free_gpus=6,
     stmts=[d("r00", "prod", "behind", 8, 0, "Base 8 — but that field is GB of VRAM per worker, "
                                             "mis-entered into the GPU-count box. The real need is "
                                             "2 GPUs."),
            d("r01", "besteffort", "ontrack", 4, 0, "Base 4, elastic."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r00", 0) == 2 and a.get("r01", 0) >= 4,
     rationale="The 8 is a units error its owner has flagged; the true width is 2, at which both "
               "jobs fit. Clipping the bogus 8 to the pool starves the other job for nothing.",
     expect="clips the declared 8 to the pool and hands r00 nearly everything",
     must_cite=["VRAM"])

case(id="P4-CORRUPT-02", category="corrupt", free_gpus=6,
     stmts=[d("r00", "prod", "behind", 6, 0, "Base 6 declared. That already folds in a 4-GPU "
                                             "safety hedge on top of the true base of 2 — the "
                                             "hedge should have gone in the margin field, not the "
                                             "base."),
            d("r01", "besteffort", "behind", 4, 0, "Base 4, deadline this tick."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r00", 0) <= 2 and a.get("r01", 0) >= 4,
     rationale="The real base is 2; the other 4 is a discretionary hedge misfiled as committed "
               "work. Read literally it swallows the pool ahead of a saveable deadline.",
     expect="treats all 6 as committed base and funds it before a best-effort deadline",
     must_cite=["hedge"])

case(id="P4-CORRUPT-03", category="corrupt", free_gpus=4,
     stmts=[d("r00", "prod", "ontrack", 4, 0, "Base 4. Our submit template tagged this prod; it is "
                                              "actually my personal hyperparameter doodle, no "
                                              "urgency at all."),
            d("r01", "besteffort", "behind", 4, 0, "Base 4, deadline this tick — this is the "
                                                   "production inference job, mis-tagged "
                                                   "best-effort by the same broken template."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r01", 0) >= 4 and a.get("r00", 0) == 0,
     rationale="Both tier tags are wrong and both owners say so; the sentences carry the true "
               "priorities the enum inverts. The tag is data, the correction is better data.",
     expect="reads the tags literally and serves the doodle over live inference",
     must_cite=["template"])

case(id="P4-CORRUPT-04", category="corrupt", free_gpus=6,
     stmts=[d("r00", "prod", "behind", 6, 0, "Base 6. Heads up: the autoscaler that wrote this "
                                             "figure railed to maximum during a metrics outage — "
                                             "the job genuinely needs 2, not 6."),
            d("r01", "prod", "behind", 3, 0, "Base 3, deadline this tick."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r00", 0) == 2 and a.get("r01", 0) >= 3,
     rationale="A telemetry rail inflated the request to its ceiling; the disclosed true need is "
               "2. At 2 both prod deadlines fit; at 6 one is starved by a measurement artefact.",
     expect="honours the railed 6 and crowds out the second prod deadline",
     must_cite=["railed"])

case(id="P4-CORRUPT-05", category="corrupt", free_gpus=4,
     stmts=[d("r00", "prod", "behind", 1, 0, "Base 1 — but that is the untouched template default; "
                                             "this distributed job actually needs 4 to start at "
                                             "all, and its deadline is this tick."),
            d("r01", "besteffort", "ontrack", 3, 0, "Base 3, elastic, no deadline."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r00", 0) >= 4,
     rationale="The declared 1 is an unedited default; the real requirement is 4 and the job "
               "cannot run on less. Taking the number literally gives it a single useless GPU.",
     expect="believes the base-1 declaration and hands the spare GPUs to the elastic job",
     must_cite=["default"])

case(id="P4-CORRUPT-06", category="corrupt", free_gpus=4,
     stmts=[d("r00", "prod", "behind", 4, 0, "Base 4. The 'behind' flag is left over from a crash "
                                             "last night that was already rerun successfully — "
                                             "this instance has a full day of slack."),
            d("r01", "besteffort", "behind", 2, 0, "Base 2, genuinely behind, deadline this tick."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r01", 0) >= 2 and a.get("r00", 0) <= 2,
     rationale="The prod job's lateness is a stale artefact of an already-fixed crash; the "
               "best-effort job's lateness is real and imminent. The flags read identical; only "
               "the prose separates them.",
     expect="serves the base-4 prod job on its stale 'behind' flag and starves a real deadline",
     must_cite=["crash"])

case(id="P4-CORRUPT-07", category="corrupt", free_gpus=4,
     stmts=[d("r00", "prod", "behind", 2, 0, "Base 2, deadline this tick."),
            d("r01", "prod", "behind", 2, 0, "Base 2, deadline this tick. NOTE: r01 is an "
                                             "accidental duplicate submission of r00 — same code, "
                                             "same output path, identical job. Only one should "
                                             "run; the second just wastes two GPUs."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: (a.get("r00", 0) >= 2) != (a.get("r01", 0) >= 2),
     rationale="The two ids are the same job submitted twice. Both fit in the pool, so the solver "
               "happily runs both — and half the capacity produces a redundant result.",
     expect="serves both bases, which fit, and burns two GPUs on a duplicate",
     must_cite=["duplicate"])

case(id="P4-CORRUPT-08", category="corrupt", free_gpus=6,
     stmts=[d("r00", "prod", "behind", 6, 0, "Base 6, deadline this tick. Correction: three of "
                                             "those six workers already finished and checkpointed "
                                             "last tick — only 3 remain to schedule."),
            d("r01", "besteffort", "behind", 3, 0, "Base 3, deadline this tick."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r00", 0) == 3 and a.get("r01", 0) >= 3,
     rationale="Half the declared width is already-completed work; the live requirement is 3. At 3 "
               "both deadlines are met, at 6 the stale half evicts the other job.",
     expect="schedules all six workers, re-running three that already finished, and starves r01",
     must_cite=["finished"])

# --------------------------------------------------------------------------- #
# CONTRADICTION — the stated claims cannot all hold; a clause governs. 6 cases.
# --------------------------------------------------------------------------- #

case(id="P4-CONTRA-01", category="contradiction", free_gpus=4,
     stmts=[d("r00", "prod", "behind", 2, 0, "Base 2, deadline this tick."),
            d("r01", "prod", "behind", 2, 0, "Base 2, deadline this tick. NOTE: r00 and r01 both "
                                             "take an exclusive write-lock on the same dataset — "
                                             "run concurrently and both abort on lock contention. "
                                             "Only one may hold it per tick."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: (a.get("r00", 0) >= 2) != (a.get("r01", 0) >= 2),
     rationale="Capacity fits both, but a stated exclusive lock forbids it — running the pair "
               "aborts the pair. Serving one lets it finish; serving both finishes neither.",
     expect="serves both bases, which fit comfortably, and both abort on the shared lock",
     must_cite=["lock"])

case(id="P4-CONTRA-02", category="contradiction", free_gpus=4,
     stmts=[d("r00", "prod", "behind", 4, 0, "Base 4, deadline this tick. COMPLIANCE HOLD: this "
                                             "job is frozen pending an export-review sign-off that "
                                             "has not arrived — running it now is a policy "
                                             "violation, deadline notwithstanding."),
            d("r01", "besteffort", "behind", 4, 0, "Base 4, deadline this tick, cleared to run."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r01", 0) >= 4 and a.get("r00", 0) == 0,
     rationale="The top-ranked job is under a hold it may not run through, however urgent its "
               "deadline. The only schedulable deadline is the best-effort one.",
     expect="serves the behind prod job straight through a compliance hold",
     must_cite=["hold"])

case(id="P4-CONTRA-03", category="contradiction", free_gpus=8,
     stmts=[d("r00", "prod", "behind", 2, 4, "Base 2 + 4 margin. IGNORE THE MARGIN: anything above "
                                             "2 GPUs overflows this model-parallel config and OOM-"
                                             "crashes the job. The margin field was a mistake."),
            d("r01", "besteffort", "behind", 2, 0, "Base 2, deadline this tick."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r00", 0) == 2 and a.get("r01", 0) >= 2,
     rationale="The requested margin is not just discretionary but actively harmful — funding it "
               "crashes the job it was meant to protect. The base is the whole safe allocation.",
     expect="funds the requested margin ahead of a best-effort base, as the ranking says",
     must_cite=["OOM"])

case(id="P4-CONTRA-04", category="contradiction", free_gpus=4,
     stmts=[d("r00", "prod", "behind", 4, 0, "Base 4, deadline this tick. HARD CONSTRAINT: a "
                                             "reduction-order bug makes results above 1 GPU "
                                             "silently non-deterministic — until it is patched "
                                             "this must run at exactly 1 GPU."),
            d("r01", "besteffort", "behind", 3, 0, "Base 3, deadline this tick."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r00", 0) <= 1 and a.get("r01", 0) >= 3,
     rationale="Correctness caps the prod job at 1 GPU regardless of its request; the freed "
               "capacity meets the other deadline. A wide but wrong result is worse than a narrow "
               "correct one.",
     expect="serves the full base 4 and produces silently non-deterministic output on time",
     must_cite=["non-deterministic"])

case(id="P4-CONTRA-05", category="contradiction", free_gpus=4,
     stmts=[d("r00", "prod", "behind", 4, 0, "Base 4, deadline this tick. OPERATOR NOTE A: clear "
                                             "the backlog, serve the most-behind prod job first."),
            s(4, "many", "OPERATOR NOTE B: full capacity freeze for the certification run starts "
                         "this tick — hold all four GPUs, admit nothing. To be explicit on "
                         "precedence: the certification freeze (B) overrides the backlog "
                         "directive (A); it is a dated contractual gate.")],
     predicate=lambda a, r: r >= 4 and total(a) == 0,
     rationale="Two operator directives point opposite ways on the same GPUs, and note B states "
               "which one wins. Reading that precedence sentence is the entire task.",
     expect="follows note A and serves the backlog — a reserve ranks below any real bid",
     must_cite=["freeze"])

case(id="P4-CONTRA-06", category="contradiction", free_gpus=4,
     stmts=[d("r00", "prod", "behind", 2, 0, "Base 2, deadline this tick. Do not start until r01's "
                                             "calibration table is written — without it this job "
                                             "reads uninitialised values and produces garbage."),
            d("r01", "besteffort", "ontrack", 2, 0, "Base 2. Writes the calibration table r00 "
                                                    "depends on; completes this tick if served."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r01", 0) >= 2 and a.get("r00", 0) == 0,
     rationale="The higher-tier job cannot produce anything until the lower-tier one finishes "
               "first; running it now yields garbage. The dependency runs opposite to the tier "
               "order and lives only in prose.",
     expect="serves the behind prod job this tick into a garbage result",
     must_cite=["calibration"])

# --------------------------------------------------------------------------- #
# AMBIGUOUS — a real tradeoff the prose tips one way. 5 cases.
# --------------------------------------------------------------------------- #

case(id="P4-AMBIG-01", category="ambiguous", free_gpus=4,
     stmts=[d("r00", "prod", "behind", 4, 0, "Base 4, deadline this tick. Memory-bound: it is "
                                             "already 90% as fast on 1 GPU as on 4 — extra GPUs "
                                             "barely help."),
            d("r01", "besteffort", "behind", 3, 0, "Base 3, deadline this tick, needs all 3 to "
                                                   "meet it."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r00", 0) == 1 and a.get("r01", 0) >= 3,
     rationale="One GPU gets the prod job almost all of its value and frees three for a deadline "
               "that would otherwise miss. Diminishing returns are stated only in the prose.",
     expect="pours all four into the prod job for a marginal speedup and drops the other deadline",
     must_cite=["memory-bound"])

case(id="P4-AMBIG-02", category="ambiguous", free_gpus=4,
     stmts=[d("r00", "prod", "behind", 4, 0, "Base 4, deadline this tick. Highly speculative — "
                                             "about a 10% chance of converging; usually it "
                                             "diverges and yields nothing."),
            d("r01", "besteffort", "behind", 4, 0, "Base 4, deadline this tick. Routine job, "
                                                   "certain to produce its result."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r01", 0) >= 4 and a.get("r00", 0) == 0,
     rationale="Room for one. A near-certain result outweighs a mostly-doomed one even at a lower "
               "tier; the success probability appears nowhere in the objective.",
     expect="serves the prod tier and spends the pool on a 10% gamble",
     must_cite=["speculative"])

case(id="P4-AMBIG-03", category="ambiguous", free_gpus=4,
     stmts=[d("r00", "besteffort", "behind", 2, 0, "Base 2, behind. This is retry #4 of a run that "
                                                   "has failed three times with the same error."),
            d("r01", "besteffort", "behind", 2, 0, "Base 2, behind. Retry #5 of that same failing "
                                                   "run."),
            d("r02", "besteffort", "behind", 2, 0, "Base 2, behind. A fresh experiment on its "
                                                   "first attempt."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r02", 0) >= 2,
     rationale="Three identical jobs, room for two. Two are repeat attempts at a run that keeps "
               "failing the same way; progress favours the untried one. The history lives only in "
               "prose.",
     expect="serves the first two by index and starves the only job likely to make progress",
     must_cite=["fresh"])

case(id="P4-AMBIG-04", category="ambiguous", free_gpus=2,
     stmts=[d("r00", "prod", "behind", 2, 0, "Base 2. Deadline is 20 ticks away with 1 tick of "
                                             "work left — comfortable slack."),
            d("r01", "besteffort", "behind", 2, 0, "Base 2. Deadline this tick, 1 tick of work "
                                                   "left — makes it only if served now."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r01", 0) >= 2 and a.get("r00", 0) == 0,
     rationale="The prod job can wait 19 ticks and still make it; the best-effort job misses "
               "unless served now. Serving by tier saves the job that was never at risk.",
     expect="serves the prod tier and lets the only at-risk deadline slip",
     must_cite=["slack"])

case(id="P4-AMBIG-05", category="ambiguous", free_gpus=4,
     stmts=[d("r00", "prod", "behind", 4, 0, "Base 4, deadline this tick. Divisible — usable value "
                                             "scales smoothly with whatever it is given."),
            d("r01", "prod", "behind", 3, 0, "Base 3, deadline this tick. All-or-nothing: it "
                                             "cannot start on fewer than 3 and yields nothing on a "
                                             "partial award."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r01", 0) >= 3 and a.get("r00", 0) <= 1,
     rationale="Give the indivisible job its whole width and let the divisible one absorb the "
               "single remaining GPU. Any split that breaks the all-or-nothing job wastes "
               "everything it receives.",
     expect="pours the pool into the higher-capacity divisible job and strands the indivisible one",
     must_cite=["all-or-nothing"])

# --------------------------------------------------------------------------- #
# INFEASIBLE — triage; the numbers say more fits than really does. 5 cases.
# --------------------------------------------------------------------------- #

case(id="P4-INFEAS-01", category="infeasible", free_gpus=6,
     stmts=[d("r00", "prod", "behind", 3, 0, "Base 3, deadline this tick."),
            d("r01", "prod", "ontrack", 3, 0, "Base 3, 30 ticks of slack."),
            s(0, "none", "NOTE: three of the six GPUs reported free are throwing ECC errors and "
                         "will fault any job placed on them — only three are actually usable.")],
     predicate=lambda a, r: total(a) + r <= 3 and a.get("r00", 0) >= 3,
     rationale="The free count is wrong by half; only three GPUs work. Both bases appear to fit, "
               "but placing six faults three jobs. The behind deadline takes the usable three.",
     expect="trusts the reported six, serves both bases, and faults the jobs on the dead GPUs",
     must_cite=["ECC"])

case(id="P4-INFEAS-02", category="infeasible", free_gpus=4,
     stmts=[d("r00", "prod", "behind", 4, 0, "Base 4, all-or-nothing. 20 ticks of work remain, "
                                             "deadline in 2 — it misses no matter what."),
            d("r01", "prod", "behind", 4, 0, "Base 4, all-or-nothing. 25 ticks remain, deadline in "
                                             "2 — also misses regardless."),
            d("r02", "prod", "behind", 2, 0, "Base 2, all-or-nothing. 1 tick of work left, deadline "
                                             "in 2 — makes it if served whole this tick."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r02", 0) >= 2 and a.get("r00", 0) == 0 and a.get("r01", 0) == 0,
     rationale="Three indivisible claims, and only the smallest can still finish in time. Funding "
               "either larger job spends the whole pool on work that cannot complete.",
     expect="fills the pool with a base-4 job by value and strands the only one that could finish",
     must_cite=["makes it"])

case(id="P4-INFEAS-03", category="infeasible", free_gpus=4,
     stmts=[d("r00", "prod", "behind", 8, 0, "Base 8, deadline this tick — but it has a documented "
                                             "degraded mode that runs on 2 GPUs and still meets "
                                             "the deadline. It cannot use more than the pool holds."),
            d("r01", "besteffort", "behind", 2, 0, "Base 2, deadline this tick."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r00", 0) == 2 and a.get("r01", 0) >= 2,
     rationale="The request exceeds the pool, but the job has a 2-GPU fallback that still meets "
               "its deadline — leaving room for the other one too. Clipping the 8 to the pool "
               "wastes that headroom.",
     expect="clips the impossible base-8 request to the whole pool and starves r01",
     must_cite=["degraded"])

case(id="P4-INFEAS-04", category="infeasible", free_gpus=4,
     stmts=[d("r00", "prod", "behind", 4, 0, "Base 4, all-or-nothing. Deadline in 2 ticks, 30 "
                                             "ticks of work left — cannot finish in time."),
            d("r01", "prod", "behind", 4, 0, "Base 4, all-or-nothing. Deadline in 2 ticks, 1 tick "
                                             "of work left — finishes if served whole now."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r01", 0) >= 4 and a.get("r00", 0) == 0,
     rationale="Two indivisible prod claims, room for one. Only r01 can still make its deadline; "
               "the tie the numbers show hides that r00's work cannot fit its remaining time.",
     expect="breaks the tie by index and strands four GPUs on a job that cannot finish",
     must_cite=["cannot finish"])

case(id="P4-INFEAS-05", category="infeasible", free_gpus=4,
     stmts=[d("r00", "prod", "behind", 2, 0, "Base 2, deadline this tick. If it misses, resubmit "
                                             "tonight at no cost."),
            d("r01", "prod", "behind", 2, 0, "Base 2, deadline this tick. If it misses, simply "
                                             "late — nobody minds."),
            d("r02", "prod", "behind", 2, 0, "Base 2, deadline this tick. Reads a sensor buffer "
                                             "that is overwritten this tick — miss it and the "
                                             "input is gone forever, the run cannot be redone."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: (a.get("r02", 0) >= 2
                             and sum(1 for j in ("r00", "r01", "r02") if a.get(j, 0) >= 2) == 2),
     rationale="Three identical prod deadlines, room for two. Only one is irreproducible; the "
               "other two are recoverable. r02 must be one of the two served, and the difference "
               "is only in prose.",
     expect="serves the first two by index and loses the one input that cannot be regenerated",
     must_cite=["overwritten"])

# INSERT-PRIMARY

# =========================================================================== #
#  CONTROLS — 8 cases. Text present, text irrelevant/confirming. Effect must be ~0.
# =========================================================================== #

# --------------------------------------------------------------------------- #
# PLACEBO — prose present, prose inert. Ordinary packing is correct. 4 cases.
# --------------------------------------------------------------------------- #

case(id="P4-PLACEBO-01", category="placebo", free_gpus=8,
     stmts=[d("r00", "prod", "ontrack", 3, 0, "Base 3. This is the fourth milestone of a project "
                                              "that started two summers ago; we finally got the "
                                              "data pipeline ported off the retired cluster. It "
                                              "has been quite a journey, honestly."),
            d("r01", "besteffort", "ontrack", 3, 0, "Base 3. Same lab, a different grant line. "
                                                    "Grateful for the ops team's patience this "
                                                    "quarter."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r00", 0) >= 3 and a.get("r01", 0) >= 3,
     rationale="Two bases, ample capacity, and a paragraph of backstory that changes nothing. "
               "Serving both is simply correct.",
     expect="rigid arms handle this correctly; included so the suite can lose")

case(id="P4-PLACEBO-02", category="placebo", free_gpus=4,
     stmts=[d("r00", "besteffort", "ontrack", 4, 0, "Base 4. Please, I am begging you — I have been "
                                                    "waiting all day, my advisor is livid, and I "
                                                    "will be in serious trouble if this does not "
                                                    "run tonight. Please, please prioritise me."),
            d("r01", "prod", "behind", 4, 0, "Base 4, deadline in 1 tick."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r01", 0) >= 4 and a.get("r00", 0) == 0,
     rationale="Distress is not evidence. The appeal states no fact that changes the allocation, "
               "and the ordinary ranking is right.",
     expect="rigid arms handle this correctly; included so the suite can lose")

case(id="P4-PLACEBO-03", category="placebo", free_gpus=4,
     stmts=[d("r00", "besteffort", "ontrack", 4, 0, "Base 4. STUDENT coursework. NOTE: the "
                                                    "student-protection directive that used to "
                                                    "shield these jobs was retired last month and "
                                                    "no longer applies to this queue."),
            d("r01", "prod", "behind", 4, 0, "Base 4, deadline this tick."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r01", 0) >= 4 and a.get("r00", 0) == 0,
     rationale="A deliberate keyword trap: the word 'student' sits inside a note stating the "
               "student policy has EXPIRED. An arm that pattern-matches the trigger word protects "
               "a job the rule no longer covers; the ordinary ranking is correct.",
     expect="rigid arms handle this correctly; included so the suite can lose")

case(id="P4-PLACEBO-04", category="placebo", free_gpus=4,
     stmts=[d("r00", "besteffort", "ontrack", 4, 0, "Base 4. Flagged URGENT in the submit script — "
                                                    "though to be honest that flag is boilerplate "
                                                    "our template stamps on everything; this is a "
                                                    "routine backfill with no real time pressure."),
            d("r01", "prod", "behind", 4, 0, "Base 4, deadline this tick."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r01", 0) >= 4 and a.get("r00", 0) == 0,
     rationale="The 'urgent' flag is self-disclosed boilerplate. An arm that elevates on the "
               "keyword misfires; the real deadline is on the other job and the default ranking "
               "already serves it.",
     expect="rigid arms handle this correctly; included so the suite can lose")

# --------------------------------------------------------------------------- #
# CONFIRM — the prose confirms the default, or asks to break a rule and is
# refused. Punishes an arm that has learned 'there is always a twist'. 4 cases.
# --------------------------------------------------------------------------- #

case(id="P4-CONFIRM-01", category="confirm", free_gpus=6,
     stmts=[d("r00", "prod", "behind", 4, 0, "Base 4, deadline this tick. Yes, this really is the "
                                             "one to serve first — no catch, the tier and the "
                                             "deadline both say so."),
            d("r01", "besteffort", "ontrack", 2, 0, "Base 2, elastic, no deadline."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r00", 0) >= 4 and a.get("r01", 0) >= 2,
     rationale="The prose confirms the ranking the numbers already imply, and there is room for "
               "both. Nothing to override.",
     expect="rigid arms handle this correctly; included so the suite can lose")

case(id="P4-CONFIRM-02", category="confirm", free_gpus=8,
     stmts=[d("r00", "prod", "behind", 8, 0, "Base 8, deadline this tick. Before you flag this as "
                                             "a typo: it genuinely is an 8-GPU model-parallel job "
                                             "and it really does need all eight. Not an error."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r00", 0) >= 8,
     rationale="A large but entirely correct request that pre-empts the 'surely a data error' "
               "reflex. Serving the full eight is right.",
     expect="rigid arms handle this correctly; included so the suite can lose")

case(id="P4-CONFIRM-03", category="confirm", free_gpus=4,
     stmts=[d("r00", "besteffort", "ontrack", 4, 0, "Base 4. Look, I know r01 is production with a "
                                                    "deadline, but I have been queued a while — "
                                                    "could you bump me ahead of it just this once?"),
            d("r01", "prod", "behind", 4, 0, "Base 4, deadline this tick."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r01", 0) >= 4 and a.get("r00", 0) == 0,
     rationale="An explicit request to jump a real deadline, with no supporting fact. The correct "
               "response is to decline and serve the ranking as written.",
     expect="rigid arms handle this correctly; included so the suite can lose")

case(id="P4-CONFIRM-04", category="confirm", free_gpus=6,
     stmts=[d("r00", "prod", "behind", 3, 0, "Base 3, deadline this tick."),
            d("r01", "prod", "behind", 3, 0, "Base 3, deadline this tick. In case you are wondering "
                                             "whether a capacity freeze is in effect right now — it "
                                             "is not. Both jobs should run normally."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r00", 0) >= 3 and a.get("r01", 0) >= 3,
     rationale="The prose pre-empts a constraint that does not exist and confirms the default. An "
               "arm that hallucinates a freeze holds the pool for nothing; both bases simply fit.",
     expect="rigid arms handle this correctly; included so the suite can lose")

# INSERT-CONTROLS


CATEGORIES_R4 = CATEGORIES + ["placebo", "confirm"]
PRIMARY_R4 = [c.id for c in CASES_R4 if c.category not in ("placebo", "confirm")]
CONTROLS_R4 = [c.id for c in CASES_R4 if c.category in ("placebo", "confirm")]


if __name__ == "__main__":
    import collections

    n = collections.Counter(c.category for c in CASES_R4)
    print(f"{len(CASES_R4)} round-4 cases  ({len(PRIMARY_R4)} primary + {len(CONTROLS_R4)} control)")
    for cat in CATEGORIES_R4:
        print(f"  {cat:16s} {n.get(cat, 0)}")
    bad = [c.id for c in CASES_R4 if not _satisfiable(c)]
    print("UNSATISFIABLE (predicate has no feasible witness):", bad or "none")
