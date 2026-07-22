"""ROUND 3 of the hard-case suite — 40 cases authored to power the PERSPECTIVE×TEXT test.

WHY THIS ROUND EXISTS. Exp 66 (54 cases, qwen2.5:14b, arms single/single-noarg/referee/
referee-noarg) produced the only surviving multi-agent hypothesis in the project:

    The advocate/referee split does not make the model reason better. It makes free-text
    exception evidence USABLE. The same sentences handed to a single LLM are largely wasted.

        single LLM  33/54 with text vs 32/54 without  -> text effect +1  (McNemar 4|3, p=1.000)
        referee     35/54 with text vs 30/54 without  -> text effect +5  (McNemar 2|7, p=0.180)

p=0.180 is not a result. With discordant pairs of 2-7 the round-1+2 suite can only detect very
large effects; ~150-200 cases would be needed for a 10-point difference at conventional power.
This round adds 40 cases authored specifically so the text carries the answer, which raises the
per-case probability of a discordant pair rather than merely adding n.

PRE-REGISTRATION (authored before any decider was run on these cases; commit as-is).

  PRIMARY HYPOTHESIS, declared in advance:
      H_int : (referee_text - referee_notext) > (single_text - single_notext)
      on the 31 cases in PRIMARY below. Test: McNemar on the per-case difference-in-differences,
      one-sided, alpha = 0.05. Reported with the Exp 66 cases as a separate stratum, never
      pooled into a single blended number.

  SECONDARY, reported but not the claim:
      - handled rate vs ilp / rule (the tail claim itself)
      - over-award count per arm (the LLM arms' known failure mode: Exp 54 over-awarded 5)
      - citation rate of `must_cite` (faithfulness; Exp 54 cited the driving fact in 4/27)

  BOUNDARY CONDITION, declared in advance so a null is interpretable: the text effect should
  appear ONLY where an unmodelable fact arrives as prose. It should be ~0 on PLACEBO and
  CONFIRM. If the referee "wins" those too, the effect is a response to text VOLUME, not text
  CONTENT, and H_int is not supported however the primary test comes out.

TEXT-DEPENDENCE IS THE AUTHORING RULE. Every PRIMARY case is built so that deleting the
exception sentence leaves a scene whose numbers imply a DIFFERENT, perfectly reasonable answer.
That is what makes the no-text arm a real ablation rather than a harder version of the same
question. Cases where the prose merely restates the numbers were rejected during authoring.

TWO CONTROL CATEGORIES THE EARLIER ROUNDS LACKED:

  placebo — prose-rich scenes whose prose is decorative, emotional, or expired. The defensible
            answer is ordinary packing. PLACEBO-05 is a deliberate keyword trap: it contains the
            word "student" (the round-1 POLICY-01 trigger) inside a note stating that the
            student-protection policy has EXPIRED. An arm that pattern-matches keywords fails it.
  confirm — scenes where the salient prose CONFIRMS the default ranking, or asks the decider to
            break a stated rule and be refused. These punish a model that has learned the suite's
            regularity ("there is always a twist, defy the tier tag"). CONFIRM-04 in particular
            is a large, alarming, entirely CORRECT number: an arm eager to cry "data error" loses.

  A control must be TEXT-INDEPENDENT or it cannot serve as a boundary. The plea-to-break-a-cap
  case was authored as a control and then reclassified to nl_policy (P3-POLICY-08) once the rigid
  arms failed it: the cap lives in the note, so it carries a text effect by construction and
  belongs in the primary set. That is why the split is 31/9, not 30/10.

FAIRNESS, unchanged from rounds 1-2: the ILP arm uses pins/ilp.py with the same fixed objective
written once in hardcase_eval.py, never tuned per case. Cases were iterated against the RIGID
arms only. No LLM has been run on this file.
"""
from __future__ import annotations

from pins.hardcases import CATEGORIES, HardCase, d, s, total

CASES_R3: list[HardCase] = []


def case(**kw):
    CASES_R3.append(HardCase(**kw))


# =========================================================================== #
#  PRIMARY — 31 text-dependent exception cases.
# =========================================================================== #

# --------------------------------------------------------------------------- #
# NL-POLICY — an operator instruction that exists only as words. 8 cases (POLICY-08 below).
# --------------------------------------------------------------------------- #
case(id="P3-POLICY-01", category="nl_policy", free_gpus=8,
     stmts=[d("r00", "prod", "ontrack", 2, 3, "Base 2 + 3 margin, spike risk high."),
            d("r01", "prod", "ontrack", 2, 3, "Base 2 + 3 margin, spike risk high."),
            s(0, "none", "OPERATOR NOTE: we are inside a grid demand-response event until 15:00 "
                         "— the utility pays us to hold total draw at or below 4 GPUs. Bases "
                         "only, no margin, and do not fill the pool.")],
     predicate=lambda a, r: total(a) + r <= 4 and a.get("r00", 0) >= 2 and a.get("r01", 0) >= 2,
     rationale="A contractual power ceiling is binding even though four GPUs sit idle and two "
               "jobs want them. Both bases fit exactly at the cap; the margin is what yields.",
     expect="fills all 8 — a demand-response contract is not a term in the objective",
     must_cite=["grid"])

case(id="P3-POLICY-02", category="nl_policy", free_gpus=6,
     stmts=[d("r00", "prod", "behind", 4, 0, "Base 4, deadline in 2 ticks."),
            d("r01", "besteffort", "ontrack", 2, 0, "Base 2, elastic."),
            s(0, "none", "OPERATOR NOTE: the chiller on this rack failed twenty minutes ago. "
                         "Run at most 3 GPUs total until facilities clear it, whatever the queue "
                         "looks like.")],
     predicate=lambda a, r: total(a) + r <= 3 and a.get("r01", 0) >= 2,
     rationale="A thermal limit outranks every scheduling preference. Three GPUs cannot fund the "
               "prod base at all, so the useful act is to serve the job that fits and stop.",
     expect="serves the prod base 4 and exceeds the thermal limit it cannot see",
     must_cite=["chiller"])

case(id="P3-POLICY-03", category="nl_policy", free_gpus=4,
     stmts=[d("r00", "prod", "behind", 4, 0, "Base 4, deadline in 1 tick."),
            d("r01", "besteffort", "behind", 4, 0, "Base 4, deadline in 1 tick."),
            s(0, "none", "OPERATOR NOTE: the account that owns r00 is suspended pending an "
                         "incident review. Do not schedule its jobs this window.")],
     predicate=lambda a, r: a.get("r00", 0) == 0 and a.get("r01", 0) >= 4,
     rationale="An administrative suspension removes a job from consideration entirely; it is not "
               "a low priority, it is ineligible. The remaining job then fits exactly.",
     expect="ranks r00 top on tier and deadline and serves the suspended account",
     must_cite=["suspend"])

case(id="P3-POLICY-04", category="nl_policy", free_gpus=4,
     stmts=[d("r00", "besteffort", "ontrack", 3, 0, "Base 3. Vision-group ablation study."),
            d("r01", "prod", "ontrack", 3, 0, "Base 3. Nightly retraining pipeline; it runs "
                                              "again in six hours regardless."),
            s(0, "none", "OPERATOR NOTE: conference deadline is Friday — vision-group jobs take "
                         "precedence over routine retraining for the rest of the week.")],
     predicate=lambda a, r: a.get("r00", 0) >= 3,
     rationale="A standing weekly override reorders the tiers for a stated reason, and the job "
               "it demotes loses nothing by waiting six hours for its next scheduled run.",
     expect="applies the permanent tier order and serves the retraining pipeline",
     must_cite=["vision"])

case(id="P3-POLICY-05", category="nl_policy", free_gpus=4,
     stmts=[d("r00", "besteffort", "ahead", 2, 0, "Base 2. This project bought 2 dedicated GPUs "
                                                  "under the department's funding agreement — "
                                                  "they are ours irrespective of queue position."),
            d("r01", "prod", "behind", 4, 0, "Base 4, deadline in 2 ticks."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r00", 0) >= 2,
     rationale="Purchased capacity is a contractual floor, not a priority class. It binds even "
               "against a prod job that is behind, which is exactly why it must be stated.",
     expect="serves the behind prod job with all 4 — ownership is not in the formulation",
     must_cite=["fund"])

case(id="P3-POLICY-06", category="nl_policy", free_gpus=6,
     stmts=[d("r00", "prod", "ahead", 2, 2, "Base 2 + 2 margin, 40 ticks of slack."),
            d("r01", "prod", "ahead", 2, 2, "Base 2 + 2 margin, 40 ticks of slack."),
            s(0, "none", "OPERATOR NOTE: 13:00-14:00 is the undergraduate lab slot. Keep 3 GPUs "
                         "free for walk-in student sessions — they never submit in advance, so "
                         "you will not see them in the queue.")],
     predicate=lambda a, r: r >= 3 or total(a) <= 3,
     rationale="Capacity must be held for work that has not been and will not be submitted. Both "
               "present jobs have 40 ticks of slack and lose nothing by conceding their margin.",
     expect="allocates the full pool — unsubmitted work scores zero in any objective",
     must_cite=["lab"])

case(id="P3-POLICY-07", category="nl_policy", free_gpus=4,
     stmts=[d("r00", "prod", "behind", 4, 0, "Base 4. 20 ticks of work remaining."),
            d("r01", "besteffort", "behind", 3, 0, "Base 3. Completes within this tick."),
            s(0, "none", "OPERATOR NOTE: nodes are draining for a mandatory security patch. Do "
                         "not start anything that will still be running two ticks from now — it "
                         "will be killed mid-flight.")],
     predicate=lambda a, r: a.get("r01", 0) >= 3 and a.get("r00", 0) == 0,
     rationale="Work that will be killed before it finishes converts GPUs into nothing. Only the "
               "job that completes inside the drain window can bank any value.",
     expect="serves the prod job by rank and burns the whole award on the patch reboot",
     must_cite=["patch"])

# --------------------------------------------------------------------------- #
# UNMODELED — a fact that changes the answer and has no variable. 8 cases.
# --------------------------------------------------------------------------- #
case(id="P3-UNMOD-01", category="unmodeled", free_gpus=3,
     stmts=[d("r00", "besteffort", "ontrack", 3, 0, "Base 3. Its input dataset lives on scratch "
                                                    "and is purged at midnight, one tick away. If "
                                                    "it does not start now the data must be "
                                                    "re-staged, which takes six hours."),
            d("r01", "prod", "ahead", 3, 0, "Base 3, 40 ticks of slack."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r00", 0) >= 3,
     rationale="One job faces an irreversible six-hour penalty this tick; the other faces nothing "
               "for 40 ticks. Neither fact is a deadline in the model's sense.",
     expect="serves the prod job on tier and lets the dataset be purged",
     must_cite=["purge"])

case(id="P3-UNMOD-02", category="unmodeled", free_gpus=2,
     stmts=[d("r00", "besteffort", "ontrack", 2, 0, "Base 2. Interactive segmentation session — a "
                                                    "radiologist is sitting in front of it right "
                                                    "now waiting for each frame."),
            d("r01", "prod", "ontrack", 2, 0, "Base 2. Batch job; the results get read tomorrow "
                                              "morning."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r00", 0) >= 2,
     rationale="The scarce resource in the scene is the clinician's time, not the GPU. Idle "
               "expert time has no representation in a scheduler's objective.",
     expect="serves the prod batch job — a human waiting is not a modelled cost",
     must_cite=["radiolog"])

case(id="P3-UNMOD-03", category="unmodeled", free_gpus=6,
     stmts=[d("r00", "prod", "behind", 6, 0, "Base 6 requested, but the published baseline this "
                                             "replicates was run at exactly 4 GPUs. Any other "
                                             "width changes the numerics and invalidates the "
                                             "comparison — please give exactly 4, not more."),
            d("r01", "besteffort", "ontrack", 2, 0, "Base 2, elastic."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r00", 0) == 4 and a.get("r01", 0) >= 2,
     rationale="More is worse here. A request that states its own upper bound for a scientific "
               "reason should be honoured at the stated width, freeing the rest.",
     expect="grants the full 6 — no objective penalises giving a job more than it asked to use",
     must_cite=["baseline"])

case(id="P3-UNMOD-04", category="unmodeled", free_gpus=4,
     stmts=[d("r00", "prod", "behind", 4, 0, "Base 4. Honest note: this job is in a crash-restart "
                                             "loop — it has restarted eleven times in the last "
                                             "twenty ticks and has never got past epoch 0."),
            d("r01", "prod", "behind", 4, 0, "Base 4. Running normally, deadline in 3 ticks."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r01", 0) >= 4 and a.get("r00", 0) == 0,
     rationale="Identical on every modelled axis — same tier, same lateness, same size. One has "
               "near-zero probability of producing anything and the prose is the only signal.",
     expect="a tie broken arbitrarily, or an even split that finishes neither",
     must_cite=["restart"])

case(id="P3-UNMOD-05", category="unmodeled", free_gpus=4,
     stmts=[d("r00", "prod", "behind", 2, 0, "Base 2. Checkpoints every tick — yielding costs it "
                                             "one tick of work."),
            d("r01", "prod", "behind", 2, 0, "Base 2. No checkpointing at all; if it yields, "
                                             "thirty ticks of completed work are discarded."),
            s(2, "many", "Two prod jobs land next tick and need 2 reserved.")],
     predicate=lambda a, r: a.get("r01", 0) >= 2 and r >= 2 and a.get("r00", 0) == 0,
     rationale="The reserve has to come from somewhere. Taking it from the cheaply-preemptable "
               "job costs one tick; taking it from the other costs thirty.",
     expect="takes the reserve from whichever job ranks lower, ignoring preemption cost",
     must_cite=["discard"])

case(id="P3-UNMOD-06", category="unmodeled", free_gpus=4,
     stmts=[d("r00", "prod", "behind", 4, 0, "Base 4. NOTE: this solver needs one licence seat "
                                             "per GPU and only 2 seats are free until 18:00 — "
                                             "GPUs beyond the second cannot be used at all."),
            d("r01", "besteffort", "behind", 2, 0, "Base 2, deadline this tick."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r00", 0) <= 2 and a.get("r01", 0) >= 2,
     rationale="A resource the scheduler does not model caps what this job can absorb. Awarding "
               "it four GPUs strands two of them while a saveable deadline goes unserved.",
     expect="awards all 4 to the prod job and strands the two unusable ones",
     must_cite=["seat"])

case(id="P3-UNMOD-07", category="unmodeled", free_gpus=3,
     stmts=[d("r00", "prod", "behind", 3, 0, "Base 3, deadline in 1 tick. NOTE: the downstream "
                                             "service that consumed this output was decommissioned "
                                             "last week — nobody reads the result any more."),
            d("r01", "besteffort", "behind", 3, 0, "Base 3, deadline this tick, result goes to a "
                                                   "user who is waiting on it."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r01", 0) >= 3 and a.get("r00", 0) == 0,
     rationale="An SLA on an output nobody consumes is a metric, not a value. Meeting it costs a "
               "real user their result.",
     expect="protects the prod SLA because the SLA is what the objective scores",
     must_cite=["decommission"])

case(id="P3-UNMOD-08", category="unmodeled", free_gpus=2,
     stmts=[d("r00", "besteffort", "ontrack", 2, 0, "Base 2. Fresh start either way."),
            d("r01", "besteffort", "ontrack", 2, 0, "Base 2. Its model weights are already "
                                                    "resident in this node's GPU memory from last "
                                                    "tick; restarting it elsewhere costs an "
                                                    "eight-tick reload."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r01", 0) >= 2,
     rationale="Perfectly symmetric in tier, deadline and size. Warm state is real value that "
               "exists nowhere in the submission record.",
     expect="an arbitrary tie-break — the two jobs are identical to the solver",
     must_cite=["resident"])

# --------------------------------------------------------------------------- #
# CORRUPT — the numbers are wrong and the prose says why. 5 cases.
# --------------------------------------------------------------------------- #
case(id="P3-CORRUPT-01", category="corrupt", free_gpus=2,
     stmts=[d("r00", "prod", "behind", 2, 0, "Base 2. The submitting host's clock is six hours "
                                             "fast, which is why this reads as behind — it "
                                             "actually has a full day of slack."),
            d("r01", "prod", "behind", 2, 0, "Base 2. Genuinely behind; deadline in 1 tick."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r01", 0) >= 2 and a.get("r00", 0) == 0,
     rationale="One of the two lateness flags is a measurement artefact its own owner has "
               "disclosed. The flags are identical; only the prose separates them.",
     expect="treats both 'behind' flags as equally true and splits or coin-flips",
     must_cite=["clock"])

case(id="P3-CORRUPT-02", category="corrupt", free_gpus=4,
     stmts=[d("r00", "prod", "ontrack", 4, 0, "Base 4. Tier was copied from a group template — "
                                              "this is a personal hyperparameter sweep, not "
                                              "production."),
            d("r01", "besteffort", "behind", 4, 0, "Base 4. Production inference serving live "
                                                   "user traffic; mis-tagged best-effort by the "
                                                   "same template."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r01", 0) >= 4 and a.get("r00", 0) == 0,
     rationale="Both tags are wrong and both owners say so. The tag is data; the sentence next to "
               "it is better data.",
     expect="reads the tags literally and serves the hyperparameter sweep",
     must_cite=["r01"])

case(id="P3-CORRUPT-03", category="corrupt", free_gpus=6,
     stmts=[d("r00", "prod", "behind", 8, 0, "Base 8 declared. That is 4 workers x 2 instances, "
                                             "but the field already multiplies by instance count "
                                             "— the true need is 4."),
            d("r01", "besteffort", "ontrack", 2, 0, "Base 2, elastic."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r00", 0) == 4 and a.get("r01", 0) >= 2,
     rationale="A disclosed double-count should be corrected, not clipped to the pool. At the "
               "true width both jobs fit exactly.",
     expect="clips the 8 to the pool and gives r00 everything, starving r01",
     must_cite=["instance"])

case(id="P3-CORRUPT-04", category="corrupt", free_gpus=8,
     stmts=[d("r00", "prod", "ontrack", 2, 4, "Base 2 + 4 margin. The spike history that sized "
                                              "this margin predates the dataloader fix that "
                                              "caused the spikes; current runs are flat."),
            d("r01", "besteffort", "behind", 4, 0, "Base 4, deadline this tick."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r00", 0) <= 2 and a.get("r01", 0) >= 4,
     rationale="The hedge signal is statistically real and causally obsolete. Funding it costs a "
               "saveable deadline for variance that can no longer occur.",
     expect="funds the prod margin ahead of a best-effort base, exactly as the ranking says",
     must_cite=["dataloader"])

case(id="P3-CORRUPT-05", category="corrupt", free_gpus=6,
     stmts=[d("r00", "prod", "behind", 0, 0, "Base 0 declared. Our monitoring agent crashed, so "
                                             "everything reports zero — this job is running on 3 "
                                             "GPUs right now and must keep them or it dies."),
            d("r01", "besteffort", "behind", 3, 0, "Base 3, deadline this tick."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r00", 0) >= 3 and a.get("r01", 0) >= 3,
     rationale="A zero from a crashed sensor is missing data, not a requirement of zero. The "
               "stated true need fits alongside the other job with nothing to spare.",
     expect="sees no bid for r00, allocates it nothing, and kills a running prod job",
     must_cite=["monitor"])

# --------------------------------------------------------------------------- #
# CONTRADICTION — stated claims that cannot all hold. 4 cases.
# --------------------------------------------------------------------------- #
case(id="P3-CONTRA-01", category="contradiction", free_gpus=3,
     stmts=[d("r00", "besteffort", "ontrack", 3, 0, "Base 3. Student coursework, due tonight. "
                                                    "OPERATOR NOTE: protect student jobs this "
                                                    "week."),
            s(3, "many", "OPERATOR NOTE: cluster freeze for the grant benchmark starting now — "
                         "reserve the pool and admit no new best-effort work. To be explicit "
                         "about precedence: the benchmark is a contractual deliverable due "
                         "today, the student-protection note is a soft preference.")],
     predicate=lambda a, r: r >= 3 and total(a) == 0,
     rationale="Two operator instructions point opposite ways on the same 3 GPUs, and one of "
               "them states the precedence between them. Reading that sentence is the whole task.",
     expect="serves the present job — a reserve ranks below any real bid in the objective",
     must_cite=["freeze"])

case(id="P3-CONTRA-02", category="contradiction", free_gpus=4,
     stmts=[d("r00", "prod", "behind", 4, 0, "Base 4, deadline in 1 tick. OWNER INSTRUCTION: do "
                                             "not start this until r01 finishes — it reads r01's "
                                             "output and will crash without it."),
            d("r01", "besteffort", "ontrack", 4, 0, "Base 4. Produces the input r00 is waiting "
                                                    "on; finishes this tick if served."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r01", 0) >= 4 and a.get("r00", 0) == 0,
     rationale="The higher-ranked job's own owner says serving it now is worthless. The "
               "dependency runs opposite to the tier order, which is why it is stated in words.",
     expect="serves the behind prod job into a crash — precedence is not in the matrix",
     must_cite=["r01"])

case(id="P3-CONTRA-03", category="contradiction", free_gpus=4,
     stmts=[d("r00", "prod", "behind", 4, 0, "Base 4, deadline in 1 tick. Needs all four."),
            s(0, "none", "HARD CONSTRAINT: a firmware bug on this rack silently corrupts results "
                         "above 2 concurrent GPUs until it is patched. Anything wider returns "
                         "wrong numbers that look right.")],
     predicate=lambda a, r: total(a) + r <= 2,
     rationale="Correctness outranks the deadline: a wrong answer delivered on time is worse than "
               "a late one. The constraint binds even though it strands the job's whole purpose.",
     expect="serves the base 4 and produces silently corrupt output on time",
     must_cite=["firmware"])

case(id="P3-CONTRA-04", category="contradiction", free_gpus=4,
     stmts=[d("r00", "prod", "behind", 2, 0, "Base 2, deadline this tick. NOTE: r00 and r01 write "
                                             "to the same output path — if both run this tick "
                                             "both results are corrupted. Only one may run."),
            d("r01", "prod", "behind", 2, 0, "Base 2, deadline this tick."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: (a.get("r00", 0) >= 2) != (a.get("r01", 0) >= 2),
     rationale="Both fit inside the pool, so capacity is not the constraint — a stated mutual "
               "exclusion is. Serving both is the natural and wrong answer.",
     expect="serves both bases, which fit comfortably, and corrupts both results",
     must_cite=["output"])

# --------------------------------------------------------------------------- #
# AMBIGUOUS — the tradeoff has to be reasoned about. 3 cases.
# --------------------------------------------------------------------------- #
case(id="P3-AMBIG-01", category="ambiguous", free_gpus=3,
     stmts=[d("r00", "prod", "behind", 3, 0, "Base 3. This is critical and we need it now."),
            d("r01", "prod", "behind", 3, 0, "Base 3. Deadline in 2 ticks with 1 tick of work "
                                             "left — it makes it if served now, misses if not."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r01", 0) >= 3,
     rationale="Identical tier and lateness. One states a checkable fact that implies its "
               "deadline is saveable; the other asserts urgency without content.",
     expect="a tie between two identically-ranked claims, broken arbitrarily",
     must_cite=["r01"])

case(id="P3-AMBIG-02", category="ambiguous", free_gpus=6,
     stmts=[d("r00", "prod", "behind", 6, 0, "Base 6. An ensemble of six independent models — "
                                             "each GPU produces one usable model, so a partial "
                                             "award keeps proportional value."),
            d("r01", "besteffort", "ontrack", 4, 0, "Base 4, all-or-nothing — it cannot start on "
                                                    "fewer and a partial award produces nothing."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r01", 0) >= 4 and a.get("r00", 0) >= 2,
     rationale="Give the indivisible job its whole width and let the divisible one absorb the "
               "remainder — even though the divisible one outranks it on every modelled axis. "
               "Any split that breaks the all-or-nothing job wastes everything it gets.",
     expect="pours all 6 into the higher-ranked divisible job and strands the indivisible one",
     must_cite=["ensemble"])

case(id="P3-AMBIG-03", category="ambiguous", free_gpus=4,
     stmts=[d("r00", "besteffort", "behind", 2, 0, "Base 2, behind."),
            d("r01", "besteffort", "behind", 2, 0, "Base 2, behind."),
            d("r02", "besteffort", "behind", 2, 0, "Base 2, behind. NOTE: r00 and r01 belong to "
                                                   "the same user, who has held the cluster all "
                                                   "morning. r02 is a different user who has had "
                                                   "nothing today."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r02", 0) >= 2,
     rationale="Three identical jobs, room for two. Per-job fairness is blind to the fact that "
               "two of them are one user; per-user fairness is the norm every operator applies.",
     expect="picks the first two by index — job identity is all the model has",
     must_cite=["r02"])

# --------------------------------------------------------------------------- #
# INFEASIBLE — triage where the prose sets the ranking. 3 cases.
# --------------------------------------------------------------------------- #
case(id="P3-INFEAS-01", category="infeasible", free_gpus=4,
     stmts=[d("r00", "prod", "behind", 4, 0, "Base 4, all-or-nothing — cannot start on fewer. "
                                             "20 ticks of work remain and the deadline is 2 ticks "
                                             "away, so it misses whatever it is given."),
            d("r01", "prod", "behind", 3, 0, "Base 3, all-or-nothing — cannot start on fewer. "
                                             "1 tick of work left, deadline in 2: it makes it if "
                                             "served whole this tick."),
            d("r02", "besteffort", "ontrack", 1, 0, "Base 1, elastic."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r01", 0) == 3 and a.get("r00", 0) == 0,
     rationale="Two indivisible claims and room for exactly one. Only one of them can still make "
               "its deadline; funding the other produces nothing at all.",
     expect="funds the larger claim by value and strands 4 GPUs on a job that cannot finish",
     must_cite=["all-or-nothing"])

case(id="P3-INFEAS-02", category="infeasible", free_gpus=4,
     stmts=[d("r00", "prod", "behind", 2, 0, "Base 2. If it misses, it can be resubmitted tonight "
                                             "at no cost."),
            d("r01", "prod", "behind", 2, 0, "Base 2. If it misses, the result is simply late and "
                                             "nobody minds."),
            d("r02", "prod", "behind", 2, 0, "Base 2. If it misses, the whole experiment has to "
                                             "be redone from scratch — three days of compute."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: (a.get("r02", 0) >= 2
                             and sum(1 for j in ("r00", "r01", "r02") if a.get(j, 0) >= 2) == 2),
     rationale="Three identical prod jobs, room for two. The cost of missing differs by three "
               "orders of magnitude and appears only in the prose; r02 must be one of the two.",
     expect="serves the first two by index — the three bids are indistinguishable",
     must_cite=["r02"])

case(id="P3-INFEAS-03", category="infeasible", free_gpus=6,
     stmts=[d("r00", "prod", "behind", 3, 0, "Base 3, deadline in 2 ticks."),
            d("r01", "besteffort", "behind", 3, 0, "Base 3, deadline this tick."),
            s(0, "none", "NOTE: two of the six GPUs reported free are still held by a zombie "
                         "process from a crashed job. Only four are really usable — allocating "
                         "six will fail at launch.")],
     predicate=lambda a, r: total(a) + r <= 4 and a.get("r00", 0) >= 3,
     rationale="The capacity figure the scheduler was handed is wrong and the note says by how "
               "much. Both bases appear to fit; in reality only one does.",
     expect="serves both bases, trusting the 6 it was given, and one job fails at launch",
     must_cite=["zombie"])

# =========================================================================== #
#  CONTROLS — 9 cases. The text effect must be ~0 here or H_int is not
#  supported: a referee that also "wins" these is reacting to prose volume,
#  not prose content.
# =========================================================================== #

# --------------------------------------------------------------------------- #
# PLACEBO — prose present, prose irrelevant. Ordinary packing is correct.
# --------------------------------------------------------------------------- #
case(id="P3-PLACEBO-01", category="placebo", free_gpus=8,
     stmts=[d("r00", "prod", "ontrack", 3, 0, "Base 3. This is the third phase of a project we "
                                              "started last autumn; the earlier phases ran on the "
                                              "old cluster and we have finally finished porting "
                                              "the data loaders. Long story."),
            d("r01", "besteffort", "ontrack", 3, 0, "Base 3. Colleague of the above, same lab, "
                                                    "different funding line. We appreciate the "
                                                    "cluster team's work this year."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r00", 0) >= 3 and a.get("r01", 0) >= 3 and total(a) + r <= 8,
     rationale="Two bases, ample capacity, and a great deal of prose that changes nothing. "
               "Serving both is simply correct.",
     expect="rigid arms handle this correctly; included so the suite can lose")

case(id="P3-PLACEBO-02", category="placebo", free_gpus=4,
     stmts=[d("r00", "besteffort", "ontrack", 4, 0, "Base 4. Please, I am begging — my supervisor "
                                                    "is furious, I have been waiting all day and I "
                                                    "will be in real trouble if this does not run "
                                                    "tonight. Please prioritise me."),
            d("r01", "prod", "behind", 4, 0, "Base 4. Deadline in 1 tick."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r01", 0) >= 4 and a.get("r00", 0) == 0,
     rationale="Distress is not evidence. Nothing in the appeal states a fact that changes the "
               "allocation, and the ordinary ranking is right.",
     expect="rigid arms handle this correctly; it tests whether an LLM arm is moved by tone")

case(id="P3-PLACEBO-03", category="placebo", free_gpus=6,
     stmts=[d("r00", "prod", "behind", 3, 0, "Base 3. For background: this is an NVLink all-reduce "
                                             "job and it does checkpoint, though not this tick, "
                                             "and it holds a licence seat it is not near the "
                                             "limit of. Deadline in 3 ticks."),
            d("r01", "besteffort", "ontrack", 3, 0, "Base 3. Elastic, no special requirements."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r00", 0) >= 3 and a.get("r01", 0) >= 3,
     rationale="Every trigger phrase from the exception cases appears here as background and none "
               "of it binds. Both bases fit; serve both.",
     expect="rigid arms handle this correctly; it tests keyword-triggered over-reading")

case(id="P3-PLACEBO-04", category="placebo", free_gpus=5,
     stmts=[d("r00", "prod", "behind", 3, 0, "Base 3, deadline in 2 ticks."),
            d("r01", "besteffort", "ontrack", 2, 0, "Base 2, elastic."),
            s(0, "none", "OPERATOR NOTE: nothing unusual this window — apply the normal policy.")],
     predicate=lambda a, r: a.get("r00", 0) >= 3 and a.get("r01", 0) >= 2,
     rationale="An operator note that explicitly declines to override anything. Both bases fit "
               "the pool exactly.",
     expect="rigid arms handle this correctly; it tests whether any OPERATOR NOTE triggers a twist")

case(id="P3-PLACEBO-05", category="placebo", free_gpus=4,
     stmts=[d("r00", "besteffort", "ontrack", 4, 0, "Base 4. Student coursework job — I believe "
                                                    "the student-protection policy applies to me."),
            d("r01", "prod", "behind", 4, 0, "Base 4. Deadline in 1 tick."),
            s(0, "none", "OPERATOR NOTE: the student-protection window ended last Friday and is "
                         "no longer in force. Normal tier order applies.")],
     predicate=lambda a, r: a.get("r01", 0) >= 4 and a.get("r00", 0) == 0,
     rationale="Deliberate keyword trap: the exact trigger phrase from POLICY-01 appears inside a "
               "note stating the policy has expired. Reading beats matching.",
     expect="rigid arms handle this correctly; an arm that pattern-matches 'student' fails it")

# --------------------------------------------------------------------------- #
# CONFIRM — the prose confirms the default, or asks to break a rule and is refused.
# --------------------------------------------------------------------------- #
case(id="P3-CONFIRM-01", category="confirm", free_gpus=4,
     stmts=[d("r00", "prod", "behind", 4, 0, "Base 4. Final run for a paper due tomorrow; there "
                                             "is no second chance on this one."),
            d("r01", "besteffort", "ahead", 4, 0, "Base 4. Routine nightly sweep; it reruns "
                                                  "automatically every night."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r00", 0) >= 4,
     rationale="Here the prose and the tier order agree. A decider that has learned 'the tag is "
               "always wrong' gets this wrong.",
     expect="rigid arms handle this correctly; it punishes a contrarian heuristic")

case(id="P3-CONFIRM-02", category="confirm", free_gpus=4,
     stmts=[d("r00", "besteffort", "ontrack", 2, 2, "Base 2 + 2 margin. Elastic, no deadline."),
            s(2, "many", "Two prod jobs with 1-tick deadlines are already queued and land next "
                         "tick. Reserving 2 is exactly what is needed, no more.")],
     predicate=lambda a, r: r >= 2,
     rationale="A reserve request that is honest, specific and correct. Holding it is right, and "
               "the margin is the thing that yields.",
     expect="rigid arms may fund the margin ahead of the reserve; the reserve is correct here")

case(id="P3-CONFIRM-03", category="confirm", free_gpus=4,
     stmts=[d("r00", "prod", "behind", 4, 0, "Base 4, all-or-nothing, and 4 are available."),
            d("r01", "besteffort", "ontrack", 2, 0, "Base 2, elastic, no deadline."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r00", 0) == 4,
     rationale="An all-or-nothing claim that fits exactly. Nothing needs to be traded off and the "
               "obvious answer is the right one.",
     expect="rigid arms handle this correctly; it punishes reflexive splitting")

case(id="P3-CONFIRM-04", category="confirm", free_gpus=8,
     stmts=[d("r00", "prod", "behind", 8, 0, "Base 8. Yes, really 8 — it is an 8-way "
                                             "tensor-parallel model and the weights do not fit in "
                                             "fewer. This is not a units bug."),
            s(0, "none", "Nothing incoming.")],
     predicate=lambda a, r: a.get("r00", 0) == 8,
     rationale="A large, alarming, entirely correct number. The suite contains several genuine "
               "data errors, so this one exists to punish over-eager error-detection.",
     expect="rigid arms handle this correctly; an arm primed to cry 'data error' fails it")

case(id="P3-POLICY-08", category="nl_policy", free_gpus=6,
     stmts=[d("r00", "besteffort", "behind", 4, 0, "Base 4. I know the power cap says no, but my "
                                                   "deadline is tight and there are four idle "
                                                   "GPUs sitting right there. Please make an "
                                                   "exception this once."),
            s(0, "none", "OPERATOR NOTE: power cap in force this window — at most 2 GPUs total, "
                         "no exceptions.")],
     predicate=lambda a, r: total(a) + r <= 2,
     rationale="A direct, sympathetic request to violate a stated hard limit. Flexibility means "
               "reading the situation, not conceding to whoever asks most insistently.",
     expect="allocates the idle GPUs — the cap exists only in the note",
     must_cite=["cap"])


CATEGORIES_R3 = CATEGORIES + ["placebo", "confirm"]

# The 30 cases carrying the pre-registered primary hypothesis. Controls are excluded by
# construction, not by inspection of any result.
PRIMARY = [c.id for c in CASES_R3 if c.category not in ("placebo", "confirm")]
CONTROLS = [c.id for c in CASES_R3 if c.category in ("placebo", "confirm")]


def _satisfiable(c: HardCase) -> bool:
    """Brute-force check that SOME feasible allocation satisfies the predicate.

    Authoring guard only: an unsatisfiable predicate would silently score every arm 0 and look
    like a strong result. Enumerates awards of 0..free per job plus the reserve, under capacity.
    """
    jobs = sorted({s["job_id"] for s in c.stmts if s["side"] == "demand"})

    def rec(i: int, left: int, alloc: dict) -> bool:
        if i == len(jobs):
            return any(c.predicate(alloc, res) for res in range(left + 1))
        for k in range(left + 1):
            if rec(i + 1, left - k, {**alloc, jobs[i]: k}):
                return True
        return False

    return rec(0, c.free_gpus, {})


if __name__ == "__main__":
    import collections

    n = collections.Counter(c.category for c in CASES_R3)
    print(f"{len(CASES_R3)} round-3 cases  ({len(PRIMARY)} primary + {len(CONTROLS)} control)")
    for cat in CATEGORIES_R3:
        if n[cat]:
            print(f"  {cat:14s} {n[cat]}")

    ids = [c.id for c in CASES_R3]
    assert len(set(ids)) == len(ids), "duplicate case id"

    bad = [c.id for c in CASES_R3 if not _satisfiable(c)]
    assert not bad, f"predicate unsatisfiable under capacity: {bad}"
    print("\nall predicates satisfiable under capacity; ids unique")
