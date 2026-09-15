# Policy debate v2.4 design

Status: protocol 2.4.3 is implemented for focused comparison. Frozen 2.4 results are attributable
to commit `b6d0d6c`; the 2.4.1 trial-start diagnostic is attributable to `956d289`; and the completed
2.4.2 pilot and its frozen-gate analysis are attributable to `df531df`.

## Decision clocks

- Ordering is debated every 1,800 simulated seconds. Demand and Supply openings execute in
  parallel, followed by one Referee call.
- Sizing is reviewed deterministically every 300 simulated seconds. No LLM schema contains a
  sizing or GPU-count output.
- Actual malleable-job allocation changes remain restricted to ElastiSim work boundaries and the
  existing resize legality, cooldown, and budget checks.

## LLM-decided rotation

A rotation can become eligible once 3,600 simulated seconds have elapsed since the last trial,
provided the queue has at least eight jobs from at least two users, deadline-pressure fraction is
below 0.25, and production share is below 0.10.

Eligibility never selects a branch. During the ordinary ordering debate:

1. Demand proposes `auction_deadline`, `auction_priority`, or `abstain` and independently votes
   `request_trial=true|false`.
2. Supply proposes `auction_fairness`, `auction_wait`, or `abstain` and independently votes
   `request_trial=true|false`.
3. The Referee chooses `hold`, `trial_demand`, or `trial_supply`. A trial action is allowed only
   when that advocate requested it, supplied a non-abstaining candidate different from the
   incumbent, and the safety gate says rotation is eligible.

Hold and invalid output retain the incumbent. The deterministic manager only validates votes,
starts exposure, and enforces probation/rollback. Rotation is the only non-emergency branch-change
path; the manager contains no random or automatic branch selection.

## Result feedback

At trial completion, the manager records the selected role and ordering, baseline and ending queue
depth, baseline and ending deadline-pressure fraction, and whether the trial was accepted or rolled
back. It settles an expired phase before inference so the next Demand, Supply, and Referee calls
receive:

- the three most recent raw outcomes; and
- an accumulated per-role scorecard; and
- a transition scorecard keyed by the exact incumbent-to-challenger pair.

Protocol 2.4.3 calculates learning deltas from the trial or probation phase whose checks produced
the verdict. A rollback explicitly means `harmful_challenger`; only an acceptance contributes to
positive aggregate evidence. Two consecutive rollbacks of an exact transition remove it from
allowed Referee actions for four simulated hours. The reverse transition remains independent.

This is in-context learning within one simulated window. Each evaluation window starts with empty
trial history so results cannot leak between test windows. Cross-window prompt revision remains a
separate analysis step after a complete frozen sweep.

## Trial decision

A 30-minute first phase must pass queue-trend, deadline-pressure, and p90-wait non-regression checks
plus its role objective: Supply must improve queue drainage or concentration; Demand must improve
queue drainage or deadline pressure. A pass enters a second 30-minute probation with a fresh
baseline. Only a second pass updates the incumbent. Failure restores the previous incumbent, and
the one-hour cooldown starts when exposure ends. Material production or broad severe deadline
pressure immediately aborts a Supply trial and executes Demand's ordering.

## Deterministic sizing hysteresis

- Enter `adaptive` at queue pressure >= 0.75.
- Return to `as_requested` only when queue pressure <= 0.35 and one-hour arrival load <= 0.50.
- Hold the current mode between those thresholds.

These thresholds and clocks must be frozen before a v2.4 comparison starts.
