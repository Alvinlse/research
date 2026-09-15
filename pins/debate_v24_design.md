# Policy debate v2.4 design

Status: implemented as a separate arm; do not start its evaluation sweep until the frozen v2.3
24-window sweep is complete and analysed.

Protocol version 2.4.1 fixes trial-start bookkeeping found by the first v2.4 audit: a requested
challenger is compared with the saved pre-debate incumbent. The ordinary Referee ordering updates
the incumbent only when no trial starts; an accepted trial updates it after observation. Frozen
2.4 result files remain attributable to commit `b6d0d6c`.

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

1. Demand proposes `auction_deadline` or `auction_priority` and independently votes
   `request_trial=true|false`.
2. Supply proposes `auction_fairness` or `auction_wait` and independently votes
   `request_trial=true|false`.
3. The Referee selects an advocate ordering and `trial_role=none|demand|supply`. A side is an
   allowed trial role only if it requested the trial and the safety gate says rotation is eligible.

The deterministic manager only validates those votes, starts the selected 30-minute trial, and
enforces emergency rollback. It contains no random or automatic branch selection.

## Result feedback

At trial completion, the manager records the selected role and ordering, baseline and ending queue
depth, baseline and ending deadline-pressure fraction, and whether the trial was accepted or rolled
back. Later Demand, Supply, and Referee calls receive:

- the three most recent raw outcomes; and
- an accumulated per-role scorecard containing completions, accepts, rollbacks, mean queue delta,
  and mean deadline-pressure delta.

This is in-context learning within one simulated window. Each evaluation window starts with empty
trial history so results cannot leak between test windows. Cross-window prompt revision remains a
separate analysis step after a complete frozen sweep.

## Trial decision

A trial is accepted when ending queue depth is no higher than baseline and deadline-pressure
fraction has not risen by more than 0.10. Otherwise the incumbent is restored. Material production
or broad severe deadline pressure immediately aborts a Supply-side trial and executes Demand's
ordering.

## Deterministic sizing hysteresis

- Enter `adaptive` at queue pressure >= 0.75.
- Return to `as_requested` only when queue pressure <= 0.35 and one-hour arrival load <= 0.50.
- Hold the current mode between those thresholds.

These thresholds and clocks must be frozen before a v2.4 comparison starts.
