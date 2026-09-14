# Frozen development protocol: policy debate v2.3 on all 24 training windows

**Frozen before the first v2.3 row is written.** This is an in-sample repair study: every rule below
was derived after inspecting the completed v1 results on these same 24 training windows. The run can
show whether v2.3 repairs those observed failures, but it cannot estimate generalisation and does not
touch the reserved validation or test splits.

## Inputs and comparator data

Run all 24 `split == "train"` windows in `pins/core_2x2_manifest.json`. Inputs are the existing frozen
worlds under `runs/core_2x2_worlds`; v2.3 uses a unique output tag and never edits `in/jobs.json` or
the completed `runs/sweep_train_debate/rows.jsonl` chain.

The saved comparators are:

- `policy_select` (single) and `policy_debate` v1 from the completed 24-window sweep;
- fixed `auction_deadline+adaptive` and `auction_fairness+adaptive` diagnostic controls;
- v2.0/v2.1 only as development history, not as complete comparators.

## Frozen v2.3 decision contract

- Demand ordering is deadline or production priority; Supply ordering is fairness or accumulated
  wait. Their sizing menus remain disjoint except for adaptive.
- Demand and Supply openings are issued concurrently. The Referee is issued only after both finish.
  A valid epoch therefore uses three calls; concurrency changes latency, not call count.
- Every pair of valid openings reaches the Referee. It may choose only from the component-wise
  cross-product of the two openings. Invalid openings fail to the configured deadline/adaptive floor.
- Queue trend is the current queue depth minus the previous selection epoch's depth. V2.2's
  within-epoch proxy is not used.
- Demand escalates at queue depth at least 8 if either production share is at least 0.10, or at least
  half the queue has consumed half its deadline budget and no user owns more than half of those
  threatened jobs.
- Supply concentration is not a veto by itself. Supply escalates only after two consecutive epochs
  where queue depth is at least 8, at least two users wait, one user owns at least half of queued jobs
  or accumulated wait, production share is below 0.10, no distributed deadline emergency exists,
  the current ordering is Demand-side, and the queue grew since the previous epoch.
- An escalated role's proposed ordering is binding until queue depth falls below 8. A Demand emergency
  can pre-empt an active Supply episode. The Referee still selects sizing.
- These constants and prompts must not change after the first row. Any successor is a new arm/version.

## Fixed run configuration

```text
--arm policy_debate_v2_3 --family mkt
--model qwen2.5:14b --temperature 0.1 --llm-seed 17 --num-predict 400
--sel-every 1800 --interval 300 --rcon-threshold-s 300
--fallback-ordering auction_deadline --fallback-sizing adaptive
```

Only one window runs at a time so windows do not contend for the inference server. Parallelism is
limited to the two independent advocate calls within an epoch.

## Frozen analysis

For every window report deadline-violation percentage and v2.3 minus single, v1, fixed deadline, and
fixed fairness. Summarise the paired v2.3-minus-single and v2.3-minus-v1 means, paired-t 95% intervals,
win/tie/loss counts, and exact sign-flip p-values as descriptive diagnostics only. Separately report:

- the four fixed-fairness-favouring windows (`d22h6`, `d223h11`, `d188h6`, `d102h8`);
- the 14 fixed-deadline-favouring windows and six fixed ties;
- Demand/Supply escalation epochs and triggers;
- ordering/sizing selections, invalid responses, LLM calls, inference errors, opening-round latency,
  and total wall time.

No window may be dropped, no rule may be retuned, and no result may be called validation evidence.
