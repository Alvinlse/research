# Pre-registration: bounded debate vs single selector across the training split

**Frozen before the first row is written.** Exploratory, not confirmatory: this sweep does not touch
the reserved test split and makes no claim on it. Its purpose is to decide whether the load-dependent
gap seen on two windows survives the full training set, and therefore whether a confirmatory test of
reasoning structure is worth pre-registering at all.

## Windows

All 24 windows with `split == "train"` in `pins/core_2x2_manifest.json`, offered load 0.32x to 4.99x.
The reserved 12 test windows and the 12 validation windows are excluded and must not be run under this
protocol.

## Arms

Exactly two, paired within window:

- `policy_select` — the single referee.
- `policy_debate` — the bounded debate of the architecture section, advocates stating a policy and
  code deriving the move.

The deterministic floor is not an arm here. It is already characterised on both splits by the 32
deterministic controls, and this comparison is between reasoning structures.

## Fixed configuration

```
--family mkt --model qwen2.5:14b --temperature 0.1 --llm-seed 17 --num-predict 400
--sel-every 1800 --interval 300 --rcon-threshold-s 300
--fallback-ordering auction_deadline --fallback-sizing adaptive
```

One inference seed (17). Decoding noise is therefore inseparable from between-window variance, exactly
as in the amended core factorial.

## Primary outcome and analysis

`deadline_viol_pct`. The inferential unit is the window; the statistic is the paired difference
debate minus single, so a negative value favours debate.

Exactly two tests are declared, and no others may be added after the first row exists:

1. **Primary.** Two-sided exact paired sign-flip randomisation test on all 24 paired differences.
2. **Subgroup.** The same test within the high-contention stratum, offered load >= 2.5x, declared
   here because the two-window pilot found the effect only under contention. This stratum contains 8
   of the 24 training windows.

Holm correction across those two tests. Reported alongside: the paired mean difference with a
paired-t 95% interval, the win/tie/loss counts, and the per-window table.

## Secondary descriptive, not tested

Mean and p90 waiting time, SLA-2, useful utilisation, LLM calls and wall time; and the mechanism
counters that the pilot used to explain the gap — selection concentration (epochs spent on each arm's
modal policy), opening-agreement rate, referee calls, and rejected outputs. These are descriptive
evidence for the stability-rather-than-argument reading and carry no test.

## Stopping and amendments

The chain is resumable and append-only; a window contributes to the analysis only when both arms have
a row. Any deviation is recorded as a dated amendment in this file rather than absorbed, following the
core factorial's convention.

## Amendment 1 (2026-09-14): budget-matched control, second phase

Added after the first eight rows existed, at the author's request and disclosed rather than absorbed.
The `policy_bo3` control — the single prompt sampled three times at temperature 0.8 with seeds 17, 18
and 19 and a majority vote — runs as a **second phase**, after both pre-registered arms have completed
on all 24 windows, on the **8 windows with the largest debate-minus-single gain**. It runs in the same
chain, so no two inference processes contend for the GPU.

The control set is therefore **selected on this sweep's own outcomes**, and nothing based on it may be
reported as a test. The selection is deliberate: a control that separates structure from spend is only
informative where an advantage exists to explain, and running it on windows where the two arms tie
would answer nothing. The cost is that the controlled windows are the arm's best, so the control can
falsify a structure claim there but cannot be read as an unbiased estimate of the control's own mean.

This control is **descriptive and carries no test**. It is not an arm of the pre-registered paired
comparison, the two declared tests and their Holm family are unchanged, and the primary analysis
remains the debate-minus-single difference over all 24 training windows, every one of which receives
both pre-registered arms regardless of outcome.
