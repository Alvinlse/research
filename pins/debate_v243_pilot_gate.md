# Debate v2.4.3 pilot gate

This gate is frozen before either v2.4.3 result exists. The pilot uses seed 17,
`qwen2.5:14b`, and the same real training windows as v2.4.2: `d111h22` and `d223h11`.

## Matched predecessor baselines

| window | predecessor | deadline violation | mean wait | useful utilization |
| --- | --- | ---: | ---: | ---: |
| d111h22 | v2.4.2 (`df531df`) | 8.9% | 57,764 s | 0.721 |
| d223h11 | v2.4.2 (`df531df`) | 28.0% | 37,929 s | 0.850 |

The macro baseline is 18.45% deadline violation, 47,846.5 s mean wait, and 0.7855 useful
utilization.

## Positive-result gate

All conditions must pass:

1. Both windows complete with zero LLM errors.
2. At least one ordering trial reaches a recorded accept or rollback.
3. `d223h11` deadline violation is at most 27.1%, the already-declared repair target.
4. `d111h22` deadline violation is at most 10.9%, allowing no more than a 2-point regression.
5. Macro deadline violation is no higher than the 18.45% v2.4.2 baseline.
6. Macro mean wait is no more than 5% above the 47,846.5-second baseline.
7. Macro useful utilization is no more than 0.02 below the 0.7855 baseline.
8. Every completed outcome's learning record uses the phase-check deltas and labels rollback as
   negative challenger evidence; no trial starts while its exact transition is in backoff.

If every condition passes, publish the metrics and trace-based analysis to GitHub, then start a
fresh resumable sweep over all 24 training windows. If any condition fails, publish the analysis
but do not start the sweep.
