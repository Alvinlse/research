# Debate v2.4.2 pilot gate

This gate was frozen while the first pilot window was still running and before either v2.4.2
result was available. The pilot uses seed 17, `qwen2.5:14b`, and real training windows `d111h22`
and `d223h11`.

## Matched predecessor baselines

| window | predecessor | deadline violation | mean wait | useful utilization |
| --- | --- | ---: | ---: | ---: |
| d111h22 | v2.4.1 (`956d289`) | 20.7% | 73,764 s | 0.651 |
| d223h11 | frozen v2.4 (`b6d0d6c`) | 25.1% | 36,445 s | 0.853 |

The mixed predecessor is necessary because no full v2.4.1 d223h11 run exists. The macro baseline
is 22.9% deadline violation, 55,104.5 s mean wait, and 0.752 useful utilization.

## Positive-result gate

All conditions must pass:

1. Both windows complete with zero LLM errors.
2. At least one ordering trial reaches a recorded accept or rollback, proving the revised evaluator
   was exercised rather than merely holding throughout both windows.
3. Macro deadline violation is at least 1.0 percentage point below the 22.9% matched baseline.
4. Neither window's deadline violation is more than 2.0 percentage points above its baseline.
5. Macro mean wait is no more than 5% above the 55,104.5-second baseline.
6. Macro useful utilization is no more than 0.02 below the 0.752 baseline.

If every condition passes, publish the pilot metrics and trace-based analysis to GitHub, then start
a fresh resumable sweep over all 24 training windows. If any condition fails, publish the analysis
but do not start the sweep.
