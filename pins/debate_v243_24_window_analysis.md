# Debate v2.4.3 — 24-window sweep analysis

This is an exploratory repair sweep over the same 24 training/development windows used to revise
the protocol. It is appropriate for the presentation as development evidence, not as an unbiased
generalization result. The sweep ran by explicit user override regardless of the two-window gate.

## Macro results

| method | deadline violations | mean wait (s) | p90 wait (s) | useful utilization |
| --- | ---: | ---: | ---: | ---: |
| v2.4.3 | 10.99% | 24999 | 45678 | 0.825 |
| single | 13.36% | 24149 | 46859 | 0.834 |
| debate_v1 | 10.96% | 25580 | 49564 | 0.818 |

## Paired deadline result

| baseline | wins | ties | losses | mean v2.4.3 minus baseline |
| --- | ---: | ---: | ---: | ---: |
| single | 12 | 9 | 3 | -2.37 pp |
| debate | 8 | 9 | 7 | +0.03 pp |

## Protocol behavior

- LLM calls: 4712; LLM errors: 0.
- Trials: 76 started, 43 reached probation,
  24 accepted, and 52 rolled back.
- Transition-backoff blocks: 14 advocate requests;
  defense-layer holds: 0.
- Invalid debates held the incumbent 65 times without fixed fallback.

## Per-window deadline results

| window | load | v2.4.3 | single | debate v1 | v2.4.3 util | source |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| d48h23 | 4.99 | 15.2% | 24.1% | 14.3% | 0.864 | sw_v243_s17 |
| d111h22 | 3.76 | 14.2% | 26.9% | 12.7% | 0.721 | pilot_v243_s17 |
| d22h6 | 3.53 | 30.5% | 31.5% | 32.0% | 0.984 | sw_v243_s17 |
| d118h14 | 3.01 | 21.9% | 25.8% | 22.6% | 0.870 | sw_v243_s17 |
| d223h11 | 2.90 | 33.5% | 28.1% | 34.8% | 0.850 | pilot_v243_s17 |
| d188h6 | 2.74 | 4.6% | 4.8% | 7.7% | 0.750 | sw_v243_s17 |
| d202h19 | 2.68 | 30.9% | 34.7% | 24.0% | 0.993 | sw_v243_s17 |
| d160h17 | 2.54 | 7.7% | 7.7% | 7.7% | 0.853 | sw_v243_s17 |
| d207h17 | 2.46 | 5.5% | 5.5% | 5.3% | 0.882 | sw_v243_s17 |
| d102h8 | 1.88 | 8.0% | 8.5% | 9.5% | 0.948 | sw_v243_s17 |
| d109h7 | 1.82 | 25.6% | 28.8% | 26.8% | 1.000 | sw_v243_s17 |
| d190h0 | 1.78 | 11.4% | 10.6% | 10.0% | 0.958 | sw_v243_s17 |
| d185h6 | 1.69 | 4.7% | 5.1% | 4.7% | 0.935 | sw_v243_s17 |
| d225h0 | 1.47 | 9.1% | 12.8% | 7.8% | 1.000 | sw_v243_s17 |
| d58h19 | 1.35 | 13.0% | 13.0% | 13.0% | 0.465 | sw_v243_s17 |
| d209h23 | 1.32 | 5.2% | 5.2% | 6.8% | 0.898 | sw_v243_s17 |
| d221h21 | 1.23 | 2.3% | 2.1% | 2.1% | 0.946 | sw_v243_s17 |
| d211h16 | 1.19 | 8.5% | 8.5% | 8.5% | 0.960 | sw_v243_s17 |
| d98h19 | 0.91 | 0.0% | 0.1% | 0.0% | 0.842 | sw_v243_s17 |
| d71h7 | 0.88 | 1.0% | 1.0% | 1.0% | 0.677 | sw_v243_s17 |
| d218h4 | 0.62 | 0.0% | 0.0% | 0.0% | 0.627 | sw_v243_s17 |
| d116h2 | 0.58 | 4.8% | 29.6% | 5.6% | 1.000 | sw_v243_s17 |
| d26h21 | 0.49 | 6.2% | 6.2% | 6.2% | 0.512 | sw_v243_s17 |
| d63h11 | 0.32 | 0.0% | 0.0% | 0.0% | 0.260 | sw_v243_s17 |
