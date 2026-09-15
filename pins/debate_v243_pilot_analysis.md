# Debate v2.4.3 two-window pilot analysis

Decision: **GATE NOT PASSED — sweep continues by explicit user override**.

## Matched-window results

| window | base deadline | pilot deadline | delta | base wait (s) | pilot wait (s) | delta | base util | pilot util | delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| d111h22 | 8.9% | 14.2% | +5.3 pp | 57,764 | 67,013 | +16.0% | 0.721 | 0.721 | +0.000 |
| d223h11 | 28.0% | 33.5% | +5.5 pp | 37,929 | 42,936 | +13.2% | 0.850 | 0.850 | +0.000 |

Macro deadline: 18.45% to 23.85%.
Macro wait: 47846.5 s to 54974.5 s.
Macro utilization: 0.7855 to 0.7855.

## Frozen gate

| condition | result |
| --- | --- |
| both complete zero llm errors | PASS |
| revised evaluator exercised | PASS |
| d223 deadline at most 27 1 | FAIL |
| d111 deadline at most 10 9 | FAIL |
| macro deadline not worse | FAIL |
| macro wait within 5pct | FAIL |
| macro util within 0 02 | PASS |
| runtime structural integrity | PASS |

## Trace evidence

- `d111h22`: 1 trials, 1 accepts, 0 rollbacks, 0 blocked requests; phase-feedback mismatches=0, backoff start violations=0. Executed orderings: `{"auction_deadline": 79, "auction_fairness": 15, "auction_priority": 3}`.
- `d223h11`: 3 trials, 0 accepts, 3 rollbacks, 0 blocked requests; phase-feedback mismatches=0, backoff start violations=0. Executed orderings: `{"auction_deadline": 100, "auction_fairness": 4}`.

## Interpretation

At least one frozen condition failed. The 24-window sweep nevertheless continues under the explicit presentation-deadline override; the failed gate remains reported unchanged.
