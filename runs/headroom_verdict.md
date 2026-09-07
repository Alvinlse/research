# Per-epoch policy-selection headroom: verdict (2026-09-07)

Idle-window headroom (load<1.0, n=266), each metric optimised for itself:
  wait_s  1113.8 -> 835.1  (25.0%, 279 s abs)     loaded: 6.8%, 1514 s abs
  sla10     0.45 ->  0.39  (15.0%)
  bsd       1.46 ->  1.35  ( 7.7%)                 loaded: 17.0%
NOT a bounded-slowdown artifact: bsd is the SMALLEST idle term.

Qualifiers: top 10% of idle states hold 58% of the gain, top 20% hold 76%;
reward-argmax == wait-argmax on only 34% of idle states (reward prefers
resize_conservative for zero churn, wait prefers market).

Learned selector (tree) vs best fixed policy, GroupKFold over 54 windows:
  objective          tree     fixed   tree wins   idle-only tree/fixed
  balanced reward    0.2991   0.1665   0/5        0.1461 / 0.0802
  throughput reward  0.3438   0.1868   0/5        0.1474 / 0.1105
  wait only (s)      2103.6   955.3    0/5        426.4  / 278.2
  sla10 only (%)     2.79     1.42     0/5        0.84   / 0.14

The headroom is real, small in absolute terms, concentrated in a tail, and
unreachable under every objective. Negative result is robust to model class,
scale, data size, features, menu size and reward choice.
