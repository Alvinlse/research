# Debate v2.4.2 two-window pilot analysis

Decision: **NOT POSITIVE — do not launch the sweep**.

The decision applies the gate frozen in `debate_v242_pilot_gate.md` before results were available.

## Matched-window results

| window | predecessor | base deadline | pilot deadline | delta | base wait (s) | pilot wait (s) | delta | base util | pilot util | delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| d111h22 | v2.4.1 | 20.7% | 8.9% | -11.8 pp | 73,764 | 57,764 | -21.7% | 0.651 | 0.721 | +0.070 |
| d223h11 | frozen v2.4 | 25.1% | 28.0% | +2.9 pp | 36,445 | 37,929 | +4.1% | 0.853 | 0.850 | -0.003 |

Macro deadline violation changed from 22.90% to
18.45% (-4.45 pp).
Macro mean wait changed from 55104.5 s to
47846.5 s (-13.17%).
Macro useful utilization changed from 0.752 to
0.785 (+0.033).

## Frozen gate

| condition | result |
| --- | --- |
| both complete zero llm errors | PASS |
| revised evaluator exercised | PASS |
| macro deadline improves by 1pp | PASS |
| no window deadline regresses over 2pp | FAIL |
| macro wait within 5pct | PASS |
| macro util within 0 02 | PASS |

## Trace evidence

- `d111h22`: 1 started, 1 probation, 1 accepted, 0 rolled back; 0 invalid debates held the incumbent. Executed ordering epochs: `{"auction_deadline": 85, "auction_priority": 8}`. Rotation actions: `{"emergency_demand": 58, "hold": 33, "trial_continue": 1, "trial_start": 1}`.
  - trial_probation for auction_priority->auction_deadline (demand): reason `None`; checks `{"deadline_pressure_delta": 0.0, "deadline_pressure_ok": true, "objective": {"deadline_pressure_jobs_delta": 0, "name": "demand_deadline_or_queue"}, "objective_ok": true, "passed": true, "queue_delta": -6, "queue_trend_limit": 55, "queue_trend_ok": true, "wait_s_p90_growth": 1800, "wait_s_p90_limit": 2100, "wait_s_p90_ok": true}`.
  - trial_accept for auction_priority->auction_deadline (demand): reason `probation_passed`; checks `{"deadline_pressure_delta": 0.0, "deadline_pressure_ok": true, "objective": {"deadline_pressure_jobs_delta": 0, "name": "demand_deadline_or_queue"}, "objective_ok": true, "passed": true, "queue_delta": -4, "queue_trend_limit": 0, "queue_trend_ok": true, "wait_s_p90_growth": 2780, "wait_s_p90_limit": 3116, "wait_s_p90_ok": true}`.
- `d223h11`: 8 started, 5 probation, 1 accepted, 7 rolled back; 4 invalid debates held the incumbent. Executed ordering epochs: `{"auction_deadline": 81, "auction_fairness": 23}`. Rotation actions: `{"emergency_demand": 24, "hold": 67, "trial_continue": 5, "trial_start": 8}`.
  - trial_probation for auction_deadline->auction_fairness (supply): reason `None`; checks `{"deadline_pressure_delta": 0.0, "deadline_pressure_ok": true, "objective": {"name": "supply_queue_or_concentration", "top_user_wait_share_delta": -0.854}, "objective_ok": true, "passed": true, "queue_delta": -30, "queue_trend_limit": 29, "queue_trend_ok": true, "wait_s_p90_growth": -920, "wait_s_p90_limit": 2123, "wait_s_p90_ok": true}`.
  - trial_rollback for auction_deadline->auction_fairness (supply): reason `probation_failed`; checks `{"deadline_pressure_delta": 0.0, "deadline_pressure_ok": true, "objective": {"name": "supply_queue_or_concentration", "top_user_wait_share_delta": 0.0}, "objective_ok": false, "passed": false, "queue_delta": 0, "queue_trend_limit": 0, "queue_trend_ok": true, "wait_s_p90_growth": 0, "wait_s_p90_limit": 2201, "wait_s_p90_ok": true}`.
  - trial_probation for auction_deadline->auction_fairness (supply): reason `None`; checks `{"deadline_pressure_delta": 0.0, "deadline_pressure_ok": true, "objective": {"name": "supply_queue_or_concentration", "top_user_wait_share_delta": -0.028}, "objective_ok": true, "passed": true, "queue_delta": 0, "queue_trend_limit": 0, "queue_trend_ok": true, "wait_s_p90_growth": 1923, "wait_s_p90_limit": 2226, "wait_s_p90_ok": true}`.
  - trial_rollback for auction_deadline->auction_fairness (supply): reason `probation_failed`; checks `{"deadline_pressure_delta": 0.0, "deadline_pressure_ok": true, "objective": {"name": "supply_queue_or_concentration", "top_user_wait_share_delta": -0.038}, "objective_ok": true, "passed": false, "queue_delta": 12, "queue_trend_limit": 0, "queue_trend_ok": false, "wait_s_p90_growth": 2271, "wait_s_p90_limit": 2574, "wait_s_p90_ok": true}`.
  - trial_probation for auction_deadline->auction_fairness (supply): reason `None`; checks `{"deadline_pressure_delta": 0.0, "deadline_pressure_ok": true, "objective": {"name": "supply_queue_or_concentration", "top_user_wait_share_delta": -0.1}, "objective_ok": true, "passed": true, "queue_delta": -1, "queue_trend_limit": 54, "queue_trend_ok": true, "wait_s_p90_growth": 2174, "wait_s_p90_limit": 2491, "wait_s_p90_ok": true}`.
  - trial_rollback for auction_deadline->auction_fairness (supply): reason `probation_failed`; checks `{"deadline_pressure_delta": 0.0, "deadline_pressure_ok": true, "objective": {"name": "supply_queue_or_concentration", "top_user_wait_share_delta": -0.05}, "objective_ok": true, "passed": false, "queue_delta": 3, "queue_trend_limit": 0, "queue_trend_ok": false, "wait_s_p90_growth": 1807, "wait_s_p90_limit": 2110, "wait_s_p90_ok": true}`.
  - trial_probation for auction_deadline->auction_fairness (supply): reason `None`; checks `{"deadline_pressure_delta": 0.0, "deadline_pressure_ok": true, "objective": {"name": "supply_queue_or_concentration", "top_user_wait_share_delta": -0.042}, "objective_ok": true, "passed": true, "queue_delta": 11, "queue_trend_limit": 17, "queue_trend_ok": true, "wait_s_p90_growth": 1866, "wait_s_p90_limit": 2179, "wait_s_p90_ok": true}`.
  - trial_accept for auction_deadline->auction_fairness (supply): reason `probation_passed`; checks `{"deadline_pressure_delta": 0.0, "deadline_pressure_ok": true, "objective": {"name": "supply_queue_or_concentration", "top_user_wait_share_delta": -0.073}, "objective_ok": true, "passed": true, "queue_delta": 0, "queue_trend_limit": 10, "queue_trend_ok": true, "wait_s_p90_growth": 2252, "wait_s_p90_limit": 2558, "wait_s_p90_ok": true}`.
  - trial_probation for auction_fairness->auction_deadline (demand): reason `None`; checks `{"deadline_pressure_delta": -0.013, "deadline_pressure_ok": true, "objective": {"deadline_pressure_jobs_delta": -4, "name": "demand_deadline_or_queue"}, "objective_ok": true, "passed": true, "queue_delta": -3, "queue_trend_limit": 51, "queue_trend_ok": true, "wait_s_p90_growth": 1482, "wait_s_p90_limit": 2115, "wait_s_p90_ok": true}`.
  - trial_rollback for auction_fairness->auction_deadline (demand): reason `probation_failed`; checks `{"deadline_pressure_delta": 0.007, "deadline_pressure_ok": true, "objective": {"deadline_pressure_jobs_delta": 2, "name": "demand_deadline_or_queue"}, "objective_ok": false, "passed": false, "queue_delta": -2, "queue_trend_limit": 0, "queue_trend_ok": true, "wait_s_p90_growth": -2155, "wait_s_p90_limit": 2317, "wait_s_p90_ok": true}`.
  - trial_rollback for auction_fairness->auction_deadline (demand): reason `trial_failed`; checks `{"deadline_pressure_delta": 0.0, "deadline_pressure_ok": true, "objective": {"deadline_pressure_jobs_delta": 0, "name": "demand_deadline_or_queue"}, "objective_ok": false, "passed": false, "queue_delta": 6, "queue_trend_limit": 3, "queue_trend_ok": false, "wait_s_p90_growth": 439, "wait_s_p90_limit": 2213, "wait_s_p90_ok": true}`.
  - trial_rollback for auction_fairness->auction_deadline (demand): reason `trial_failed`; checks `{"deadline_pressure_delta": 0.023, "deadline_pressure_ok": true, "objective": {"deadline_pressure_jobs_delta": 7, "name": "demand_deadline_or_queue"}, "objective_ok": false, "passed": false, "queue_delta": -7, "queue_trend_limit": 0, "queue_trend_ok": true, "wait_s_p90_growth": 741, "wait_s_p90_limit": 2460, "wait_s_p90_ok": true}`.
  - trial_rollback for auction_fairness->auction_deadline (demand): reason `trial_failed`; checks `{"deadline_pressure_delta": 0.154, "deadline_pressure_ok": false, "objective": {"deadline_pressure_jobs_delta": 42, "name": "demand_deadline_or_queue"}, "objective_ok": false, "passed": false, "queue_delta": -14, "queue_trend_limit": 0, "queue_trend_ok": true, "wait_s_p90_growth": 979, "wait_s_p90_limit": 2346, "wait_s_p90_ok": true}`.

## Interpretation

The gate failed at least one preregistered condition. Even if an individual metric improved, the evidence is not strong enough to justify a 24-window sweep under the frozen rule. Trace counters identify whether the failure occurred despite rotation, rollback, or mostly incumbent holds; they do not by themselves establish causal attribution.
