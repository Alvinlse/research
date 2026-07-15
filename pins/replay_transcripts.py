"""Replay pool-6 seeds 2 & 3 of the deepseek referee arm, logging full transcripts.
All LLM calls hit llm_agent_cache.json, so this is a deterministic cached replay."""
from pins.trace_replay import load_trace, make_trace_workload, TRACES
from pins.two_sided_sim import simulate
from pins.uncertainty_sim import assign, load_uncertainty_distribution
from pins.llm_agent import load_cache
from pins.referee import referee_decide
from pins.negotiation_protocol import NegotiationOutcome

MODEL, GPUS, HORIZON, N_JOBS, SCALE, SPIKE = "deepseek-r1:32b", 6, 300, 16, 3, 0.6
trace = load_trace(TRACES["v2020"][0])
tick = TRACES["v2020"][1]
dist = load_uncertainty_distribution()
cache = load_cache()

for s in (2, 3):
    jobs, cap_map, tcap, belief = make_trace_workload(trace, N_JOBS, s, HORIZON, None,
                                                      False, None, False, None, None,
                                                      tick, 1)
    cap_map = {k: min(v, GPUS) for k, v in cap_map.items()}
    tcap = {k: min(v, GPUS) for k, v in tcap.items()}
    u_map, spike_map = assign(jobs, s, dist, SPIKE)
    calls = []

    def policy(demand, supply_ctx, free, **_):
        o = referee_decide(demand, supply_ctx, free, use_llm=True, model=MODEL,
                           cache=cache, statement_model="qwen2.5:3b")
        margins = {j.jid: o.alloc.get(j.jid, 0) for j in demand}
        reserve = o.reserve
        if not o.feasible:
            margins = {j.jid: 0 for j in demand}
            reserve = 0
        calls.append({"free": free, "stmts": o.transcript, "alloc": dict(o.alloc),
                      "reserve": o.reserve, "feasible": o.feasible,
                      "violations": o.violations, "why": o.justification,
                      "src": o._source})
        return margins, reserve, NegotiationOutcome(margins=margins, reserve=reserve,
                                                    rounds=1, agreed=o.feasible,
                                                    transcript=o.transcript)

    r = simulate(jobs, policy, GPUS, HORIZON, u_map, spike_map, SCALE, SPIKE,
                 cap_map, true_cap_map=tcap)
    print(f"\n{'#'*90}\nSEED {s}  pool {GPUS}: SLA {r['sla']:.1%} prodSLA {r['prod_sla']:.1%} "
          f"util {r['util']:.0%} slowdown {r['slowdown']:.2f} | {len(calls)} referee calls")
    print('#'*90)
    prev = None
    for i, c in enumerate(calls):
        key = (c["free"], tuple(sorted(c["alloc"].items())), c["reserve"])
        dup = " (same scene as previous)" if key == prev else ""
        prev = key
        if dup and i > 0:
            continue                      # collapse consecutive identical scenes
        print(f"\n--- call {i}  free={c['free']}  src={c['src']}  feasible={c['feasible']}{dup}")
        for st in c["stmts"]:
            if st["side"] == "demand":
                print(f"  demand {st['job_id']} [{st['tier']}/{st['deadline']}] "
                      f"base={st['base_gpus']} +margin_req={st['requested_margin_gpus']}: "
                      f"{st['justification'][:150]}")
            else:
                print(f"  supply reserve_req={st['requested_reserve_gpus']} "
                      f"incoming_prod={st['incoming_prod']}: {st['justification'][:150]}")
        print(f"  REFEREE -> alloc={c['alloc']} reserve={c['reserve']}")
        if c["violations"]:
            print(f"  VIOLATIONS: {c['violations']}")
        print(f"  why: {c['why']}")
