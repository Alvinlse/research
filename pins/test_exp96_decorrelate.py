"""Exp 96 harness checks — the de-confound is only exact if the windows are untouched.

Run: .venv/bin/python -m pins.test_exp96_decorrelate   (no network, no LLM)

What is load-bearing:
  1. Every job property except `tier` must be BYTE-IDENTICAL between the two worlds. If the tier
     draw touched the shared RNG stream, windows would shift and the cross-world comparison would
     be unpaired -- the Exp 59 trap the pre-registration names.
  2. The prod MARGINAL must be preserved (~1/3), or the two worlds differ in how much prod there
     is as well as in which jobs it lands on, and the diff-in-diff confounds the two.
  3. Correlation between tier and laxity: deterministic in the correlated world, ~0 in the new one.
     That is hypothesis H3, and it is what the whole experiment rests on.
"""
from pins.trace_replay import (TRACES, load_predicted_quanta, load_trace, make_trace_workload)
from pins.two_sided_sim import tight_third


def _world(trace, seed, tick, pred, decorrelate):
    return make_trace_workload(trace, 16, seed, 300, pred, False, None, False, None, None,
                               tick, 1, decorrelate=decorrelate)


def _laxity(j):
    return (j.deadline - j.arrival) / max(sum(j.need), 1e-9)


def main() -> None:
    path, tick = TRACES["v2020"]
    trace, pred = load_trace(path), load_predicted_quanta(quantile="p50")
    seeds = range(32)
    n_prod_c = n_prod_d = n_jobs = 0
    overlap = []

    for s in seeds:
        jc, capc, tcapc, _ = _world(trace, s, tick, pred, False)
        jd, capd, tcapd, _ = _world(trace, s, tick, pred, True)
        # 1. everything but the label is identical
        assert capc == capd and tcapc == tcapd, f"seed {s}: cap maps diverged"
        for a, b in zip(jc, jd):
            assert (a.jid, a.arrival, a.need, a.urgency, a.deadline) == \
                   (b.jid, b.arrival, b.need, b.urgency, b.deadline), f"seed {s}: window shifted"
        n_jobs += len(jc)
        n_prod_c += sum(1 for j in jc if j.tier == "prod")
        n_prod_d += sum(1 for j in jd if j.tier == "prod")
        # 3. how much of the tight tercile is prod, in each world
        for js in (jc, jd):
            tight = {j.jid for j in tight_third(js)}
            prod = {j.jid for j in js if j.tier == "prod"}
            overlap.append(len(tight & prod) / max(len(tight), 1))

    corr = sum(overlap[0::2]) / len(seeds)      # correlated world
    deco = sum(overlap[1::2]) / len(seeds)      # decorrelated world
    print(f"windows identical except tier across {len(list(seeds))} seeds, {n_jobs} jobs")
    print(f"prod marginal: correlated {n_prod_c/n_jobs:.3f}  decorrelated {n_prod_d/n_jobs:.3f}"
          f"  (target {1/3:.3f})")
    print(f"share of the tightest-laxity tercile that is prod: "
          f"correlated {corr:.3f} -> decorrelated {deco:.3f}")

    assert abs(n_prod_d / n_jobs - 1 / 3) < 0.06, "prod marginal moved"
    # not 1.0: deadline = arrival + round(work * slack), so integer rounding perturbs the laxity
    # ratio and a short besteffort job can land inside the tercile. The confound is strong, not
    # total -- which is itself worth knowing, since it bounds how much of prodSLA can be laxity.
    assert corr > 0.80, f"correlated world should be prod-dominated in the tight tercile: {corr}"
    assert abs(deco - 1 / 3) < 0.10, f"decorrelated tercile should be at chance, got {deco}"

    # the correlated world's confound, stated as the number it is
    jc, _, _, _ = _world(trace, 0, tick, pred, False)
    lax_prod = sorted(round(_laxity(j), 2) for j in jc if j.tier == "prod")
    lax_be = sorted(round(_laxity(j), 2) for j in jc if j.tier != "prod")
    print(f"seed 0 laxity, correlated world: prod {lax_prod} | besteffort {lax_be}")
    med = lambda xs: xs[len(xs) // 2]
    assert not lax_prod or not lax_be or med(lax_prod) < med(lax_be), \
        "correlated world should give prod systematically tighter deadlines"
    print("OK")


if __name__ == "__main__":
    main()
