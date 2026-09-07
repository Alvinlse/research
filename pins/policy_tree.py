"""Tabular baseline: does the LLM earn its place over a boosted tree on the same features?

The referee's packet is mostly static -- the policy menu and the cluster rules never change. The
part that varies is ten numeric fields plus the current policy. A gradient-boosted tree on exactly
those fields has a far stronger sample-efficiency prior than a 3B causal LM fine-tuned on 407
examples, so it is the baseline the LLM has to beat before any claim about reasoning is credible.

The tree regresses the WHOLE 15-entry reward vector (centred per state, so it learns which action
is relatively better rather than how hard the window is) and picks the argmax at inference. That
uses the same supervision the LLM had and is scored with the identical regret metric.

Evaluation is grouped by window everywhere: GroupKFold for model selection and a window-level
bootstrap for intervals, because states inside one window share a trajectory and are not
independent draws.

    .venv/bin/python -m pins.policy_tree --data runs/dataset --out runs/tree_report.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

from pins.elastisim_bench import POLICY_MENU

ACTIONS = [f"{o}+{z}" for o in POLICY_MENU["ordering"] for z in POLICY_MENU["sizing"]]
ORDERINGS = list(POLICY_MENU["ordering"])
SIZINGS = list(POLICY_MENU["sizing"])
NUM = ["now", "pool", "free", "running", "queue_depth", "waiting_demand",
       "queue_pressure", "arrival_load_1h", "prod_waiting", "declared_walltime"]


def features(packet: str) -> list[float]:
    """The ten numeric fields plus the current policy, one-hot. Everything else in the packet is
    the same static menu text on every state and carries no information."""
    f = {}
    for k in NUM:
        m = re.search(rf"\b{k}=(-?[\d.]+)", packet)
        f[k] = float(m.group(1)) if m else 0.0
    o = re.search(r"current_policy=ordering:(\S+)", packet)
    z = re.search(r"sizing:(\S+)", packet)
    cur_o = o.group(1) if o else ""
    cur_z = z.group(1) if z else ""
    return ([f[k] for k in NUM]
            + [1.0 * (cur_o == x) for x in ORDERINGS]
            + [1.0 * (cur_z == x) for x in SIZINGS])


def load(data: Path, split: str):
    rows = [json.loads(l) for l in (data / f"{split}.jsonl").read_text().splitlines() if l.strip()]
    X = np.array([features(r["messages"][1]["content"]) for r in rows], dtype=float)
    Y = np.array([[r["rewards"][a] for a in ACTIONS] for r in rows], dtype=float)
    g = np.array([r["window"] for r in rows])
    return rows, X, Y, g


def regret(Y: np.ndarray, pick: np.ndarray, menu_idx: list[int]) -> np.ndarray:
    """Per-state shortfall against the best action available in `menu_idx`."""
    sub = Y[:, menu_idx]
    best = sub.max(1)
    got = np.array([Y[i, p] if p in menu_idx else sub[i].min() for i, p in enumerate(pick)])
    return best - got


def fit_predict(Xtr, Ytr, Xte, kind: str):
    """One regressor per action on the per-state-CENTRED reward.

    Centring matters: raw reward is dominated by how hard the window is, which is shared by all 15
    actions and tells the model nothing about which to choose. Removing the per-state mean leaves
    exactly the contrast the decision turns on.
    """
    Yc = Ytr - Ytr.mean(1, keepdims=True)
    P = np.zeros((len(Xte), len(ACTIONS)))
    for j in range(len(ACTIONS)):
        if kind == "tree":
            m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.06,
                                              max_depth=4, min_samples_leaf=10,
                                              l2_regularization=1.0, random_state=0)
        elif kind == "ridge":
            m = Ridge(alpha=1.0)
        else:
            m = DummyRegressor(strategy="mean")
        m.fit(Xtr, Yc[:, j])
        P[:, j] = m.predict(Xte)
    return P


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("runs/dataset"))
    ap.add_argument("--out", type=Path, default=Path("runs/tree_report.json"))
    ap.add_argument("--boot", type=int, default=2000)
    a = ap.parse_args()

    meta = json.loads((a.data / "meta.json").read_text())
    cat = meta["catastrophic_arms"]
    safe_idx = [i for i, x in enumerate(ACTIONS) if x not in cat]

    tr_rows, Xtr, Ytr, gtr = load(a.data, "train")
    va_rows, Xva, Yva, gva = load(a.data, "val")
    te_rows, Xte, Yte, gte = load(a.data, "test")
    # Train and val are both legitimately available at fit time; the test windows are untouched.
    X = np.vstack([Xtr, Xva]); Y = np.vstack([Ytr, Yva]); g = np.concatenate([gtr, gva])
    print(f"train+val {len(X)} states / {len(set(g))} windows | test {len(Xte)} / {len(set(gte))}")

    report = {"n_test": len(Xte), "catastrophic": cat, "arms": {}}

    # Grouped CV on train+val: the honest estimate of how this generalises to an unseen WINDOW.
    cv = GroupKFold(n_splits=min(5, len(set(g))))
    for kind in ("tree", "ridge"):
        rs = []
        for tr_i, te_i in cv.split(X, Y, g):
            P = fit_predict(X[tr_i], Y[tr_i], X[te_i], kind)
            pick = np.array([safe_idx[int(np.argmax(P[i, safe_idx]))] for i in range(len(te_i))])
            rs.append(regret(Y[te_i], pick, safe_idx).mean())
        report["arms"].setdefault(kind, {})["cv_regret_safe"] = round(float(np.mean(rs)), 4)
        report["arms"][kind]["cv_folds"] = [round(float(x), 4) for x in rs]

    # Held-out test, same split the LLM was scored on.
    for kind in ("tree", "ridge"):
        P = fit_predict(X, Y, Xte, kind)
        pick = np.array([safe_idx[int(np.argmax(P[i, safe_idx]))] for i in range(len(Xte))])
        r = regret(Yte, pick, safe_idx)
        d = report["arms"][kind]
        d["test_regret_safe"] = round(float(r.mean()), 4)
        d["eps5"] = round(float((r <= 0.05).mean()), 3)
        d["n_distinct_picks"] = int(len(set(pick.tolist())))
        d["picks"] = {ACTIONS[i]: int((pick == i).sum()) for i in sorted(set(pick.tolist()))}
        # Window-level bootstrap: resample WINDOWS, not states.
        wins = sorted(set(gte))
        rng = np.random.default_rng(0)
        bs = []
        for _ in range(a.boot):
            pick_w = rng.choice(wins, len(wins), replace=True)
            idx = np.concatenate([np.where(gte == w)[0] for w in pick_w])
            bs.append(r[idx].mean())
        d["ci95"] = [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]

    # Same-metric reference points, recomputed here so the comparison is apples to apples.
    for name, idx in (("majority_train", ACTIONS.index(
            max(set(ACTIONS), key=lambda x: (Ytr.argmax(1) == ACTIONS.index(x)).sum()))),):
        r = regret(Yte, np.full(len(Yte), idx), safe_idx)
        report["arms"][name] = {"test_regret_safe": round(float(r.mean()), 4),
                                "eps5": round(float((r <= 0.05).mean()), 3),
                                "picks": {ACTIONS[idx]: len(Yte)}}
    a.out.write_text(json.dumps(report, indent=1))

    print(f"\n{'arm':<18} {'cv_regret':>10} {'test_regret':>12} {'95% CI':>18} {'eps5':>7} {'picks':>6}")
    for k, v in report["arms"].items():
        print(f"{k:<18} {v.get('cv_regret_safe', float('nan')):>10.4f} "
              f"{v['test_regret_safe']:>12.4f} "
              f"{str(v.get('ci95', '-')):>18} {v['eps5']:>7.3f} {v.get('n_distinct_picks', 1):>6}")
    print(f"\nreport -> {a.out}")


if __name__ == "__main__":
    main()
