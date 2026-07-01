"""
LLM AFFINITY agent (placement extension) — the one open door Exp 21 left for relational reasoning.

Exp 21 Part C found GPU co-location slowdown is `k`-determined (a one-line `slowdown ≈ k` rule) for
HOMOGENEOUS conv tasks — no relational signal an attention/GNN model could exploit. The single
exception it flagged: HETEROGENEOUS bottlenecks. Two compute-bound tasks on one node contend on the
SMs (bad); a compute-bound + a bandwidth-bound task overlap PRODUCTIVELY (the bandwidth task uses
the memory system while the compute task uses the cores). That mix-dependence is exactly what a
`≈k` rule cannot express — and it is a SEMANTIC judgement ("what is this task's dominant
bottleneck?"), which is what an LLM can do and a numbers-only solver cannot infer from a GPU count.

So the LLM's placement role is NOT to decide which task goes to which node (that is the ILP's job,
which Exp 18 showed it does provably-optimally). The LLM only classifies each task's BOTTLENECK
CLASS (categorical, justified); deterministic code (`affinity_matrix`) turns the classes into a
soft same-node AFFINITY the ILP (`pins/ilp.allocate_placement`, new `affinity` arg) optimises
inside the hard capacity/co-location constraints. Same hinge: LLM reasons, ILP decides + guarantees.

Run:  .venv/bin/python -m pins.affinity            # classify a few task profiles -> affinity matrix
"""
from __future__ import annotations

from itertools import combinations

from pins.llm_agent import DEFAULT_MODEL, HOST, _parse, load_cache

# The dominant resource a task contends on. compute = SM/tensor-core bound (matmul/conv);
# bandwidth = memory/interconnect bound (all-reduce, embedding lookups); io = host/data bound.
BOTTLENECKS = ["compute", "bandwidth", "io"]
# Same-contending-class co-location HURTS (they fight over the same unit); a code constant, not the
# LLM's to set. compute and bandwidth contend on their own resource; io tasks barely use the GPU,
# so co-locating io with anything is cheap (penalty 0).
CONTENDS = {"compute", "bandwidth"}

SYSTEM_BOTTLENECK = (
    "You are a placement advisor for a shared GPU cluster. Given ONE training/inference task's "
    "description, classify its DOMINANT hardware bottleneck — which single resource it spends most "
    "of its time contending for. You output NO numbers.\n"
    "Respond with ONLY this JSON, using EXACTLY the allowed values:\n"
    '{"bottleneck": "compute|bandwidth|io", "justification": "<one short sentence>"}\n'
    "Guidance: 'compute' = dominated by dense matrix multiply / convolution (saturates the tensor "
    "cores). 'bandwidth' = dominated by memory or interconnect traffic (large all-reduce / "
    "parameter sync / huge embedding lookups / memory-bound kernels). 'io' = dominated by host-side "
    "data loading / preprocessing and barely uses the GPU. Two tasks with the SAME compute or "
    "bandwidth bottleneck contend if co-located; complementary ones overlap well. Justify briefly."
)


def bottleneck_state_key(ctx: dict) -> str:
    return ctx["op_profile"]


def _bottleneck_prompt(ctx: dict) -> str:
    return f"Task description: {ctx['op_profile']}. Classify its dominant bottleneck."


def _rule_bottleneck(ctx: dict) -> dict:
    """Deterministic fallback: keyword-match the op profile. Stands in for the LLM's semantic read
    and is the Ollama-down path."""
    p = ctx["op_profile"].lower()
    if any(w in p for w in ("load", "preprocess", "augment", "decode", "etl")):
        cls = "io"
    elif any(w in p for w in ("all-reduce", "allreduce", "sync", "embedding", "shard",
                              "memory-bound", "bandwidth", "comm")):
        cls = "bandwidth"
    else:                                                # matmul / conv / attention / default
        cls = "compute"
    return {"bottleneck": cls, "justification": f"rule: keyword match on '{ctx['op_profile']}'",
            "_source": "rule"}


def llm_bottleneck(ctx: dict, use_llm: bool = True, model: str = DEFAULT_MODEL,
                   host: str = HOST, cache: dict | None = None) -> dict:
    """Return `{bottleneck, justification, _source}` for a task profile, cached per profile."""
    cache = load_cache() if cache is None else cache
    key = f"bneck|{bottleneck_state_key(ctx)}|{'llm:' + model if use_llm else 'rule'}"
    if key in cache:
        return cache[key]

    out = None
    if use_llm:
        try:
            import ollama
            client = ollama.Client(host=host)
            resp = client.chat(
                model=model, format="json",
                options={"temperature": 0, "num_predict": 100},
                messages=[{"role": "system", "content": SYSTEM_BOTTLENECK},
                          {"role": "user", "content": _bottleneck_prompt(ctx)}],
            )
            obj = _parse(resp.message.content)
            if obj is not None:
                cls = str(obj.get("bottleneck", "")).strip().lower()
                if cls not in BOTTLENECKS:
                    cls = "compute"
                why = str(obj.get("justification", "")).strip().replace("\n", " ")[:200]
                out = {"bottleneck": cls, "justification": why, "_source": f"llm:{model}"}
        except Exception as e:
            print(f"  ! llm_bottleneck fallback for [{bottleneck_state_key(ctx)}]: "
                  f"{type(e).__name__}: {e}")
    if out is None:
        out = _rule_bottleneck(ctx)

    cache[key] = out
    return out


def affinity_matrix(classes: dict[str, str], penalty: float = 1.0) -> dict[tuple[str, str], float]:
    """Turn per-task bottleneck CLASSES into a soft same-node affinity for the ILP. Code owns the
    numbers: a pair of tasks with the SAME contending class (compute+compute or bandwidth+bandwidth)
    gets a NEGATIVE coefficient (-penalty) so the ILP avoids co-locating them; complementary or io
    pairs get no term (0) — they overlap fine. The ILP maximises Σ coeff·same_node, so it spreads
    contenders across nodes and packs complementary tasks together, all inside the hard capacity /
    co-location constraints."""
    aff: dict[tuple[str, str], float] = {}
    for a, b in combinations(sorted(classes), 2):
        ca, cb = classes[a], classes[b]
        if ca == cb and ca in CONTENDS:
            aff[(a, b)] = -penalty
    return aff


def main() -> None:
    profiles = {
        "t_resnet":  "ResNet-50 training, dense conv forward/backward",
        "t_bert":    "BERT pretraining with large all-reduce parameter sync each step",
        "t_dlrm":    "DLRM with huge embedding-table lookups, memory-bandwidth bound",
        "t_loader":  "ImageNet data loading, decode and augmentation pipeline",
        "t_gpt":     "GPT attention + matmul heavy transformer block",
    }
    cache = load_cache()
    classes = {}
    print("=== task -> bottleneck class (rule fallback) ===")
    for jid, prof in profiles.items():
        d = llm_bottleneck({"op_profile": prof}, use_llm=False, cache=cache)
        classes[jid] = d["bottleneck"]
        print(f"  {jid:10s} -> {d['bottleneck']:9s}  ({d['_source']})  {d['justification']}")
    print("\n=== affinity matrix (negative = avoid same node) ===")
    for (a, b), c in affinity_matrix(classes, penalty=1.0).items():
        print(f"  ({a}, {b}) [{classes[a]}+{classes[b]}] -> {c:+.1f}")


if __name__ == "__main__":
    main()
