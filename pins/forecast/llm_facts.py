"""
The LLM 'reasoner' that sits ON TOP of the deterministic forecaster (pins/forecast/model.py).

Per the project hinge (research_plan.md §6, cold-start; CLAUDE.md "LLM reasons, code decides"):
the LLM NEVER emits a number. Here it reads only the job's MODEL-TYPE LABEL — e.g. 'dimenet',
'vgg19', 'bert-base-uncased' — which is the single piece of task metadata the *anonymised* MIT
Supercloud traces preserve (no submission scripts survive; see data/fetch_supercloud.py). From
that label the LLM emits STRUCTURED, CATEGORICAL FACTS about the architecture's *dynamic resource
regime*. Deterministic code (`facts_to_vec`) then encodes those facts into a fixed-length static
covariate vector, which `model.py` concatenates onto every history step (the Temporal-Fusion-
Transformer "static covariate" pattern) so self-attention can condition the forecast on *what kind
of job this is*.

Why facts and not a one-hot of the label? Transferability. A one-hot of 'pna' is useless if only
a few pna jobs were seen in training. Facts like {family: gnn, gpu_mem_pattern: bursty} are SHARED
across classes, so a rare class still inherits the regime of its family. The Step-4 ablation
(attn+facts vs attn+onehot vs plain attn) tests exactly this.

The LLM runs ONCE per distinct label (cached to facts_cache.json), out of the hot loop, and
degrades gracefully:
  Ollama up + valid JSON      -> LLM facts
  Ollama down / unparseable   -> rule-based family map (RULE_FAMILY) with neutral defaults
  family unknown              -> zero vector (forecaster behaves as if facts were absent)

Run:  .venv-forecast/bin/python -m pins.forecast.llm_facts            # smoke: facts+vec per label
      .venv-forecast/bin/python -m pins.forecast.llm_facts --no-llm   # rule fallback only
"""
from __future__ import annotations

import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_PATH = os.path.join(HERE, "facts_cache.json")
DEFAULT_MODEL = "qwen2.5:3b"
HOST = "http://localhost:11434"

# ---- the categorical facts schema (LLM extracts these; code encodes them) ----
# Each field: allowed values in a FIXED order. The encoding below depends on this order,
# so append-only if you extend it (never reorder) or old caches/checkpoints misalign.
FAMILIES = ["cnn", "transformer", "gnn", "unet", "other"]
SCHEMA = {
    "family":            FAMILIES,                       # one-hot (len 5)
    "gpu_util_level":    ["low", "medium", "high"],       # ordinal -> 0, .5, 1
    "gpu_util_dynamics": ["steady", "bursty"],            # ordinal -> 0, 1
    "gpu_mem_pattern":   ["flat", "grows", "sawtooth"],   # one-hot (len 3)
    "cpu_role":          ["light", "moderate", "heavy"],  # ordinal -> 0, .5, 1
    "phase_structure":   ["single", "epochal"],           # ordinal -> 0, 1
}
FEATURE_DIM = len(FAMILIES) + 1 + 1 + 3 + 1 + 1           # = 12

# Rule-based family fallback (substring match on the lowercased label). Only the family is
# safe to infer by string; the regime fields fall back to neutral defaults when the LLM is down.
RULE_FAMILY = [
    (("vgg", "inception", "resnet", "alexnet", "mobilenet"), "cnn"),
    (("bert", "gpt", "distilbert", "transformer", "t5", "roberta"), "transformer"),
    (("schnet", "dimenet", "pna", "gcn", "gat", "gin", "conv"), "gnn"),  # molecular GNNs
    (("u3-", "u4-", "u5-", "unet", "u-net"), "unet"),
]
NEUTRAL_DEFAULTS = {
    "gpu_util_level": "medium", "gpu_util_dynamics": "steady",
    "gpu_mem_pattern": "flat", "cpu_role": "moderate", "phase_structure": "epochal",
}

SYSTEM = (
    "You are a deep-learning systems analyst. Given ONLY a model/architecture NAME, you describe "
    "the DYNAMIC RESOURCE REGIME a training run of that model typically exhibits over time. You do "
    "NOT estimate any numbers (no GB, no %, no seconds). Respond with ONLY this JSON object, using "
    "EXACTLY the allowed string values:\n"
    '{"family": "cnn|transformer|gnn|unet|other", '
    '"gpu_util_level": "low|medium|high", '
    '"gpu_util_dynamics": "steady|bursty", '
    '"gpu_mem_pattern": "flat|grows|sawtooth", '
    '"cpu_role": "light|moderate|heavy", '
    '"phase_structure": "single|epochal"}\n'
    "Guidance: family = architecture class. gpu_util_level = typical GPU utilisation while the "
    "step is running. gpu_util_dynamics = 'bursty' if utilisation oscillates a lot between steps "
    "(common for GNNs with variable graph sizes / heavy CPU batching), else 'steady'. "
    "gpu_mem_pattern = 'flat' if memory is allocated once and held, 'grows' if it climbs, "
    "'sawtooth' if it cycles. cpu_role = how heavy the data-loading / preprocessing pressure is. "
    "phase_structure = 'epochal' if the run repeats per-epoch cycles, 'single' if it is one phase."
)


def _coerce(obj: dict) -> dict:
    """Keep only known fields, snap each to an allowed value (else neutral default / 'other')."""
    out = {}
    for field, allowed in SCHEMA.items():
        v = str(obj.get(field, "")).strip().lower()
        if v in allowed:
            out[field] = v
        elif field == "family":
            out[field] = "other"
        else:
            out[field] = NEUTRAL_DEFAULTS[field]
    return out


def _parse(text: str) -> dict | None:
    try:
        obj = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except Exception:
            return None
    return _coerce(obj) if isinstance(obj, dict) else None


def _rule_facts(label: str) -> dict:
    """Family from a substring rule + neutral regime defaults. Always returns a full dict."""
    lab = label.lower()
    family = "other"
    for keys, fam in RULE_FAMILY:
        if any(k in lab for k in keys):
            family = fam
            break
    return {"family": family, **NEUTRAL_DEFAULTS}


# ---- cache (one query per distinct label; persisted for reproducibility/speed) ----
def _load_cache() -> dict:
    if os.path.exists(CACHE_PATH):
        try:
            return json.load(open(CACHE_PATH))
        except Exception:
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    tmp = CACHE_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cache, f, indent=2, sort_keys=True)
    os.replace(tmp, CACHE_PATH)


def get_facts(label: str, use_llm: bool = True, model: str = DEFAULT_MODEL,
              host: str = HOST, cache: dict | None = None) -> dict:
    """Structured regime facts for one model-type label. Cached per (label, source)."""
    own_cache = cache is None
    cache = _load_cache() if own_cache else cache
    key = f"{label}|{'llm:' + model if use_llm else 'rule'}"
    if key in cache:
        return cache[key]

    facts = None
    if use_llm:
        try:
            import ollama
            client = ollama.Client(host=host)
            resp = client.chat(
                model=model, format="json",
                options={"temperature": 0, "num_predict": 160},
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": f"Model name: {label}\nDescribe its regime."}],
            )
            facts = _parse(resp.message.content)
        except Exception as e:                       # Ollama down / network / bad JSON
            print(f"  ! llm_facts fallback for '{label}': {type(e).__name__}: {e}")
    if facts is None:                                # graceful degradation -> rules
        facts = _rule_facts(label)
        facts["_source"] = "rule"
    else:
        # HYBRID (the §5 lesson): the small LLM reliably describes the qualitative regime but
        # is flaky on the single `family` classification (returns 'other' for clear CNNs/GNNs).
        # `family` is exactly what a deterministic string rule gets right for these known names,
        # so let the rule decide it and keep the LLM only for the regime fields it adds value on.
        rule_fam = _rule_facts(label)["family"]
        if rule_fam != "other":
            facts["family"] = rule_fam
        facts["_source"] = f"llm:{model}+rulefam"

    cache[key] = facts
    if own_cache:
        _save_cache(cache)
    return facts


def facts_to_vec(facts: dict) -> np.ndarray:
    """Deterministic encode -> (FEATURE_DIM,) float32. No LLM, no learning; pure lookup."""
    v: list[float] = []
    fam = facts.get("family", "other")
    v += [1.0 if fam == f else 0.0 for f in FAMILIES]                    # one-hot family
    v.append({"low": 0.0, "medium": 0.5, "high": 1.0}[facts["gpu_util_level"]])
    v.append({"steady": 0.0, "bursty": 1.0}[facts["gpu_util_dynamics"]])
    pat = facts["gpu_mem_pattern"]
    v += [1.0 if pat == p else 0.0 for p in SCHEMA["gpu_mem_pattern"]]   # one-hot mem pattern
    v.append({"light": 0.0, "moderate": 0.5, "heavy": 1.0}[facts["cpu_role"]])
    v.append({"single": 0.0, "epochal": 1.0}[facts["phase_structure"]])
    return np.asarray(v, dtype=np.float32)


def label_vectors(labels, use_llm: bool = True, model: str = DEFAULT_MODEL,
                  host: str = HOST) -> dict[str, np.ndarray]:
    """Map each distinct label -> its static-covariate vector (one LLM call per label)."""
    cache = _load_cache()
    out = {}
    for lab in sorted(set(labels)):
        out[lab] = facts_to_vec(get_facts(lab, use_llm, model, host, cache))
    _save_cache(cache)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-llm", action="store_true", help="use the rule fallback only")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    args = ap.parse_args()

    probe = ["vgg19", "resnet152", "inception4", "bert-base-uncased",
             "distilbert-base-uncased", "dimenet", "schnet", "pna", "conv", "U3-128"]
    print(f"facts source: {'rule' if args.no_llm else args.model}   FEATURE_DIM={FEATURE_DIM}\n")
    for lab in probe:
        f = get_facts(lab, use_llm=not args.no_llm, model=args.model)
        vec = facts_to_vec(f)
        regime = {k: f[k] for k in SCHEMA}
        print(f"{lab:24s} {f['_source']:12s} {regime}")
        print(f"{'':24s} vec={np.round(vec, 2)}")
