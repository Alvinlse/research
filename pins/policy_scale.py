"""Zero-shot policy pick vs model scale, via Ollama. No training, no downloads."""
import json, re, sys, collections, numpy as np
from pins.elastisim_bench import POLICY_MENU, POLICY_UNSAFE, POLICY_SELECT
from pins.correction import _ask
from pins.llm_agent import HOST
A=[f"{o}+{z}" for o in POLICY_MENU["ordering"] for z in POLICY_MENU["sizing"]]
safe=[a for a in A if a not in POLICY_UNSAFE]
rows=[json.loads(l) for l in open('runs/dataset_v3/test.jsonl')]
Y=np.array([[r['rewards'][a] for a in safe] for r in rows]); best=Y.max(1)
print("%-14s %8s %8s %7s  %s"%("model","regret","eps5","invalid","picks"))
for model in sys.argv[1:]:
    cache={}; picks=[]
    for i,r in enumerate(rows):
        ans=_ask(POLICY_SELECT, r['messages'][1]['content'], model, HOST, cache, f"scale-{i}", num_predict=200)
        o=str((ans or {}).get("ordering","")).strip(); z=str((ans or {}).get("sizing","")).strip()
        picks.append(f"{o}+{z}" if f"{o}+{z}" in safe else "INVALID")
    idx=[safe.index(p) if p in safe else int(np.argmin(Y[i])) for i,p in enumerate(picks)]
    reg=best-Y[np.arange(len(Y)),idx]
    c=collections.Counter(picks)
    print("%-14s %8.4f %8.3f %7d  %s"%(model,reg.mean(),(reg<=0.05).mean(),
          c['INVALID'],dict(c.most_common(3))))
