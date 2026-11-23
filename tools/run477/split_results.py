#!/usr/bin/env python3
"""Final MO/ED/SP label per SAP (added-line fraction; deleted lines only for
pure-deletion changes) joined with results-477-v2.jsonl."""
import json, collections
RUN="/home/adam/Documents/work/rcap-class-project/run477"
L=[json.loads(l) for l in open(f"{RUN}/mo-relabel/mo_labels.jsonl")]
R={}
for l in open(f"{RUN}/results-477-v2.jsonl"):
    j=json.loads(l); R[j["key"]]=j
def frac(r):
    if r.get("n_add"): return r["add_present"]/r["n_add"]
    if r.get("n_del"): return r["del_absent"]/r["n_del"]
    return None
def lab(f,lo=0.2,hi=0.8):
    return "UNKNOWN" if f is None else "ED" if f>=hi else "MO" if f<=lo else "SP"
out=[]
for r in L:
    f=frac(r); res=R[r["key"]]
    ch=[c for c in (res.get("changed") or []) if c is not None] if res["outcome"]=="completion" else []
    out.append({"key":r["key"],"label":lab(f),"applied_fraction":None if f is None else round(f,4),
                "outcome":res["outcome"],"units_completed":len(ch),"units_edited":sum(ch),"units_echoed":len(ch)-sum(ch)})
with open(f"{RUN}/mo-relabel/labels_final.jsonl","w") as fh:
    for o in out: fh.write(json.dumps(o)+"\n")
S={}
for cls in ["MO","SP","ED","UNKNOWN"]:
    rows=[o for o in out if o["label"]==cls]
    oc=collections.Counter(o["outcome"] for o in rows)
    S[cls]=dict(saps=len(rows),completion=oc["completion"],outcomes=dict(oc),
      units_completed=sum(o["units_completed"] for o in rows),units_edited=sum(o["units_edited"] for o in rows),
      units_echoed=sum(o["units_echoed"] for o in rows))
for lo,hi in [(0.1,0.9),(0.34,0.67)]:
    S[f"sensitivity_{lo}_{hi}"]=dict(collections.Counter(lab(o["applied_fraction"],lo,hi) for o in out))
json.dump(S,open(f"{RUN}/mo-relabel/summary.json","w"),indent=1)
print(json.dumps(S,indent=1))
