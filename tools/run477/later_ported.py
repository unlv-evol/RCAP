#!/usr/bin/env python3
"""Step 1 of candidate validation: for each SAP that is a missed opportunity at
its target commit, does the fork's LATEST commit (HEAD of the local mirror)
contain the upstream change? Those SAPs have a real fork-authored port to score
RCAP's candidate against. Same added-line test as classify_mo.py.
If the file is gone at HEAD, look for a single same-named file (rename/move).
Output: later_ported.jsonl + printed summary."""
import json, glob, os, subprocess, collections, sys
sys.path.insert(0, os.path.dirname(__file__))
from classify_mo import meaningful, patch_lines
RUN = "/home/adam/Documents/work/rcap-class-project/run477"
CACHE = "/home/adam/Documents/work/SALP/data/repos"
lab = {json.loads(l)["key"]: json.loads(l) for l in open(f"{RUN}/mo-relabel/labels_final.jsonl")}
def git(g, *a):
    r = subprocess.run(["git", "--git-dir", g, *a], capture_output=True, text=True, errors="replace")
    return r.stdout if r.returncode == 0 else None
trees = {}
def head_tree(g):
    if g not in trees:
        trees[g] = (git(g, "rev-parse", "HEAD").strip(), (git(g, "ls-tree", "-r", "--name-only", "HEAD") or "").splitlines())
    return trees[g]
out = []
for key, L in lab.items():
    if L["label"] not in ("MO", "SP"): continue
    sapdir = f"{RUN}/salp-477-out/{key}"
    sap = json.load(open(f"{sapdir}/sap.json")); pr = json.load(open(f"{os.path.dirname(sapdir)}/pr.json"))
    g = f"{CACHE}/{pr['target_repo'].replace('/', '__')}.git"
    prn = key.split("/")[1][3:]
    gpair = pr["source_repo"].replace("/", "_") + "-" + pr["target_repo"].replace("/", "_")
    fdir = f"{RUN}/gacpd-477/{gpair}/{prn}_MO/MO/" + sap["source_file"].replace("/", "_").replace(".", "_")
    add, dele = patch_lines(open(glob.glob(f"{fdir}/patch/*.patch")[0], errors="replace").read())
    head, files = head_tree(g)
    path, how = sap["target_file"], "same_path"
    text = git(g, "show", f"HEAD:{path}")
    if text is None:
        base = os.path.basename(path); cands = [f for f in files if os.path.basename(f) == base]
        if len(cands) == 1: path, how, text = cands[0], "moved", git(g, "show", f"HEAD:{cands[0]}")
        else: how = "gone" if not cands else f"ambiguous_{len(cands)}"
    rec = {"key": key, "target_label": L["label"], "outcome": L["outcome"], "fork": pr["target_repo"], "head_path_how": how}
    if text is not None:
        fork = {m for ln in text.splitlines() if (m := meaningful(ln))}
        if add: f = sum(a in fork for a in add) / len(add)
        elif dele: f = sum(d not in fork for d in dele) / len(dele)
        else: f = None
        rec["head_fraction"] = None if f is None else round(f, 4)
        rec["later"] = "UNKNOWN" if f is None else "ported" if f >= 0.8 else "partial" if f > 0.2 else "not_ported"
    else:
        rec["later"] = "file_gone" if how == "gone" else "file_ambiguous"
    out.append(rec)
with open(f"{RUN}/mo-relabel/later_ported.jsonl", "w") as fh:
    for r in out: fh.write(json.dumps(r) + "\n")
for lbl in ("MO", "SP"):
    rows = [r for r in out if r["target_label"] == lbl]
    print(lbl, len(rows), dict(collections.Counter(r["later"] for r in rows)))
mo = [r for r in out if r["target_label"] == "MO"]
print("MO by fork:"); 
for f, c in sorted(collections.Counter(r["fork"] for r in mo).items(), key=lambda x: -x[1]):
    rs = [r for r in mo if r["fork"] == f]; print(f"  {f}: {c} MO, later ported {sum(r['later']=='ported' for r in rs)}, partial {sum(r['later']=='partial' for r in rs)}")
print("MO ported & RCAP completed:", sum(1 for r in mo if r["later"] == "ported" and r["outcome"] == "completion"))
print("MO partial & RCAP completed:", sum(1 for r in mo if r["later"] == "partial" and r["outcome"] == "completion"))
print("path:", dict(collections.Counter(r["head_path_how"] for r in out)))
