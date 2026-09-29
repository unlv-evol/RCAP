#!/usr/bin/env python3
"""Relabel each SAP of the 477 run as MO / ED / SP, GACPD-style.

make_gacpd.py labels a file MO whenever it merely EXISTS in the fork; real GACPD
calls it a missed opportunity only if the upstream change is ABSENT from the fork
(ED = fork already has it, SP = partly). This approximates that check locally:
for the PR's whole-file patch, count "applied" signals in the fork's file at the
target commit -- each meaningful added line already present, each meaningful
deleted line already gone -- and take applied_fraction = applied / signals.

Lines are compared stripped of whitespace; trivial lines (braces, blank, lone
comment markers, short tokens) carry no signal and are ignored. A line that is
both added and deleted (moved) is ignored. This is OUR approximation, not GACPD.
Output: mo_labels.jsonl, one row per SAP key.
"""
import json, glob, os, re, sys
from pathlib import Path
RUN = Path("/home/adam/Documents/work/rcap-class-project/run477")
TRIVIAL = re.compile(r"^(?:[{}()\[\];,]+|//+|/\*+|\*+/?|\*|@Override|else|try|finally|return;|break;|continue;|\)\s*;|\}\s*else\s*\{|\}\s*\)\s*;?)$")

def meaningful(line):
    s = " ".join(line.split())
    return s if len(s) >= 4 and not TRIVIAL.match(s) else None

def patch_lines(text):
    add, dele = [], []
    for ln in text.splitlines():
        if ln.startswith(("+++", "---")): continue
        if ln[:1] == "+": m = meaningful(ln[1:]); m and add.append(m)
        elif ln[:1] == "-": m = meaningful(ln[1:]); m and dele.append(m)
    both = set(add) & set(dele)
    return [a for a in add if a not in both], [d for d in dele if d not in both]

def classify(frac, lo=0.2, hi=0.8):
    if frac is None: return "UNKNOWN"
    return "ED" if frac >= hi else ("MO" if frac <= lo else "SP")

out = []
for sapdir in sorted(glob.glob(str(RUN / "salp-477-out/*/PR-*/sap-*"))):
    rel = sapdir.split("salp-477-out/")[1]
    sap = json.load(open(f"{sapdir}/sap.json"))
    pr = json.load(open(f"{os.path.dirname(sapdir)}/pr.json"))
    prn = rel.split("/")[1][3:]
    gpair = pr["source_repo"].replace("/", "_") + "-" + pr["target_repo"].replace("/", "_")
    flat = sap["source_file"].replace("/", "_").replace(".", "_")
    fdir = RUN / "gacpd-477" / gpair / f"{prn}_MO" / "MO" / flat
    rec = {"key": rel, "source_file": sap["source_file"]}
    patches = glob.glob(str(fdir / "patch" / "*.patch")); cmps = glob.glob(str(fdir / "cmp" / "*"))
    if not patches or not cmps:
        rec.update(label="UNKNOWN", reason="gacpd files not found"); out.append(rec); continue
    add, dele = patch_lines(open(patches[0], encoding="utf-8", errors="replace").read())
    fork = {m for ln in open(cmps[0], encoding="utf-8", errors="replace").read().splitlines() if (m := meaningful(ln))}
    a_hit = sum(a in fork for a in add); d_gone = sum(d not in fork for d in dele)
    n = len(add) + len(dele)
    frac = (a_hit + d_gone) / n if n else None
    rec.update(n_add=len(add), n_del=len(dele), add_present=a_hit, del_absent=d_gone,
               applied_fraction=None if frac is None else round(frac, 4), label=classify(frac))
    out.append(rec)
with open(RUN / "mo-relabel/mo_labels.jsonl", "w") as fh:
    for r in out: fh.write(json.dumps(r) + "\n")
import collections
print(len(out), collections.Counter(r["label"] for r in out))
