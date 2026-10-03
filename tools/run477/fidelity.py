#!/usr/bin/env python3
"""Upstream-fidelity labels for the units of a PR-level run (reprompt_pr.py output), post hoc.

  fidelity.py --run-dir results/round3 --tag k1 [--tag k1-mech ...] [--root W] [--all-final]

For every unit that ends VALID_PR (with --all-final: every unit left applied in the final state)
the applied candidate is compared with the upstream change of its function (rcap.fidelity):
FAITHFUL / PARTIAL / WEAKENED / NO_CHANGE, with evidence. Mechanically applied hunks are upstream
text verbatim, not model output: they are listed per PR as MECHANICAL and never labelled.

Writes <run-dir>/<tag>/fidelity.jsonl (one row per judged unit, one `mech` row per PR) and prints
a summary. Reads only; the default pipeline is not involved.
"""
import argparse, collections, glob, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from rcap.fidelity import judge  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True); ap.add_argument("--tag", action="append", required=True)
ap.add_argument("--root", default=str(Path(__file__).resolve().parents[3]),
                help="directory the SAP dirs in pr-results.jsonl are relative to (the experiment dir W)")
ap.add_argument("--all-final", action="store_true")
a = ap.parse_args()
ROOT = Path(a.root)


def read(p):
    return Path(p).read_text(errors="replace") if p and Path(p).exists() else ""


def unit_texts(run, tag, row, key, u):
    sap, fn = key.split("::")
    adir = run / tag / "attempts" / row["pr"]
    att = u.get("applied_attempt") or u.get("first_success_attempt") or 1
    cand = json.loads(read(adir / f"{sap}.{fn}.attempt-{att:02d}.candidate.json") or "{}").get("candidate_expanded") or ""
    orig = read(adir / f"{sap}.{fn}.target_original.txt")
    sdir = next(x["dir"] for x in row["saps"] if x["sap"] == sap)
    fdir = (ROOT / sdir) / "functions" / fn
    before = next(iter(glob.glob(f"{fdir}/source.before.*")), None)
    after = next(iter(glob.glob(f"{fdir}/source.after.*")), None)
    return orig, cand, read(before), read(after), att


def run_tag(run, tag):
    rows = [json.loads(l) for l in open(run / tag / "pr-results.jsonl")]
    out, labels = [], collections.Counter()
    for row in rows:
        if "units" not in row:
            continue
        us = row["units"]
        # production files of the PR some unit of which is not in the final state (dropped, not
        # applied, or no candidate at all), by class name: a FooTest judged VALID while Foo's
        # upstream change is missing may pin the fork's behaviour instead of upstream's
        src_of = {x["sap"]: x.get("target_file") or "" for x in row["saps"]}
        prod_missing = {Path(u["file"]).stem for u in us.values() if not u["is_test"] and not u["in_final_state"]}
        prod_missing |= {Path(src_of.get(m["sap"], "")).stem for m in row.get("missing_siblings", [])
                         if "/src/main/" in src_of.get(m["sap"], "")}
        for key, u in us.items():
            if not u["in_final_state"] or (u["final_verdict"] != "VALID_PR" and not a.all_final):
                continue
            orig, cand, before, after, att = unit_texts(run, tag, row, key, u)
            f = judge(orig, cand, before, after)
            flags = []
            stem = Path(u["file"]).stem
            if u["is_test"] and stem.endswith("Test") and stem[:-4] in prod_missing:
                flags.append(f"production_unit_of_{stem[:-4]}_missing")
            if not before.strip():
                flags.append("no_upstream_before")
            r = {"pr": row["pr"], "unit": key, "file": u["file"], "is_test": u["is_test"],
                 "final_verdict": u["final_verdict"], "attempt": att, "flags": flags,
                 "mech_in_final_state": len(row.get("final_mech", [])),
                 "mech_same_file": sum(1 for m in row.get("final_mech", [])
                                       if any(c.get("mech_id") == m and c.get("target") == u["file"]
                                              for c in row["coverage"]["hunks"])),
                 **f.as_dict()}
            out.append(r); labels[f.label] += 1
        if row.get("mech_hunks"):
            out.append({"pr": row["pr"], "mech": True, "label": "MECHANICAL",
                        "candidates": row.get("mech_candidates"), "applied_initially": len(row.get("mech_applied_initially", [])),
                        "in_final_state": row.get("final_mech", []),
                        "targets": sorted({c["target"] for c in row["coverage"]["hunks"]
                                           if c.get("mech_id") in row.get("final_mech", [])})})
    (run / tag / "fidelity.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))
    return out, labels


for tag in a.tag:
    out, labels = run_tag(Path(a.run_dir).resolve(), tag)
    print(f"## {tag}: {dict(labels)}")
    for r in out:
        if r.get("mech"):
            continue
        print(f"  {r['label']:9} {r['pr']} {r['unit'][:90]} recall={r['recall']} extra={r['extra_share']} "
              f"{','.join(r['flags'])} {'; '.join(r['reasons'])[:200]}")
        for e in r["evidence"][:3]:
            print(f"      - {e[:220]}")
