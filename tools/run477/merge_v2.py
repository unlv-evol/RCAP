#!/usr/bin/env python3
"""Merge the per-SAP verdicts with the follow-up analyses into verdicts-<repo>.v2.jsonl + a summary.

Inputs (all in validate/):
  verdicts-linkedin__kafka.jsonl   validate.py output (never modified here)
  buildfail-classified.jsonl       classify_buildfail.py  -> "build_cause" on BUILD_FAIL rows
  valid-classified.jsonl           classify_valid.py      -> "change_class" on VALID rows
  baseline-tests-linkedin__kafka.json   baseline logs, for the INCONCLUSIVE relabel
Relabels (verdict is unchanged; only "reason_v2" is refined):
  INCONCLUSIVE "baseline test did not run cleanly" whose baseline log says gradle found no tests
      -> "no tests in file (test-support class)"
  ... whose baseline failed on the spotless configuration race -> "baseline flaky (spotless config); rerun"
Usage: merge_v2.py
"""
import collections, json
from pathlib import Path

V = Path("/home/adam/Documents/work/rcap-class-project/run477/validate")
rows = [json.loads(l) for l in open(V / "verdicts-linkedin__kafka.jsonl")]
bf = {r["key"]: r for r in map(json.loads, open(V / "buildfail-classified.jsonl"))}
vc = {r["key"]: r for r in map(json.loads, open(V / "valid-classified.jsonl"))}
base = json.loads((V / "baseline-tests-linkedin__kafka.json").read_text())

def base_log(row):
    # keyed by the TEST's project, which differs from the file's for other_project tests
    suffix = "|" + ",".join(row.get("test_class", []))
    return next((b.get("log_tail", "") for k, b in base.items() if k.endswith(suffix)), "")

out = []
for r in rows:
    r = dict(r)
    if r["verdict"] == "BUILD_FAIL":
        b = bf.get(r["key"], {}); r["build_cause"] = b.get("cause", "NO_LOG")
        r["build_error_causes"] = dict(collections.Counter(e["cause"] for e in b.get("errors", [])))
    elif r["verdict"] == "VALID":
        c = vc[r["key"]]; r["change_class"] = c["class"]; r["changed_tokens"] = c["changed_tokens"]
    elif r["verdict"] == "INCONCLUSIVE":
        reason = r.get("reason", "")
        if reason.startswith("baseline test did not run"):
            log = base_log(r)
            reason = ("no tests in file (test-support class)" if "No tests found for given includes" in log else
                      "baseline flaky (spotless config); rerun" if "spotless" in log else reason)
        r["reason_v2"] = reason
    out.append(r)
(V / "verdicts-linkedin__kafka.v2.jsonl").write_text("".join(json.dumps(r) + "\n" for r in out))

n = len(out); v = collections.Counter(r["verdict"] for r in out)
print(f"rows {n}: " + ", ".join(f"{k} {c} ({c / n:.1%})" for k, c in v.most_common()))
print("BUILD_FAIL by cause:", dict(collections.Counter(r["build_cause"] for r in out if "build_cause" in r).most_common()))
print("VALID by change class:", dict(collections.Counter(r["change_class"] for r in out if "change_class" in r).most_common()))
print("INCONCLUSIVE by reason:", dict(collections.Counter(r["reason_v2"] for r in out if "reason_v2" in r).most_common()))
