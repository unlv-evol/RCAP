"""PPatHF-style vs RCAP on the same function units (roadmap item 9), before build/test.

Both answers of a unit are labelled by RCAP's own scorer (rcap.scoring.score_unit) against the same texts: the
fork's function (the SAP's target payload) and upstream's before/after. RCAP's answer is the one its run saved
(ordered/attempts/<PR>/<sap>.<fn>.candidate.json, placeholders expanded); a unit RCAP did not answer carries
the outcome its run recorded. A PPatHF-style answer that still holds a /* placeholder_N */ the model dropped or
moved is a PLACEHOLDER_LEAK, as RCAP's own leaked placeholders are.

    python compare.py <ppathf_out.jsonl> <rcap run dir (has ordered/)> <SALP pair dir> <out.json>
"""
import collections
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from rcap import fidelity as fd  # noqa: E402
from rcap.scoring import NO_CANDIDATE, score_unit  # noqa: E402

PPATHF_PH = re.compile(r"placeholder_\d+")


def rcap_answers(ordered: Path):
    """unit id (PR/sap::fn) -> (expanded candidate or None, outcome / status)."""
    ans = {}
    for line in open(ordered / "pr-results.jsonl"):
        row = json.loads(line)
        for s in row.get("saps", []):
            for m in s.get("missing_units", []):
                ans[row["pr"] + "/" + m["unit"]] = (None, m["outcome"])
    for f in ordered.glob("attempts/PR-*/*.candidate.json"):
        c = json.loads(f.read_text())
        ans[f.parent.name + "/" + c["unit"]] = (c.get("candidate_expanded"), c.get("status"))
    return ans


def label(fork, cand, before, after):
    if cand and PPATHF_PH.search(cand):
        return fd.PLACEHOLDER_LEAK
    return score_unit(fork, cand, before, after, language="java").label


def main():
    ppathf_out, run_dir, pair_dir, out = map(Path, sys.argv[1:5])
    rc = rcap_answers(run_dir / "ordered")
    units = []
    for r in map(json.loads, open(ppathf_out)):
        fdir = pair_dir / r["pr"] / r["sap"] / "functions" / r["fn"]
        before, after, fork = ((fdir / f"{k}.java").read_text(errors="replace")
                               for k in ("source.before", "source.after", "target"))
        rcand, rstatus = rc.get(r["unit_id"], (None, "not in the RCAP run"))
        units.append({"unit": r["unit_id"], "placeholders": r["placeholders"],
                      "ppathf_outcome": r["outcome"], "ppathf_label": label(fork, r.get("candidate"), before, after),
                      "rcap_outcome": rstatus, "rcap_label": label(fork, rcand, before, after)})
    t = collections.Counter
    summary = {"units": len(units),
               "ppathf": dict(t(u["ppathf_label"] for u in units).most_common()),
               "rcap": dict(t(u["rcap_label"] for u in units).most_common()),
               "pairs": dict(t(f"{u['rcap_label']} -> {u['ppathf_label']}" for u in units).most_common())}
    code = fd.FAITHFUL_CODE
    summary["faithful_code"] = {"rcap": sum(u["rcap_label"] == code for u in units),
                                "ppathf": sum(u["ppathf_label"] == code for u in units),
                                "both": sum(u["rcap_label"] == code == u["ppathf_label"] for u in units),
                                "only_rcap": sum(u["rcap_label"] == code != u["ppathf_label"] for u in units),
                                "only_ppathf": sum(u["ppathf_label"] == code != u["rcap_label"] for u in units)}
    Path(out).write_text(json.dumps({"summary": summary, "units": units}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
