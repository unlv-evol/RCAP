"""PPatHF-style vs RCAP after build/test (roadmap item 9): the verdicts of an --override-candidates run against the
RCAP run whose other answers it kept. A real port is a unit that passed (VALID_PR) and whose answer compare.py
labelled FAITHFUL_CODE; strict also allows no line beyond the fork's and upstream's added lines.

    python compare_build.py <compare.py out.json> <override run dir> <rcap run dir> <SALP pair dir> <out.json>
"""
import collections
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

from compare import rcap_answers  # noqa: E402


def lines(text):
    return {x.strip() for x in (text or "").splitlines() if x.strip()}


def strict(cand, fork, before, after):
    return not (lines(cand) - lines(fork) - (lines(after) - lines(before)))


def binom_p(a, b):
    n = a + b
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(a, b) + 1)) / 2 ** n) if n else 1.0


def verdicts(ordered):
    """unit id -> status, for every unit of the run, by PR."""
    out = {}
    for line in open(ordered / "pr-results.jsonl"):
        row = json.loads(line)
        for u, v in row.get("units", {}).items():
            out[row["pr"] + "/" + u] = v.get("status")
        for s in row.get("saps", []):
            for m in s.get("missing_units", []):
                out.setdefault(row["pr"] + "/" + m["unit"], m["outcome"])
    return out


def main():
    cmp_json, over_dir, rcap_dir, pair_dir, out = map(Path, sys.argv[1:6])
    units = json.load(open(cmp_json))["units"]
    ao, ar = rcap_answers(over_dir / "ordered"), rcap_answers(rcap_dir / "ordered")
    rows = []
    for u in units:
        uid = u["unit"]
        pr, rest = uid.split("/", 1)
        sap, fn = rest.split("::", 1)
        fdir = pair_dir / pr / sap / "functions" / fn
        before, after, fork = ((fdir / f"{k}.java").read_text(errors="replace")
                               for k in ("source.before", "source.after", "target"))
        pc, ps = ao[uid]
        rc, rs = ar[uid]
        pv, rv = ps == "VALID_PR", rs == "VALID_PR"
        rows.append({"unit": uid, "ppathf_status": ps, "rcap_status": rs,
                     "ppathf_label": u["ppathf_label"], "rcap_label": u["rcap_label"],
                     "same_answer": pc == rc,
                     "ppathf_real": pv and u["ppathf_label"] == "FAITHFUL_CODE",
                     "rcap_real": rv and u["rcap_label"] == "FAITHFUL_CODE",
                     "ppathf_strict": pv and u["ppathf_label"] == "FAITHFUL_CODE" and strict(pc, fork, before, after),
                     "rcap_strict": rv and u["rcap_label"] == "FAITHFUL_CODE" and strict(rc, fork, before, after),
                     "ppathf_faithful_strict": u["ppathf_label"] == "FAITHFUL_CODE" and strict(pc, fork, before, after),
                     "rcap_faithful_strict": u["rcap_label"] == "FAITHFUL_CODE" and strict(rc, fork, before, after)})
    t = collections.Counter

    def paired(k):
        a = sum(r["ppathf_" + k] and not r["rcap_" + k] for r in rows)
        b = sum(r["rcap_" + k] and not r["ppathf_" + k] for r in rows)
        return {"ppathf": sum(r["ppathf_" + k] for r in rows), "rcap": sum(r["rcap_" + k] for r in rows),
                "only_ppathf": a, "only_rcap": b, "p": binom_p(a, b)}

    ids = {r["unit"] for r in rows}
    vo, vr = verdicts(over_dir / "ordered"), verdicts(rcap_dir / "ordered")
    others = [k for k in vo if k not in ids and k in vr]
    summary = {"units": len(rows), "ppathf_status": dict(t(r["ppathf_status"] for r in rows).most_common()),
               "rcap_status": dict(t(r["rcap_status"] for r in rows).most_common()),
               "real_port": paired("real"), "real_port_strict": paired("strict"),
               "faithful_strict_prebuild": paired("faithful_strict"),
               "same_answer": sum(r["same_answer"] for r in rows),
               "status_pairs": dict(t(f"{r['rcap_status']} -> {r['ppathf_status']}" for r in rows).most_common(25)),
               "kept_units": {"n": len(others), "same": sum(vo[k] == vr[k] for k in others),
                              "changes": dict(t(f"{vr[k]} -> {vo[k]}" for k in others if vo[k] != vr[k]).most_common(15))}}
    Path(out).write_text(json.dumps({"summary": summary, "units": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
