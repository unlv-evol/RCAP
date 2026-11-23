"""Per-configuration table of symbol_check.py outputs over a unit list (default: the 2,604 shared units).

    symbol_summary.py OUT1.jsonl [OUT2.jsonl ...] [--units shared-units.tsv] [--md table.md] [--json summary.json]

Each output file is one configuration (its rows' `config`; the file name when several files share one config,
e.g. one per model). A listed unit with no row counts as NO_CANDIDATE (missing). Percentages of the
verdicts are of the real changes (candidate present, not an echo, no leaked placeholder).
"""
import argparse
import collections
import json
from pathlib import Path

from symbol_check import read_units_tsv

VERDICTS = ["SYMBOLS_OK", "NEEDS_PR_EDITS", "ARITY_MISMATCH", "UNRESOLVED"]
PASS = ["NO_CANDIDATE", "ECHO", "PLACEHOLDER_LEAK", "NO_TARGET"]


def pct(a, b):
    return f"{100 * a / b:.1f}%" if b else "-"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outs", nargs="+")
    ap.add_argument("--units", default=str(Path(__file__).resolve().parent / "shared-units.tsv"))
    ap.add_argument("--all-rows", action="store_true", help="no unit list: every row of each file")
    ap.add_argument("--md")
    ap.add_argument("--json")
    a = ap.parse_args()
    keep = None if a.all_rows else read_units_tsv(a.units)
    table = []
    names_unres = {}
    for path in a.outs:
        rows = {r["unit_id"]: r for r in (json.loads(l) for l in open(path) if l.strip())}
        cfgs = {r["config"] for r in rows.values()}
        label = Path(path).stem if len(a.outs) > 1 else (cfgs.pop() if len(cfgs) == 1 else Path(path).stem)
        units = keep if keep is not None else set(rows)
        c = collections.Counter()
        unres = collections.Counter()
        for u in units:
            r = rows.get(u)
            v = r["verdict"] if r else "NO_CANDIDATE"
            c[v] += 1
            for x in (r or {}).get("unresolved", []):
                unres[x.split(":")[0]] += 1
        real = sum(c[v] for v in VERDICTS)
        row = {"config": label, "units": len(units), "missing": len(units - set(rows)), "real_change": real,
               **{v: c[v] for v in VERDICTS + PASS}}
        table.append(row)
        names_unres[label] = unres.most_common(15)
    hdr = ["Config", "Units", "Real change", "SYMBOLS_OK", "NEEDS_PR_EDITS", "ARITY_MISMATCH", "UNRESOLVED",
           "OK or PR (symbol-clean)", "No candidate", "Echo", "Leak"]
    lines = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in table:
        real = r["real_change"]
        lines.append("| " + " | ".join(str(x) for x in [
            r["config"], r["units"], f"{real} ({pct(real, r['units'])})",
            f"{r['SYMBOLS_OK']} ({pct(r['SYMBOLS_OK'], real)})",
            f"{r['NEEDS_PR_EDITS']} ({pct(r['NEEDS_PR_EDITS'], real)})",
            f"{r['ARITY_MISMATCH']} ({pct(r['ARITY_MISMATCH'], real)})",
            f"{r['UNRESOLVED']} ({pct(r['UNRESOLVED'], real)})",
            f"{r['SYMBOLS_OK'] + r['NEEDS_PR_EDITS']} ({pct(r['SYMBOLS_OK'] + r['NEEDS_PR_EDITS'], r['units'])} of units)",
            r["NO_CANDIDATE"] + r["NO_TARGET"], r["ECHO"], r["PLACEHOLDER_LEAK"]]) + " |")
    md = "\n".join(lines)
    print(md)
    for k, v in names_unres.items():
        print(f"\n{k}: most frequent unresolved names {v}")
    if a.md:
        Path(a.md).write_text(md + "\n")
    if a.json:
        Path(a.json).write_text(json.dumps({"table": table, "unresolved_names": names_unres}, indent=1))


if __name__ == "__main__":
    main()
