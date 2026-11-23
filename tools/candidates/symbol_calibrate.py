"""Agreement of symbol_check.py verdicts with a local build (order_pr.py run dir).

Build truth per unit (only units whose candidate the build applied):
  compiled  VALID_PR / TEST_FAIL / TEST_FAIL_UNATTRIBUTED (the PR state with this unit compiled)
  cfs       BUILD_FAIL charged with `cannot find symbol` (javac's symbol names read from the round's build log)
  cba       BUILD_FAIL charged with `cannot be applied` / `no suitable method|constructor` / `constructor .. in ..`
  other     any other BUILD_FAIL
Units BLOCKED_BY / INCONCLUSIVE / INTEGRATION_FAIL / NOTHING_TO_PORT ... carry no compile verdict and are left out.

    symbol_calibrate.py <checker out.jsonl> <ordered dir> [--pair P] [--show N] [--json OUT]
"""
import argparse
import collections
import json
import re
from pathlib import Path

COMPILED = {"VALID_PR", "TEST_FAIL", "TEST_FAIL_UNATTRIBUTED"}
CBA = re.compile(r"cannot be applied|no suitable (method|constructor)|constructor \S+ in (class|enum) \S+ cannot")
SYM = re.compile(r"^\s*symbol:\s+(\w+)\s+([\w$]+)")
ERR = re.compile(r"^(/\S+?\.(?:java|scala)):(\d+): error: (.*)$")


def log_symbols(ordered: Path, pr: str):
    """{(relpath suffix, line): set(symbol names)} of every `cannot find symbol` in the PR's build logs."""
    out = collections.defaultdict(set)
    for log in (ordered / "rounds" / pr).glob("*.build.log"):
        lines = log.read_text(errors="replace").splitlines()
        for i, ln in enumerate(lines):
            m = ERR.match(ln)
            if not m or "cannot find symbol" not in m.group(3):
                continue
            for k in range(i + 1, min(i + 6, len(lines))):
                s = SYM.match(lines[k])
                if s:
                    out[(m.group(1), int(m.group(2)))].add(s.group(2))
                    break
    return out


def truth(ordered: Path, pair_default=None):
    units = {}
    for line in open(ordered / "pr-results.jsonl"):
        row = json.loads(line)
        pr, pair = row["pr"], row.get("pair") or pair_default
        syms = None
        charges = collections.defaultdict(list)
        for a in row.get("attribution", []):
            for ev in a.get("evidence", []):
                for c in ev.get("charges", []):
                    if c.get("certain") and len(c.get("suspects", [])) == 1:
                        charges[c["suspects"][0]].append(c["error"])
        for key, u in row.get("units", {}).items():
            st = u["status"]
            uid = f"{pair}/{pr}/{key}"
            errs = list(charges.get(key, []))
            if u.get("detail", {}).get("error"):
                errs.append(u["detail"]["error"])
            # only errors in the unit's own file count (attribution also charges a unit for errors it causes
            # elsewhere, e.g. a removed method's callers)
            own = [e for e in errs if e.startswith(u.get("file", "?") + ":")]
            foreign = len(errs) - len(own)
            errs = own
            names = set()
            if any("cannot find symbol" in e for e in errs):
                if syms is None:
                    syms = log_symbols(ordered, pr)
                for e in errs:
                    m = re.match(r"(\S+?):(\d+): cannot find symbol", e)
                    if m:
                        for (path, ln), ns in syms.items():
                            if path.endswith("/" + m.group(1)) and ln == int(m.group(2)):
                                names |= ns
            if st in COMPILED:
                cls = "compiled"
            elif st == "BUILD_FAIL":
                if any("cannot find symbol" in e for e in errs):
                    cls = "cfs"
                elif any(CBA.search(e) for e in errs):
                    cls = "cba"
                else:
                    cls = "other"
            else:
                cls = None
            if st == "BUILD_FAIL" and not errs and foreign:
                cls = "foreign"
            fn_unit = u.get("file", "").endswith(".java") and not re.search(r"_(fn\d+|imports)$", key)
            units[uid] = {"status": st, "class": cls, "errors": errs[:4], "symbols": sorted(names),
                          "override": "override" in u.get("flags", []), "function": fn_unit}
    return units


_TARGETS = {}


def target_has_all(uid, symbols, ordered):
    """Every missing symbol already occurs in the fork's original function: the unit did not introduce the
    reference, another unit's edit (a removed / renamed declaration) broke it."""
    pair, pr, key = uid.split("/", 2)
    if not _TARGETS:
        for c in Path(ordered).glob("attempts/PR-*/*.candidate.json"):
            _TARGETS[c.parent.name + "/" + json.loads(c.read_text())["unit"]] = \
                c.with_name(c.name.replace(".candidate.json", ".target_original.txt"))
    f = _TARGETS.get(f"{pr}/{key}")
    if f is None or not f.exists():
        return False
    ids = set(re.findall(r"[A-Za-z_$][\w$]*", f.read_text(errors="replace")))
    return all(s in ids for s in symbols)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checked")
    ap.add_argument("ordered")
    ap.add_argument("--pair", default="linkedinKafka-apacheKafka")
    ap.add_argument("--only-override", action="store_true", help="only units whose candidate came from the override "
                    "treatment (Direct / PPatHF), not RCAP's own non-function units")
    ap.add_argument("--functions", action="store_true", help="only Java function units (no *_fnN / *_imports "
                    "chunks, no Scala): the kind of unit the 2,604-unit study sends")
    ap.add_argument("--show", type=int, default=0)
    ap.add_argument("--json")
    a = ap.parse_args()
    t = truth(Path(a.ordered), a.pair)
    rows = {r["unit_id"]: r for r in map(json.loads, open(a.checked))}
    tab = collections.Counter()
    names_hit = names_total = 0
    disagreements = {"compiled_unresolved": [], "cfs_ok": []}
    for uid, r in rows.items():
        g = t.get(uid)
        if g is None or g["class"] is None or (a.only_override and not g["override"]):
            continue
        if a.functions and not g["function"]:
            continue
        v = r["verdict"]
        if g["class"] == "cfs" and g["symbols"] and target_has_all(uid, g["symbols"], a.ordered):
            g = {**g, "class": "cfs_sibling"}
        tab[(g["class"], v)] += 1
        if g["class"] == "cfs" and g["symbols"]:
            flagged = {x.split(":")[0] for x in r.get("unresolved", []) + r.get("pr_only", [])}
            names_total += len(g["symbols"])
            names_hit += len(set(g["symbols"]) & flagged)
        if g["class"] == "compiled" and v == "UNRESOLVED":
            disagreements["compiled_unresolved"].append((uid, r.get("unresolved")))
        if g["class"] == "cfs" and v in ("SYMBOLS_OK",):
            disagreements["cfs_ok"].append((uid, g["symbols"], g["errors"][:1]))
    verdicts = ["SYMBOLS_OK", "NEEDS_PR_EDITS", "ARITY_MISMATCH", "UNRESOLVED", "ECHO", "NO_CANDIDATE",
                "PLACEHOLDER_LEAK"]
    classes = ["compiled", "cfs", "cfs_sibling", "cba", "other", "foreign"]
    print("build class \\ verdict | " + " | ".join(verdicts) + " | total")
    for c in classes:
        tot = sum(tab[(c, v)] for v in verdicts)
        print(f"{c:9} | " + " | ".join(str(tab[(c, v)]) for v in verdicts) + f" | {tot}")
    comp = sum(tab[("compiled", v)] for v in verdicts if v not in ("ECHO", "NO_CANDIDATE", "PLACEHOLDER_LEAK"))
    comp_un = tab[("compiled", "UNRESOLVED")]
    comp_flag = comp_un + tab[("compiled", "ARITY_MISMATCH")]
    cfs = sum(tab[("cfs", v)] for v in verdicts if v not in ("ECHO", "NO_CANDIDATE", "PLACEHOLDER_LEAK"))
    cfs_un = tab[("cfs", "UNRESOLVED")]
    cfs_flag = cfs_un + tab[("cfs", "NEEDS_PR_EDITS")]
    cba = sum(tab[("cba", v)] for v in verdicts if v not in ("ECHO", "NO_CANDIDATE", "PLACEHOLDER_LEAK"))
    s = {"compiled_checked": comp, "compiled_unresolved": comp_un, "false_alarm_rate": round(comp_un / comp, 3) if comp else None,
         "compiled_flagged_incl_arity": comp_flag,
         "compiled_symbols_ok_or_pr": comp - comp_flag,
         "cfs_checked": cfs, "cfs_unresolved": cfs_un, "recall_unresolved": round(cfs_un / cfs, 3) if cfs else None,
         "cfs_unresolved_or_pr": cfs_flag, "recall_unresolved_or_pr": round(cfs_flag / cfs, 3) if cfs else None,
         "cfs_symbol_names_flagged": f"{names_hit}/{names_total}",
         "cba_checked": cba, "cba_arity_or_unresolved": tab[("cba", "ARITY_MISMATCH")] + tab[("cba", "UNRESOLVED")]}
    print(json.dumps(s, indent=1))
    if a.show:
        for k, v in disagreements.items():
            print(f"\n== {k}: {len(v)}")
            for x in v[:a.show]:
                print("  ", json.dumps(x)[:400])
    if a.json:
        Path(a.json).write_text(json.dumps({"summary": s, "table": {f"{c}|{v}": n for (c, v), n in tab.items()},
                                            "disagreements": disagreements}, indent=1))


if __name__ == "__main__":
    main()
