#!/usr/bin/env python3
"""Task 2: split VALID verdicts into code-changing vs comment/whitespace-only vs import-only.

Per spliced unit, compare target_original with candidate_expanded after blanking every
tree-sitter comment subtree and lexing what remains into tokens (\\w+ | single non-space
char), so whitespace and comments never count. A second pass also blanks import
declarations. Unit class:
  NO_CODE_CHANGE  token streams equal (comments/whitespace only)
  IMPORT_ONLY     differ, but equal once imports are removed
  CODE            differ outside imports
Row (SAP) class = strongest unit class among units that spliced "ok".
Usage: classify_valid.py [verdicts.jsonl]   (run with the rcap venv: needs salp.structural)
"""
import collections, difflib, json, re, sys
from pathlib import Path
from salp.structural import grammar_for, parse

V = Path("/home/adam/Documents/work/rcap-class-project/run477/validate")
VERD = Path(sys.argv[1]) if len(sys.argv) > 1 else V / "verdicts-linkedin__kafka.jsonl"
OUT = V / "valid-classified.jsonl"
TOK = re.compile(r"\w+|\S")
RANK = {"NO_CODE_CHANGE": 0, "IMPORT_ONLY": 1, "CODE": 2}

def ranges(node, pred, acc):
    if pred(node.type): acc.append((node.start_byte, node.end_byte)); return acc
    for c in node.children: ranges(c, pred, acc)
    return acc

def tokens(text, ext, drop_imports):
    b = bytearray(text.encode("utf-8"))
    tree = parse(text, grammar_for(ext))
    pred = (lambda t: "comment" in t or t.startswith("import")) if drop_imports else (lambda t: "comment" in t)
    for s, e in ranges(tree.root_node, pred, []):
        b[s:e] = b" " * (e - s)
    return TOK.findall(b.decode("utf-8", errors="replace"))

def unit_class(orig, cand, ext):
    a, b = tokens(orig, ext, False), tokens(cand, ext, False)
    if a == b: return "NO_CODE_CHANGE", 0
    a2, b2 = tokens(orig, ext, True), tokens(cand, ext, True)
    if a2 == b2: return "IMPORT_ONLY", 0
    sm = difflib.SequenceMatcher(None, a2, b2, autojunk=False)
    changed = sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes() if op != "equal")
    return "CODE", changed

cands = {}
for p in (V / "candidates").glob("*.json"):
    r = json.loads(p.read_text()); cands[r["key"]] = r
rows = [json.loads(l) for l in open(VERD)]
out, per_row, per_unit = [], collections.Counter(), collections.Counter()
for v in rows:
    if v["verdict"] != "VALID": continue
    c = cands[v["key"]]; ext = Path(c["target_file"]).suffix.lstrip(".")
    units = []
    for u, s in zip(c["units"], v["splice"]):
        if s != "ok": units.append({"entity": u["entity"], "class": "NOT_SPLICED"}); continue
        cls, n = unit_class(u["target_original"], u["candidate_expanded"], ext)
        units.append({"entity": u["entity"], "class": cls, "changed_tokens": n}); per_unit[cls] += 1
    spliced = [x for x in units if x["class"] != "NOT_SPLICED"]
    row_cls = max((x["class"] for x in spliced), key=RANK.get)
    per_row[row_cls] += 1
    out.append({"key": v["key"], "file": v["file"], "test_file": "/src/test/" in v["file"], "class": row_cls,
                "changed_tokens": sum(x.get("changed_tokens", 0) for x in spliced), "units": units})
OUT.write_text("".join(json.dumps(o) + "\n" for o in out))
print(f"VALID rows: {len(out)}  ->  {dict(per_row)}")
print(f"spliced units: {dict(per_unit)}")
code = sorted(o["changed_tokens"] for o in out if o["class"] == "CODE")
if code: print(f"CODE rows changed-token median {code[len(code)//2]}, min {code[0]}, max {code[-1]}, <=5 tokens: {sum(x <= 5 for x in code)}")
print("by test/main:", collections.Counter((o["test_file"], o["class"]) for o in out))
