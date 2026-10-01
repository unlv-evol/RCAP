#!/usr/bin/env python3
"""Task 1: attribute each BUILD_FAIL to a cause, from the full compiler logs in build-logs/.

Every compiler error is parsed (javac "path:N: error: MSG" + "symbol:" line; scalac
"[Error] path:N: MSG"; javac under scalac "path:N:C:  error: MSG", symbol read at column C).
Per error:
  error in a file OTHER than the spliced one      -> DEPENDENT_IN_PR (the PR edits that caller too: PR-level fixes it)
                                                  or DEPENDENT_OUTSIDE_PR (fork caller upstream never touched: model)
  "... is not a member of Any"                    -> CASCADE (knock-on of an earlier unresolved symbol; ignored)
  missing symbol S (cannot find symbol / not found: value|type / S is not a member of T /
  package S does not exist / object s is not a member of package P => package P.s). Look for a
  DECLARATION of S (class/interface/enum/trait/object/record, def/val/var, Java method/field/enum
  constant; inside T's own file when the error names an owner T):
      not in fork, declared upstream @ PR base    -> UPSTREAM_ONLY_API (fork lacks a prerequisite change)
      not in fork, declared upstream @ PR head    -> PR_SIBLING  (added by the same PR; PR-level splicing could supply it)
      not in fork, only imported upstream         -> UPSTREAM_ONLY_API (third-party dependency the fork lacks)
      not in fork, nowhere                        -> MODEL_INVENTED
      in fork, package does not exist             -> MISSING_BUILD_DEP (fork uses it elsewhere; this module lacks the dep)
      in fork, upstream file imports S, candidate does not -> MISSING_IMPORT
      in fork, but candidate imports it from a package the fork lacks (upstream relocation) -> UPSTREAM_ONLY_API
      in fork, but added by this PR to one of its files    -> PR_SIBLING (e.g. a test-harness method)
      in fork otherwise                           -> FORK_HAS (wrong scope/receiver: a model error)
  syntax error                                    -> MODEL_SYNTAX
  other: the offending line (annotations skipped to the declaration) was introduced by the candidate
      and occurs verbatim in the upstream post-PR file -> UPSTREAM_COPY (faithful port of code the fork
      cannot support: signature/type drift), else -> MODEL_OTHER.
Row cause (CASCADE ignored): PREREQ (all errors UPSTREAM_ONLY_API/PR_SIBLING/UPSTREAM_COPY/DEPENDENT_IN_PR/
MISSING_BUILD_DEP) | IMPORT (all MISSING_IMPORT) | PREREQ+IMPORT | MODEL (all FORK_HAS/MODEL_*) | MIXED.
Compilers stop after an error phase, so a PREREQ row may still hide model errors: PREREQ/IMPORT
rows are an UPPER bound on what supplying prerequisites would recover.
Usage: classify_buildfail.py        (writes buildfail-classified.jsonl + prints a summary)
"""
import collections, functools, json, re, subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

V = Path("/home/adam/Documents/work/rcap-class-project/run477/validate")
FORK = V / "fork-linkedin-kafka"; FORK_REV = "fdb9fd013a"
UP = Path("/home/adam/Documents/work/SALP/data/repos/apache__kafka.git")
MINT = V.parent / "mint_477_log.jsonl"
OUT = V / "buildfail-classified.jsonl"

mint = {m["pr"]: m for m in map(json.loads, open(MINT)) if m["pair"] == "apache/kafka -> linkedin/kafka"}

def git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, errors="replace").stdout

def _grep(repo, rev, pats, paths=("*.java", "*.scala")):
    args = ["git", "-C", str(repo), "grep", "-q", "-E"]
    for p in pats: args += ["-e", p]
    return subprocess.run(args + [rev, "--", *paths]).returncode == 0

@functools.lru_cache(maxsize=None)
def declared(repo, rev, name, owner=None, paths=None):
    """True if `name` is declared in *.java/*.scala at repo@rev (only in <owner>.java/.scala, or in `paths`, if given)."""
    if "." in name:  # a package: does any file declare it?
        return _grep(repo, rev, [rf"^\s*package\s+{re.escape(name)}\s*;?\s*$"])
    n = re.escape(name)
    pats = [rf"\b(class|interface|enum|trait|object|record|@interface)\s+{n}\b",
            rf"\b(def|val|var|type)\s+{n}\b",
            # Java method / field declaration: modifiers or a type, then the name, then ( = ; or ,
            rf"^\s*(public|protected|private|static|final|abstract|default|synchronized|native|transient|volatile)\b.*\b{n}\s*(\(|=|;|,)",
            rf"^\s*[A-Za-z_][][A-Za-z0-9_<>,.? ]*\s+{n}\s*\([^;]*\)\s*(throws [A-Za-z0-9_., ]+)?\s*\{{?\s*$",
            rf"^\s*{n}\s*(\(.*\))?\s*[,;{{]\s*$"]  # enum constant
    paths = paths or ((f"*/{owner}.java", f"*/{owner}.scala") if owner else ("*.java", "*.scala"))
    return _grep(repo, rev, pats, paths)

@functools.lru_cache(maxsize=None)
def upstream_file(rev, path):
    return git(UP, "show", f"{rev}:{path}")

@functools.lru_cache(maxsize=None)
def changed_files(base, head):
    return tuple(f for f in git(UP, "diff", "--name-only", base, head).split() if f.endswith((".java", ".scala")))

def norm(s): return re.sub(r"\s+", "", s)

@functools.lru_cache(maxsize=None)
def fork_file(path):
    # read from the object store, never the worktree (rebuild_logs.py may be splicing it)
    return git(FORK, "show", f"{FORK_REV}:{path}")

def spliced_text(cand):
    text = fork_file(cand["target_file"])
    for u in cand["units"]:
        if text.count(u["target_original"] or "\0") == 1:
            text = text.replace(u["target_original"], u["candidate_expanded"], 1)
    return text

def imports(text):
    """Simple names brought in by import statements (Java incl. static; Scala incl. selectors)."""
    names = set()
    for m in re.finditer(r"^\s*import\s+(?:static\s+)?([\w.]+?)(?:\.\{([^}]*)\})?\s*;?\s*$", text, re.M):
        if m.group(2):
            for sel in m.group(2).split(","):
                sel = sel.strip()
                names.add(sel.split("=>")[-1].strip() if "=>" in sel else sel)
        else:
            names.add(m.group(1).rsplit(".", 1)[-1])
        names.add(m.group(1))                      # the full path, for package checks
    return names

@functools.lru_cache(maxsize=None)
def importable(repo, rev, name):
    """The fork/upstream imports `name` somewhere (JDK / third-party / generated classes)."""
    if "." in name: return _grep(repo, rev, [rf"^\s*import\s+(static\s+)?{re.escape(name)}\."])
    n = re.escape(name)
    return _grep(repo, rev, [rf"^\s*import\s+(static\s+)?[A-Za-z0-9_.]+\.{n}\s*;?\s*$",
                             rf"^\s*import\s+[A-Za-z0-9_.]+\.\{{[^}}]*\b{n}\b[^}}]*\}}"])

# javac: "path:N: error: MSG"; scalac: "[Error] path:N: MSG"; javac run by scalac (core joint compile): "path:N:C:  error: MSG"
ERR = re.compile(r"^(?:\[Error\] )?(/\S+\.(?:java|scala)):(\d+)(?::(\d+))?:\s+(?:error:\s+)?(.*)$")
SYNTAX = re.compile(r"class, interface, or enum expected|expected start of|illegal start of|';' expected|"
                    r"reached end of file|not a statement|unclosed|<identifier> expected|'\)' expected|"
                    r"expected but .* found|orphaned")

def parse_errors(log):
    lines = log.splitlines(); errs = []
    for i, l in enumerate(lines):
        m = ERR.match(l)
        if not m or " warning:" in l: continue
        msg = m.group(4); sym = kind = owner = None
        for j in range(i + 1, min(i + 8, len(lines))):
            if ERR.match(lines[j]): break
            s = re.match(r"\s*symbol:\s+(\w+)\s+([\w$]+)", lines[j])
            if s: kind, sym = s.group(1), s.group(2)
            o = re.match(r"\s*location:\s+variable \w+ of type ([\w.$]+)", lines[j])
            if o and kind in ("method", "variable"): owner = o.group(1)
        if "cannot find symbol" not in msg:
            pk = re.search(r"object ([\w$]+) is not a member of package ([\w.]+)", msg)
            if pk and pk.group(1)[:1].islower(): sym = f"{pk.group(2)}.{pk.group(1)}"   # a missing sub-package
            for p in () if sym else (r"not found: (?:value|type|object) ([\w$]+)", r"^(?:value|method|object|type|class) ([\w$]+) is not a member of",
                      r"^package ([\w.]+) does not exist", r"^([\w$]+) is not a member of"):
                s = re.search(p, msg)
                if s: sym = s.group(1); break
            o = re.search(r"is not a member of (?:object |package )?([\w.$]+)", msg)
            if o and "package" not in msg.split("is not a member of")[1][:9]: owner = o.group(1)
        if owner: owner = re.sub(r"<.*", "", owner).rsplit(".", 1)[-1].split("$")[0]
        missing = bool(re.search(r"cannot find symbol|not found:|is not a member of|does not exist", msg))
        errs.append({"path": m.group(1), "line": int(m.group(2)), "col": int(m.group(3)) if m.group(3) else None,
                     "msg": msg[:200], "symbol": sym,
                     "owner": owner, "missing": missing})
    return errs

def classify_error(e, row, pr, cand, spliced):
    rel = re.split(r"fork-linkedin-kafka(?:-\d+)?/", e["path"], maxsplit=1)[-1]   # either rebuild worktree
    base, head = mint[pr]["base"], mint[pr]["head"]
    if rel != row["file"]:
        # a caller elsewhere broke: the PR also edits it (PR-level splicing would co-update it) or it is
        # fork code upstream never touched (adapting the change to it was RCAP's job)
        return "DEPENDENT_IN_PR" if rel in changed_files(base, head) else "DEPENDENT_OUTSIDE_PR"
    up_text = upstream_file(head, row["source_file"])
    src_lines = spliced.splitlines()
    raw = src_lines[e["line"] - 1] if 0 < e["line"] <= len(src_lines) else ""
    e["src"] = raw.strip()[:300]
    if e["missing"] and not e["symbol"] and e["col"]:   # no "symbol:" line: take the identifier at the column
        c = e["col"] - 1
        while 0 < c <= len(raw) and re.match(r"[\w$]", raw[c - 1]): c -= 1   # back to the start of the word
        mm = re.match(r"[^\w$]*([\w$]+)", raw[c:])
        e["symbol"] = mm.group(1) if mm else None
    if e["owner"] in ("Any", "Nothing", "Null"):
        return "CASCADE"                 # typed Any because an earlier symbol did not resolve
    if e["missing"] and e["symbol"]:
        s, own = e["symbol"], e["owner"]
        up_imp = s in imports(up_text)
        up_path = next((i for i in imports(up_text) if i.endswith("." + s)), "")
        in_fork = declared(FORK, FORK_REV, s, own) or (not own and importable(FORK, FORK_REV, s)) \
            or up_path.startswith(("java.", "javax.", "jdk.", "scala."))   # JDK / Scala stdlib: always on the classpath
        if not in_fork:
            if declared(UP, base, s, own): return "UPSTREAM_ONLY_API"
            if declared(UP, head, s, own): return "PR_SIBLING"
            if up_imp or (not own and importable(UP, base, s)): return "UPSTREAM_ONLY_API"  # external dep/API the fork lacks
            return "MODEL_INVENTED"
        if "." in s and "does not exist" in e["msg"]:
            return "MISSING_BUILD_DEP"   # the fork uses the package elsewhere; this module lacks the dependency
        if not own and up_imp and s not in imports(spliced):
            return "MISSING_IMPORT"
        path = next((i for i in imports(spliced) if i.endswith("." + s)), None)
        if not own and path and not declared(FORK, FORK_REV, path.rsplit(".", 1)[0]) \
                and not declared(FORK, FORK_REV, path.rsplit(".", 1)[0].rsplit(".", 1)[-1]):
            return "UPSTREAM_ONLY_API"   # imported from where upstream relocated it; that package is absent in the fork
        pr_files = changed_files(base, head)
        if pr_files and declared(UP, head, s, own, pr_files) and not declared(UP, base, s, own, pr_files):
            return "PR_SIBLING"          # added by this PR to another of its files (e.g. a test harness)
        return "FORK_HAS"
    if SYNTAX.search(e["msg"]): return "MODEL_SYNTAX"
    i = e["line"]
    while raw.strip().startswith("@") and i < len(src_lines):   # judge an annotation by the declaration it annotates
        raw = src_lines[i]; i += 1
    src = norm(raw)
    introduced = src and src not in norm(fork_file(row["file"]))
    if introduced and len(src) > 8 and src in norm(up_text): return "UPSTREAM_COPY"
    return "MODEL_OTHER"

PREREQ = {"UPSTREAM_ONLY_API", "PR_SIBLING", "UPSTREAM_COPY", "DEPENDENT_IN_PR", "MISSING_BUILD_DEP"}
IMPORT = {"MISSING_IMPORT"}
MODEL = {"FORK_HAS", "MODEL_INVENTED", "MODEL_SYNTAX", "MODEL_OTHER", "DEPENDENT_OUTSIDE_PR"}

def row_cause(kinds):
    kinds = kinds - {"CASCADE"}
    if not kinds: return "NO_ERRORS_PARSED"
    if kinds <= PREREQ: return "PREREQ"
    if kinds <= IMPORT: return "IMPORT"
    if kinds <= PREREQ | IMPORT: return "PREREQ+IMPORT"
    if kinds <= MODEL: return "MODEL"
    return "MIXED"

def main():
    cands = {}
    for p in (V / "candidates").glob("*.json"):
        r = json.loads(p.read_text()); cands[r["key"]] = r
    rows = [v for v in map(json.loads, open(V / "verdicts-linkedin__kafka.jsonl")) if v["verdict"] == "BUILD_FAIL"]
    def one(v):
        pr = re.search(r"/PR-(\d+)/", v["key"]).group(1)
        log = V / "build-logs" / (v["key"].replace("/", "__") + ".log")
        cand = cands[v["key"]]
        rec = {"key": v["key"], "file": v["file"], "source_file": cand["source_file"],
               "import_region_in_unit": any(re.search(r"^\s*import\s", u["target_original"] or "", re.M) for u in cand["units"])}
        if not log.exists():
            rec["cause"] = "NO_LOG"; return rec
        errs = parse_errors(log.read_text(errors="replace")); spliced = spliced_text(cand)
        for e in errs: e["cause"] = classify_error(e, rec, pr, cand, spliced)
        rec["cause"] = row_cause({e["cause"] for e in errs}); rec["errors"] = errs
        return rec
    with ThreadPoolExecutor(6) as ex: out = list(ex.map(one, rows))
    row_c = collections.Counter(o["cause"] for o in out)
    err_c = collections.Counter(e["cause"] for o in out for e in o.get("errors", []))
    OUT.write_text("".join(json.dumps(o) + "\n" for o in out))
    print(f"BUILD_FAIL rows: {len(rows)} -> {dict(row_c)}")
    print(f"errors: {dict(err_c)}")

if __name__ == "__main__":
    main()
