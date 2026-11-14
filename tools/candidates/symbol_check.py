"""No-build symbol check of LLM candidate function ports (study 2026-11-candidates-32k).

For every candidate (the adapted fork function a configuration returned) it lists the identifiers the candidate
REFERENCES that the fork's original function (target.java) does not, and resolves each one by name:

  fork  declared in the fork at its target revision (the target file's classes and their fork supertypes, its
        static imports; any fork type / method / field; a library name the fork itself uses), or in the JDK
  pr    not in the fork, but declared by the same upstream PR (its changed files base..head in the source clone,
        and every SAP unit's source.before -> source.after of that PR)
  none  neither: invented, or needs upstream code older than the PR

Verdict per candidate: UNRESOLVED (some name resolves nowhere) > ARITY_MISMATCH (a call whose argument count
matches no declaration of a fork-only method / constructor) > NEEDS_PR_EDITS (all names resolve, some only via
the PR's own edits) > SYMBOLS_OK. Candidates without code pass through as NO_CANDIDATE / ECHO /
PLACEHOLDER_LEAK. Name-level only: no types, scopes, overloads or generics (see symbol-check/README.md).

    symbol_check.py --config {direct,ppathf,rcap-full,rcap-noprog,abl,attempts} --results PATH
                    --sap-root OUT_A --repo-cache DIR [--gen-cache DIR] [--inputs PPATHF_INPUTS.jsonl]
                    [--units shared-units.tsv] [--index-dir DIR] --out OUT.jsonl

--config attempts reads an order_pr.py run directory (ordered/: pr-results.jsonl + attempts/), i.e. the
candidates a local build actually compiled; used for calibration.
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parents[1]
import symindex as si  # noqa: E402

DEFAULT_PR_BASES = Path("/home/adam/Documents/work/rcap-class-project/1-sap-inputs/sap-2026-11/pr-bases-477.json")
FORKS_FILE = HERE / "forks.json"

PKG_ROOTS = {"java", "javax", "org", "com", "net", "io", "scala", "kafka", "sun", "jdk", "lombok", "junit", "edu",
             "de", "cz", "uk", "nl", "fr", "ch", "it", "au", "ca", "us", "info", "dev", "co"}
IMPLICIT = {"length", "values", "valueOf", "ordinal", "name", "compareTo", "getClass", "class", "this", "super",
            "clone", "hashCode", "equals", "toString", "finalize", "wait", "notify", "notifyAll"}
TYPE_PARAM = re.compile(r"^[A-Z][A-Z0-9]?$")
CONST = re.compile(r"^[A-Z][A-Z0-9_]*$")

# A bare name (no qualifier) can only come from the enclosing classes, their supertypes or a static import, so
# when the target file is indexed a declaration elsewhere in the fork does not count (False: any fork declaration).
STRICT_BARE = os.environ.get("SYMCHECK_STRICT_BARE", "1") == "1"

V_OK, V_PR, V_ARITY, V_UNRES = "SYMBOLS_OK", "NEEDS_PR_EDITS", "ARITY_MISMATCH", "UNRESOLVED"
V_NONE, V_ECHO, V_LEAK = "NO_CANDIDATE", "ECHO", "PLACEHOLDER_LEAK"


# ---------------------------------------------------------------------------------------------- references

def candidate_refs(cand: str, target: str):
    """(name, role, arity) of every identifier the candidate references that target.java does not contain.
    Roles: type, ctor, call (bare), mcall (after '.'/'::'), member (after '.', not a call), mtype
    (capitalized after '.'), const (bare ALL_CAPS), bare (bare lowercase)."""
    toks = si.lex(cand)
    ttoks = si.lex(target)
    target_ids = {t.text for t in ttoks if t.kind == "id"}
    t_arities = call_arities(ttoks)
    wrapped = si.java_decls(si.lex("class __Cand__ {\n" + cand + "\n}"))
    local = wrapped["locals"] | wrapped["fields"] | set(wrapped["methods"]) | wrapped["types"]
    refs = []
    seen = set()
    n = len(toks)
    skip_until = -1
    for i, t in enumerate(toks):
        if i <= skip_until or t.kind != "id":
            continue
        name = t.text
        if name in si.JAVA_KEYWORDS:
            continue
        prev = toks[i - 1].text if i > 0 else ""
        nxt = toks[i + 1].text if i + 1 < n else ""
        # a package-qualified name: skip the lowercase segments up to the first capitalized one
        if prev not in (".", "::") and nxt == "." and name in PKG_ROOTS and name not in local:
            j = i
            while j + 2 < n and toks[j + 1].text == "." and toks[j + 2].kind == "id" and not toks[j + 2].text[:1].isupper():
                j += 2
            skip_until = j
            continue
        if name in IMPLICIT or TYPE_PARAM.match(name):
            continue
        known = name in target_ids
        if known and not (nxt == "(" or prev == "new") or (known and name in local):
            continue
        this_q = prev in (".", "::") and i >= 2 and toks[i - 2].text in ("this", "super")
        if this_q:
            # this.x / super.x() / this::m: a member of the enclosing classes, checked like a bare name
            role = "call" if (prev == "::" or nxt == "(") else "bare"
        elif prev in (".", "::"):
            if prev == "::" or nxt == "(":
                role = "mcall"
            elif name[:1].isupper():
                role = "mtype"
            else:
                role = "member"
        elif name in local:
            continue
        elif prev == "@":
            role = "type"
        elif prev == "new":
            role = "ctor"
        elif nxt == "(":
            role = "call"
        elif name[:1].isupper():
            role = "const" if CONST.match(name) and len(name) > 1 and not _used_as_type(toks, i) else "type"
        else:
            role = "bare"
        arity = None
        if role in ("call", "mcall", "ctor"):
            j = i + 1
            if j < n and toks[j].text == "<":
                g = si.generic_close(toks, j)
                j = g + 1 if g is not None else j
            if j < n and toks[j].text == "(":
                arity = si.count_args(toks, j)[0]
        arity_only = False
        if known:
            # a name the fork's function already uses: only a call with an argument count the fork's function
            # never uses is checked, for its arity alone
            if role not in ("call", "mcall", "ctor") or arity is None or arity in t_arities.get(name, ()):
                continue
            arity_only = True
        key = (name, role, arity, arity_only)
        if key not in seen:
            seen.add(key)
            refs.append(key)
    return refs


def call_arities(toks):
    out = collections.defaultdict(set)
    for i, t in enumerate(toks):
        if t.kind != "id":
            continue
        j = i + 1
        if j < len(toks) and toks[j].text == "<" and i > 0 and toks[i - 1].text in (".", "new"):
            g = si.generic_close(toks, j)
            j = g + 1 if g is not None else j
        if j < len(toks) and toks[j].text == "(":
            out[t.text].add(si.count_args(toks, j)[0])
    return out


def _used_as_type(toks, i):
    nxt = toks[i + 1].text if i + 1 < len(toks) else ""
    return nxt in ("<",) or (i + 1 < len(toks) and toks[i + 1].kind == "id")


# ---------------------------------------------------------------------------------------------- contexts

class ForkContext:
    """Name sets of one fork at one revision, plus per-file member views (cached)."""

    def __init__(self, index: dict, jdk: si.NameIndex | None = None):
        self.files = index["files"]
        self.ix = si.NameIndex(self.files)
        if jdk is not None:
            self.ix.known_external |= jdk.types
        self.pkg_roots = {(r.get("pkg") or "").split(".")[0] for r in self.files.values()} - {""}
        self._file_cache = {}

    def file_view(self, path):
        """Members visible by simple name inside the target file: fields / methods of its classes and their fork
        supertypes, its static imports (and members of statically star-imported fork types).
        lib: some supertype or star-imported type is not in the fork (a library class), so bare names may come
        from it."""
        if path in self._file_cache:
            return self._file_cache[path]
        r = self.files.get(path)
        fields, methods, lib = set(), set(), False
        if r is None:
            view = None
        else:
            jdk_super = False
            for tname in r["types"]:
                f, m, l = self.ix.members_of_type(tname)
                fields |= f
                methods |= m
                lib = lib or l
                jdk_super = jdk_super or self.ix.has_external_super(tname)
            fields |= set(r["fields"])
            methods |= set(r["methods"])
            si_names = set(r.get("static_imports", ()))
            fields |= si_names
            methods |= si_names
            for st in r.get("static_star", ()):
                f, m, l = self.ix.members_of_type(st)
                fields |= f
                methods |= m
                lib = lib or l
            view = {"fields": fields, "methods": methods, "lib": lib, "jdk_super": jdk_super}
        self._file_cache[path] = view
        return view


def arity_ok(k, arities):
    for a in arities:
        if a is None:
            return True
        if a >= 0 and a == k:
            return True
        if a < 0 and k >= -a - 1:
            return True
    return False


def resolve(name, role, arity, fork: ForkContext, jdk: si.NameIndex, pr: dict, fview, arity_only=False):
    """(where, detail): where in fork|jdk|pr|none; detail names the evidence (or the arity problem)."""
    F, J = fork.ix, jdk
    pr_types, pr_fields, pr_methods, pr_ctors = (set(pr.get("types", ())), set(pr.get("fields", ())),
                                                 pr.get("methods", {}), pr.get("ctors", {}))
    if arity_only:
        return _arity_only(name, role, arity, F, J, pr_methods, pr_ctors)
    if role in ("type", "ctor", "mtype"):
        if name in F.types:
            where = ("fork", "type")
        elif name in J.types:
            where = ("jdk", "type")
        elif role == "mtype" and (name in F.fields or name in J.fields):
            where = ("fork", "constant")
        elif name in F.upper and name not in pr_types:
            where = ("fork", "used")          # a library type the fork uses
        elif name in pr_types or name in pr_ctors or (role == "mtype" and name in pr_fields):
            where = ("pr", "type")
        else:
            where = ("none", "")
        if role == "ctor" and arity is not None and where == ("fork", "type") and name not in J.types:
            ars = set(F.ctors.get(name, ())) | set(pr_ctors.get(name, ()))
            if not ars:
                ars = {0}
            if not arity_ok(arity, ars):
                return "fork", f"arity:{arity} not in {sorted(ars, key=str)}"
        return where
    if role == "call":
        if fview is not None and name in fview["methods"]:
            w = ("fork", "file")
        elif fview is not None and fview["lib"]:
            w = ("fork", "library-super")
        elif name in J.methods and (fview is None or fview["jdk_super"]):
            w = ("jdk", "method")       # inherited from a JDK supertype (Thread.interrupt(), ...)
        elif name in F.static_imports:
            w = ("fork", "static-import")    # importable: some fork file imports it statically
        elif (fview is None or not STRICT_BARE) and name in F.methods:
            w = ("fork", "global")
        elif name in pr_methods:
            w = ("pr", "method")
        elif fview is None and name in F.called:
            w = ("fork", "used")
        else:
            w = ("none", "")
        return _arity(name, arity, w, F, J, pr_methods)
    if role == "mcall":
        if name in F.methods:
            w = ("fork", "method")
        elif name in J.methods:
            w = ("jdk", "method")
        elif name in F.called:
            w = ("fork", "used")
        elif name in pr_methods:
            w = ("pr", "method")
        else:
            w = ("none", "")
        return _arity(name, arity, w, F, J, pr_methods)
    if role == "member":
        if name in F.fields:
            return "fork", "field"
        if name in J.fields:
            return "jdk", "field"
        if name in F.member:
            return "fork", "used"
        if name in F.types:
            return "fork", "type"
        if name in pr_fields:
            return "pr", "field"
        return "none", ""
    # bare / const
    if fview is not None and name in fview["fields"]:
        return "fork", "file"
    if fview is not None and fview["lib"]:
        return "fork", "library-super"
    if name in F.static_imports:
        return "fork", "static-import"
    if (fview is None or not STRICT_BARE) and name in F.fields:
        return "fork", "global"
    if role == "const" and (name in J.fields or name in F.upper):
        return "fork", "used"
    if name in pr_fields:
        return "pr", "field"
    return "none", ""


def _arity_only(name, role, arity, F, J, pr_methods, pr_ctors):
    """A known name called with a new argument count: fork if a fork declaration takes it, pr if only the PR's
    (a signature the PR changes), else an arity mismatch. Names the JDK declares, or that the fork only calls
    (library), are not checked."""
    if role == "ctor":
        if name in J.types or name not in F.types:
            return "fork", "unchecked"
        fork_ars = set(F.ctors.get(name, ())) or {0}
        pr_ars = set(pr_ctors.get(name, ()))
    else:
        if name in J.methods or name not in F.methods or name in F.static_imports:
            return "fork", "unchecked"     # JDK / statically imported library overloads are unknown
        fork_ars = set(F.methods.get(name, ()))
        pr_ars = set(pr_methods.get(name, ()))
    if arity_ok(arity, fork_ars):
        return "fork", "arity"
    if arity_ok(arity, pr_ars):
        return "pr", "arity"
    return "fork", f"arity:{arity} not in {sorted(fork_ars | pr_ars, key=str)}"


def _arity(name, arity, w, F, J, pr_methods):
    """Flag a call whose argument count matches no fork/PR declaration of a name that only the fork declares
    (JDK names and library names the fork merely calls are never arity-checked: overloads unknown)."""
    if arity is None or w[0] not in ("fork", "pr") or name in J.methods or name in F.static_imports \
            or w[1] in ("used", "library-super", "static-import"):
        return w
    ars = set(F.methods.get(name, ())) | set(pr_methods.get(name, ()))
    if not ars or name not in F.methods and name not in pr_methods:
        return w
    if not arity_ok(arity, ars):
        return w[0], f"arity:{arity} not in {sorted(ars, key=str)}"
    return w


def check(cand, target, fork: ForkContext, jdk, pr, target_file):
    fview = fork.file_view(target_file) if target_file else None
    refs = candidate_refs(cand, target)
    names = {"fork": [], "jdk": [], "pr": [], "none": [], "arity": []}
    for name, role, arity, arity_only in refs:
        where, detail = resolve(name, role, arity, fork, jdk, pr, fview, arity_only)
        if detail.startswith("arity:"):
            names["arity"].append(f"{name}({arity})")
        names[where].append(f"{name}:{role}" + (f":{detail}" if detail and not detail.startswith("arity") else ""))
    if names["none"]:
        verdict = V_UNRES
    elif names["arity"]:
        verdict = V_ARITY
    elif names["pr"]:
        verdict = V_PR
    else:
        verdict = V_OK
    return {"verdict": verdict, "new_refs": len(refs), "unresolved": names["none"], "pr_only": names["pr"],
            "arity_mismatch": names["arity"], "fork_resolved": len(names["fork"]), "jdk_resolved": len(names["jdk"]),
            "target_file_indexed": fview is not None}


# ---------------------------------------------------------------------------------------------- inputs

def load_forks(path=FORKS_FILE):
    return json.loads(Path(path).read_text())


class Contexts:
    """Lazily built fork / JDK / PR indexes, cached as gzipped JSON under index_dir."""

    def __init__(self, repo_cache, index_dir, pr_bases=None, jmods=None, sap_root=None):
        self.repo_cache = Path(repo_cache) if repo_cache else None
        self.index_dir = Path(index_dir)
        self.forks = load_forks()
        self.pr_bases = json.loads(Path(pr_bases).read_text()) if pr_bases and Path(pr_bases).is_file() else {}
        self.jmods = jmods or si.default_jmods()
        self.sap_root = Path(sap_root) if sap_root else None
        self._fork, self._pr, self._jdk = {}, {}, None
        self.notes = collections.Counter()

    def jdk(self):
        if self._jdk is None:
            d = si.load_or_build(self.index_dir / "jdk.json.gz", lambda: si.build_jdk_index(self.jmods))
            self._jdk = si.NameIndex(d["files"])
        return self._jdk

    def git_dir(self, repo):
        if self.repo_cache is None:
            return None
        g = self.repo_cache / (repo.replace("/", "__") + ".git")
        return g if g.is_dir() else None

    def fork(self, target_repo):
        if target_repo not in self._fork:
            f = self.forks.get(target_repo)
            if f is None:
                raise SystemExit(f"no revision for fork {target_repo} in {FORKS_FILE}")
            cache = self.index_dir / f"fork-{target_repo.replace('/', '__')}-{f['rev'][:12]}.json.gz"
            g = self.git_dir(target_repo)
            if not cache.exists() and g is None:
                raise SystemExit(f"fork {target_repo}: no index {cache} and no clone under {self.repo_cache}")
            d = si.load_or_build(cache, lambda: si.build_repo_index(g, f["rev"]))
            self._fork[target_repo] = ForkContext(d, self.jdk())
        return self._fork[target_repo]

    def pr(self, pr_dir: Path, manifest):
        """Names the PR adds: SAP units (source.before -> source.after) and, when the source clone and the PR's
        base/head are known, its changed .java/.scala files base -> head."""
        key = str(pr_dir)
        if key in self._pr:
            return self._pr[key]
        src, tgt = manifest["source_repo"], manifest["target_repo"]
        num = manifest["pr_id"].split("-")[1]
        cache = self.index_dir / "pr" / f"{pr_dir.parent.name}__{pr_dir.name}.json.gz"

        def build():
            saps = collections.defaultdict(lambda: ([], []))
            for f in sorted(pr_dir.glob("sap-*/functions/*/source.*.*")):
                side = 0 if f.name.startswith("source.before.") else 1 if f.name.startswith("source.after.") else None
                if side is not None:
                    saps[f.parts[-4]][side].append((f.name, f.read_text(errors="replace")))
            how = "sap"
            rec = self.pr_bases.get(f"{tgt.replace('/', '-')}__pr{num}")
            g = self.git_dir(src)
            if rec and g is not None:
                try:
                    import subprocess
                    out = subprocess.run(["git", "--git-dir", str(g), "diff", "--name-status", "--no-renames",
                                          rec["merge_base"], rec["head_sha"]], capture_output=True, text=True,
                                         check=True).stdout
                    paths = [ln.split("\t")[-1] for ln in out.splitlines()
                             if ln.split("\t")[-1].endswith((".java", ".scala"))]
                    bb, aa = si.read_blobs(g, rec["merge_base"], paths), si.read_blobs(g, rec["head_sha"], paths)
                    diff = si.added_names_per_file({p: ([(p, bb[p])] if p in bb else [], [(p, aa[p])] if p in aa else [])
                                                    for p in paths})
                    how = "sap+diff"
                except Exception as exc:  # recorded, falls back to the SAPs only
                    how = f"sap (diff failed: {type(exc).__name__})"
                    diff = None
            else:
                diff = None
            # the two sources are diffed separately: a SAP's before side can come from another base
            d = si.merge_added(si.added_names_per_file(saps), diff)
            d.update(version=si.INDEX_VERSION, how=how)
            return d
        d = si.load_or_build(cache, build)
        self.notes[f"pr_index:{d.get('how')}"] += 0
        self._pr[key] = d
        return d


def strip_fence(text):
    m = re.search(r"```[a-zA-Z]*[ \t]*\n(.*?)```", text or "", re.DOTALL)
    return m.group(1) if m else text


def read_units_tsv(path):
    keys = set()
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("pair\t"):
            continue
        pair, pr, sap, unit = line.split("\t")[:4]
        keys.add(f"{pair}/{pr}/{sap}::{unit.replace('functions/', '')}")
    return keys


PH_RE = re.compile(r"placeholder_\d+")


def load_generate_results(path, inputs=None):
    """Direct / PPatHF-style rows of ppathf_style/generate.py: unit id -> (candidate or None, outcome).
    `candidate` (recovered) is used when present; an answer with no placeholders needs no recovery
    (candidate_reduced); otherwise --inputs (the generate.py inputs) lets this recover it here."""
    recs = [json.loads(l) for l in open(path) if l.strip()]
    need = [r for r in recs if r.get("candidate") is None and r.get("outcome") in ("answered", "no_output")
            and (r.get("placeholders") or 0) > 0]
    if need and inputs:
        sys.path.insert(0, str(REPO / "tools" / "ppathf_style"))
        import generate as gen  # noqa: E402
        removed = {gen.unit_key(r): r["removed_pieces_target"] for r in map(json.loads, open(inputs)) if r.get("eligible")}
        reducer = gen.make_reducer()
        for r in need:
            gen.recover(r, removed[r["unit_id"]], reducer)
    out = {}
    for r in recs:
        if r.get("candidate_reduced") is None and r.get("response") and r.get("outcome") == "answered":
            r["candidate_reduced"] = strip_fence(r["response"])   # generate.py's rule: the first fenced block
        uid = r["unit_id"]
        cand = r.get("candidate")
        if cand is None and r.get("outcome") == "answered" and not (r.get("placeholders") or 0):
            cand = r.get("candidate_reduced")
        if cand is None and r.get("outcome") == "answered" and (r.get("placeholders") or 0):
            out[uid] = (None, "needs_recovery (pass --inputs)")
            continue
        out[uid] = (cand, r.get("outcome"))
    return out


def load_attempts(ordered: Path):
    """An order_pr.py run dir: unit id (pair/PR/sap::fn) -> (candidate the build applied, status), plus the
    target original the run saved."""
    out = {}
    pair_of = {}
    for line in open(ordered / "pr-results.jsonl"):
        row = json.loads(line)
        pair_of[row["pr"]] = row.get("pair")
        for k, u in row.get("units", {}).items():
            pass
    for f in ordered.glob("attempts/PR-*/*.candidate.json"):
        c = json.loads(f.read_text())
        pr = f.parent.name
        uid = f"{pair_of.get(pr)}/{pr}/{c['unit']}"
        tgt = f.with_name(f.name.replace(".candidate.json", ".target_original.txt"))
        out[uid] = (c.get("candidate_expanded"), c.get("status"),
                    tgt.read_text() if tgt.exists() else None)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True,
                    choices=["direct", "ppathf", "rcap-full", "rcap-noprog", "abl", "attempts"])
    ap.add_argument("--results", required=True, help="generate.py results jsonl, run_477.py rows, or ordered/ dir")
    ap.add_argument("--sap-root", required=True)
    ap.add_argument("--repo-cache", help="bare clones <owner>__<name>.git (fork indexes are built from them once)")
    ap.add_argument("--index-dir", default=os.path.expanduser("~/.cache/rcap-symindex"), help="cache of the built indexes")
    ap.add_argument("--gen-cache", action="append", default=[], help="RCAP answer cache dir(s) (rcap-* configs)")
    ap.add_argument("--overrides", help="RCAP overrides file for the replay (default: tools/candidates/cand32k-overrides.json)")
    ap.add_argument("--inputs", help="generate.py inputs jsonl, to recover PPatHF answers not yet recovered")
    ap.add_argument("--units", help="TSV pair/pr/sap/unit: only these units (e.g. inputs/shared-units.tsv)")
    ap.add_argument("--pr-bases", default=str(DEFAULT_PR_BASES))
    ap.add_argument("--pair", help="pair for unit ids written without one (the old Kafka runs: PR-N/sap::fn)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)

    sap_root = Path(a.sap_root)
    ctx = Contexts(a.repo_cache, a.index_dir, a.pr_bases, sap_root=sap_root)
    targets = {}
    if a.config in ("direct", "ppathf"):
        cands = {k: (c, o) for k, (c, o) in load_generate_results(a.results, a.inputs).items()}
    elif a.config == "attempts":
        raw = load_attempts(Path(a.results))
        cands = {k: (c, o) for k, (c, o, _) in raw.items()}
        targets = {k: t for k, (_, _, t) in raw.items() if t is not None}
    else:
        import rcap_candidates as rc
        mode = "semantic_reduction_only" if a.config == "rcap-noprog" else "full_rcap"
        res = rc.replay(Path(a.results), sap_root, a.gen_cache, mode, a.overrides,
                        edit_evidence=False if a.config == "abl" else None)
        cands = {k: (v["candidate"], v["outcome"]) for k, v in res.items()}
        targets = {k: v["target"] for k, v in res.items() if v.get("target") is not None}
    keep = read_units_tsv(a.units) if a.units else None

    from rcap.reprompt import code_equal  # noqa: E402

    n = collections.Counter()
    with open(a.out, "w") as out:
        for uid in sorted(cands):
            cand, outcome = cands[uid]
            full = uid if uid.count("/") >= 2 else f"{a.pair}/{uid}"
            if keep is not None and full not in keep:
                continue
            pair, pr, rest = full.split("/", 2)
            sap, fn = rest.split("::")
            pr_dir = sap_root / pair / pr
            fdir = pr_dir / sap / "functions" / fn
            row = {"unit_id": full, "config": a.config, "outcome": outcome}
            target = targets.get(uid)
            if target is None:
                tf = fdir / "target.java"
                target = tf.read_text(errors="replace") if tf.exists() else None
            if not cand or not cand.strip():
                row["verdict"] = V_NONE
            elif PH_RE.search(cand) or "RCAP_PH_" in cand:
                row["verdict"] = V_LEAK
            elif target is not None and code_equal(cand, target):
                row["verdict"] = V_ECHO
            elif target is None:
                row["verdict"] = "NO_TARGET"
            else:
                manifest = json.loads((pr_dir / "pr.json").read_text())
                sapj = json.loads((pr_dir / sap / "sap.json").read_text())
                fork = ctx.fork(manifest["target_repo"])
                prx = ctx.pr(pr_dir, manifest)
                row.update(check(cand, target, fork, ctx.jdk(), prx, sapj.get("target_file")))
            row["real_change"] = row["verdict"] not in (V_NONE, V_ECHO, V_LEAK, "NO_TARGET")
            n[row["verdict"]] += 1
            out.write(json.dumps(row) + "\n")
    print(json.dumps(dict(n.most_common()), indent=1))


if __name__ == "__main__":
    sys.path.insert(0, str(REPO / "src"))
    main()
