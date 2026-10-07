#!/usr/bin/env python3
"""ONE re-prompt on the stored 477 attempt-1 candidate, standalone: errors only (arm B) or with
the context of the missing names (arm C). Arm A only re-validates attempt 1.

  ctx_reprompt.py --sample sample.json --arm A|B|C --fork DIR --out-dir DIR --baseline FILE
                  [--screen screen.jsonl] [--part i/n] [--only SUBSTR] [--limit N]

Per sampled row (a SAP of the 477 run that was BUILD_FAIL standalone):

* attempt 1 is NOT regenerated: the units, their contexts and original texts come from the same
  pipeline stages (stub backend, whole-file targets not narrowed, i.e. exactly what the 477 run
  asked), and the candidate is the stored one (run477/validate/candidates). The recovered
  target text must equal the stored one, else the row is refused.
* validation is validate.py's: units spliced one after the other by exact text, same Gradle
  tasks and flags, only NEW test failures count. Every validation is cached by the spliced
  content (val-cache/), so arm A's build of attempt 1 serves B and C and both see the very same
  diagnostics; the full build log and the spliced file are kept with it.
* arms B and C run rcap.reprompt.run_reprompt_loop with k=2: exactly one repair request per
  unit that has errors, temperature 0, no further attempt. B: the v2 repair request with the
  diagnostics only. C: the same plus `symbol_context` (rcap.adaptctx), which selects the v3
  request. Echo, decline, placeholder and shape guards are the loop's and generate's.
  C2 (exploratory, added after C was seen): C without the ready-made decline line per name.
* every final candidate gets a fidelity label (rcap.fidelity, rename_context="isolation").

Resumable by key per arm. Writes only under --out-dir, the baseline file and the fork clone's
working tree (restored after every validation).
"""
import argparse, collections, glob, hashlib, json, os, re, subprocess, sys, time, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gencache import CachedBackend  # noqa: E402
from rcap import cochange as cc  # noqa: E402
from rcap.adaptctx import Sources, build_symbol_context, import_symbols  # noqa: E402
from rcap.backend_ollama import OllamaBackend  # noqa: E402
from rcap.feedback import (extract_build_feedback, extract_test_feedback, missing_symbols,  # noqa: E402
                           parse_compile_errors, parse_junit_failures)
from rcap.fidelity import judge  # noqa: E402
from rcap.generate import GenerationRecord, StubBackend  # noqa: E402
from rcap.pipeline import _expected_placeholders, run_case  # noqa: E402
from rcap.provenance import run_provenance  # noqa: E402
from rcap.unitkind import shape_mismatch  # noqa: E402
from rcap.prorder import find_renames  # noqa: E402
from rcap.reduction_program import PLACEHOLDER, recover  # noqa: E402
from rcap.reprompt import UnitState, Validation, run_reprompt_loop  # noqa: E402
from rcap.request import unit_kinds  # noqa: E402

RUN = Path("/home/adam/Documents/work/rcap-class-project/run477")
ap = argparse.ArgumentParser()
ap.add_argument("--sample", required=True); ap.add_argument("--arm", required=True, choices=["A", "B", "C", "C2"])
ap.add_argument("--fork", required=True); ap.add_argument("--out-dir", required=True)
ap.add_argument("--baseline", required=True)
ap.add_argument("--salp-out", default=str(RUN / "salp-477-out"))
ap.add_argument("--candidates", default=str(RUN / "validate/candidates"))
ap.add_argument("--old-verdicts", default=str(RUN / "validate/verdicts-linkedin__kafka.v2.jsonl"))
ap.add_argument("--gradle-bin-file", default=str(RUN / "validate/gradle_bin.txt"))
ap.add_argument("--repo-cache", default="/home/adam/Documents/work/SALP/data/repos")
ap.add_argument("--source-repo", default="apache__kafka.git")
ap.add_argument("--screen", help="screen.jsonl: its fork equivalents become hints, its owners a fallback")
ap.add_argument("--part", help="i/n: only rows whose index % n == i"); ap.add_argument("--only")
ap.add_argument("--limit", type=int)
ap.add_argument("--dry-run", action="store_true", help="no model call: every repair request is answered CANNOT_ADAPT; prompts are written")
a = ap.parse_args()

FORK = Path(a.fork).resolve(); OUTD = Path(a.out_dir); TAG = a.arm + ("-dry" if a.dry_run else "")
OUT = OUTD / f"arm-{TAG}.jsonl"
ATT = OUTD / "attempts" / TAG; VC = OUTD / "val-cache"; VC.mkdir(parents=True, exist_ok=True)
GRADLE = Path(a.gradle_bin_file).read_text().strip(); SRC = Path(a.repo_cache) / a.source_repo
BASE = Path(a.baseline); baseline = json.loads(BASE.read_text()) if BASE.exists() else {}
TEST_TIMEOUT = int(os.environ.get("TEST_TIMEOUT", "1800")); BUILD_TIMEOUT = 1800
EXT = {"java": "java", "scala": "scala", "sc": "scala"}


# ------------------------------------------------------------------ build / test (validate.py)
def sh(cmd, timeout):
    t = time.time()
    try:
        r = subprocess.run(cmd, cwd=FORK, capture_output=True, text=True, timeout=timeout, errors="replace")
        return r.returncode, r.stdout + r.stderr, round(time.time() - t, 1)
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"") + (e.stderr or b"")
        return "timeout", out.decode(errors="replace") if isinstance(out, bytes) else out, round(time.time() - t, 1)

def gradle(*tasks, timeout):
    return sh([GRADLE, "--offline", "-q", "-Dorg.gradle.java.home=/usr/lib/jvm/java-11-openjdk-amd64",
               "-PmaxParallelForks=" + os.environ.get("MAX_FORKS", "3"), *tasks, "-x", "checkstyleMain", "-x", "checkstyleTest",
               "-x", "spotbugsMain", "-x", "spotbugsTest", "-x", "spotlessScala"], timeout)

def project_of(path):
    return ":" + path.split("/src/")[0].replace("/", ":") if "/src/" in path else None

def fqcn(path):
    try: src = (FORK / path).read_text(errors="replace")
    except OSError: src = ""
    pk = [m.group(1) for m in re.finditer(r"^\s*package\s+([\w.]+)\s*;?\s*$", src, re.M)]
    if pk: return ".".join(pk) + "." + Path(path).stem
    m = re.search(r"/src/\w+/(?:java|scala)/(.+)\.(?:java|scala)$", path)
    return m.group(1).replace("/", ".") if m else None

def concrete(rel):
    src = (FORK / rel).read_text(errors="replace"); name = Path(rel).stem
    if not re.search(rf"abstract\s+class\s+{name}\b", src): return [fqcn(rel)]
    root = rel.split("/src/")[0]
    subs = [p for p in glob.glob(f"{FORK}/{root}/src/test/**/*.*", recursive=True)
            if p.endswith((".java", ".scala")) and re.search(rf"extends\s+{name}\b", Path(p).read_text(errors="replace"))]
    return [fqcn(os.path.relpath(p, FORK)) for p in subs]

def find_test(path):
    proj = project_of(path)
    if "/src/test/" in path: return proj, concrete(path), "self"
    name = Path(path).stem + "Test"; root = path.split("/src/")[0]
    same = [p for p in glob.glob(f"{FORK}/{root}/src/test/**/{name}.*", recursive=True) if p.endswith((".java", ".scala"))]
    hits = same or [p for p in glob.glob(f"{FORK}/**/src/test/**/{name}.*", recursive=True) if p.endswith((".java", ".scala"))]
    if len(hits) != 1: return proj, [], f"no_unique_test({len(hits)})"
    rel = os.path.relpath(hits[0], FORK)
    return project_of(rel), concrete(rel), "same_project" if same else "other_project"

def run_test(proj, classes):
    rdir = FORK / proj.strip(":").replace(":", "/") / "build/test-results/test"
    for f in glob.glob(f"{rdir}/*.xml"): os.remove(f)
    args = [x for c in classes for x in ("--tests", c)]
    rc, log, secs = gradle(f"{proj}:test", *args, timeout=TEST_TIMEOUT)
    ran, failures = parse_junit_failures(sorted(glob.glob(f"{rdir}/*.xml")))
    return {"rc": rc, "ran": ran, "failed": sorted(f.name for f in failures), "seconds": secs}, failures, log

def line_mapper(raw, art):
    spots = []
    for e in (art.placeholders if art else []):
        ph = PLACEHOLDER.format(n=e.ph_id); pos = raw.find(ph)
        if pos < 0 or raw.count(ph) != 1: return lambda n: None
        spots.append((raw.count("\n", 0, pos) + 1, e.original_text.count("\n") - ph.count("\n")))
    spots.sort()
    def to_raw(n):
        shift = 0
        for raw_line, extra in spots:
            start = raw_line + shift
            if n < start: break
            if n <= start + extra: return raw_line
            shift += extra
        return n - shift
    return to_raw

def make_validator(key, path, units, raws):
    fpath = FORK / path

    def validate(attempt, expanded):
        detail = {}
        subprocess.run(["git", "checkout", "-q", "--", path], cwd=FORK)
        text = fpath.read_text(encoding="utf-8"); orig = text; applied = []
        for u, cand in zip(units, expanded):
            n = text.count(u.target_original or "\0")
            why = shape_mismatch(u.target_original, cand, "scala" if path.endswith((".scala", ".sc")) else "java")
            if why: applied.append("shape_mismatch")      # one function for a whole-file unit: never written
            elif n == 1: text = text.replace(u.target_original, cand, 1); applied.append("ok")
            else: applied.append(f"found_{n}")
        detail["splice"] = applied
        if not any(x == "ok" for x in applied) or applied[0] != "ok":
            return Validation(verdict="INTEGRATION_FAIL", summary=f"splice {applied}", detail=detail)
        if text == orig:
            return Validation(verdict="INCONCLUSIVE", summary="candidate identical to fork code (echo)", detail=detail)
        spans = []
        for cand, ok in zip(expanded, applied):
            pos = text.find(cand) if ok == "ok" else -1
            first = text.count("\n", 0, pos) + 1 if pos >= 0 else 0
            spans.append((first, first + cand.rstrip("\n").count("\n")) if pos >= 0 else (0, -1))
        detail["unit_spans"] = spans
        # ---- cache: the same spliced file gives the same verdict (arm A's build serves B and C)
        vkey = hashlib.sha256((path + "\0" + text).encode()).hexdigest()
        detail["val_key"] = vkey; cfile = VC / f"{vkey}.json"
        mappers = [line_mapper(raws()[i], u.art) for i, u in enumerate(units)]
        if cfile.exists():
            c = json.loads(cfile.read_text()); detail.update(c["detail"]); detail["val_cached"] = True
            if c["verdict"] == "BUILD_FAIL":       # the feedback depends on how THIS candidate maps lines
                log = (VC / f"{vkey}.build.log").read_text(errors="replace")
                fb = extract_build_feedback(log, path, spans, root=c["root"], unit_line=lambda i, n: mappers[i](n), edited_source=text)
                return Validation(verdict="BUILD_FAIL", kind="build", feedback=fb.text, unit_feedback=fb.per_unit,
                                  units_with_errors=fb.units_with_errors, summary=fb.summary, detail=detail)
            return Validation(verdict=c["verdict"], kind=c.get("kind"), feedback=c.get("feedback", ""),
                              summary=c.get("summary", ""), detail=detail)
        def done(v, **extra):
            cfile.write_text(json.dumps({"key": key, "path": path, "verdict": v.verdict, "kind": v.kind, "feedback": v.feedback,
                                         "summary": v.summary, "root": str(FORK), "arm": a.arm, "attempt": attempt,
                                         "detail": {k: x for k, x in detail.items() if k not in ("val_cached",)}, **extra}))
            return v
        try:
            fpath.write_text(text, encoding="utf-8")
            (VC / f"{vkey}.spliced.txt").write_text(text)
            proj = project_of(path)
            rc, log, secs = gradle(f"{proj}:testClasses", timeout=BUILD_TIMEOUT)
            detail["build"] = {"rc": rc, "seconds": secs, "log_chars": len(log)}
            if rc == "timeout":
                return Validation(verdict="INCONCLUSIVE", summary="build timeout", detail=detail)   # not cached
            if rc != 0:
                (VC / f"{vkey}.build.log").write_text(log)
                fb = extract_build_feedback(log, path, spans, root=str(FORK), unit_line=lambda i, n: mappers[i](n), edited_source=text)
                detail["feedback"] = {"errors_total": fb.n_total, "errors_shown": fb.n_shown, "chars": len(fb.text), "mappable": fb.mappable}
                return done(Validation(verdict="BUILD_FAIL", kind="build", feedback=fb.text, unit_feedback=fb.per_unit,
                                       units_with_errors=fb.units_with_errors, summary=fb.summary, detail=detail))
            tproj, tcls, how = find_test(path); detail["test_class"] = tcls; detail["test_how"] = how
            if not tcls:
                how = how if how.startswith("no_") else "abstract test with no subclass"
                return done(Validation(verdict="INCONCLUSIVE", summary=f"build ok; {how}", detail=detail))
            bkey = f"{tproj}|{','.join(tcls)}"
            if bkey not in baseline:
                fpath.write_text(orig, encoding="utf-8")
                baseline[bkey], _, _ = run_test(tproj, tcls); BASE.write_text(json.dumps(baseline, indent=1))
                fpath.write_text(text, encoding="utf-8")
            b = baseline[bkey]; t, failures, tlog = run_test(tproj, tcls)
            new = sorted(set(t["failed"]) - set(b["failed"]))
            detail["test"] = t; detail["baseline"] = {k: b[k] for k in ("rc", "ran", "failed")}; detail["new_failures"] = new
            if b["rc"] == "timeout" or b["ran"] == 0:
                return done(Validation(verdict="INCONCLUSIVE", summary="baseline test did not run cleanly", detail=detail))
            if t["rc"] == "timeout":
                return Validation(verdict="INCONCLUSIVE", summary="test timeout", detail=detail)    # not cached
            if t["ran"] == 0 or new:
                (VC / f"{vkey}.test.log").write_text(tlog)
                fb = extract_test_feedback(failures, b["failed"], ran=t["ran"], gradle_log=tlog)
                return done(Validation(verdict="TEST_FAIL", kind="test", feedback=fb.text, summary=fb.summary, detail=detail))
            return done(Validation(verdict="VALID", summary=f"{t['ran']} tests ran, no new failures", detail=detail))
        except Exception as exc:
            return Validation(verdict="INCONCLUSIVE", summary=f"harness error: {exc!r}"[:300], detail=detail)
        finally:
            subprocess.run(["git", "checkout", "-q", "--", path], cwd=FORK)

    def timed(attempt, expanded):
        t0 = time.time(); v = validate(attempt, expanded); v.seconds = round(time.time() - t0, 1)
        return v
    return timed


# ------------------------------------------------------------------ the three trees
def sgit(*args):
    r = subprocess.run(["git", "--git-dir", str(SRC), *args], capture_output=True, text=True, errors="replace")
    return r.stdout if r.returncode == 0 else None

def fgit(*args):
    r = subprocess.run(["git", *args], cwd=FORK, capture_output=True, text=True, errors="replace")
    return r.stdout if r.returncode == 0 else None

def pr_base(head):
    """mint_477.pr_base: the fork point of the PR's own line of development."""
    parents = (sgit("rev-list", "--parents", "-1", head) or "").split()[1:]
    if subprocess.run(["git", "--git-dir", str(SRC), "merge-base", "--is-ancestor", head, "HEAD"]).returncode != 0:
        return (sgit("merge-base", head, "HEAD") or "").strip() or None, "merge_base"
    if len(parents) == 1: return parents[0], "first_parent"
    return None, "ambiguous_merge_head_already_landed"

_STEMS = {}
def stems(tag, listing):
    if tag not in _STEMS:
        d = collections.defaultdict(list)
        for p in (listing() or "").splitlines():
            if p.endswith((".java", ".scala")): d[Path(p).stem].append(p)
        _STEMS[tag] = d
    return _STEMS[tag]

_WORD = {}
def fork_has_word(name):
    """Does the identifier occur anywhere in the fork's Java/Scala sources (HEAD)?"""
    if name not in _WORD:
        r = subprocess.run(["git", "grep", "-q", "-w", "-F", "-e", name, "HEAD", "--", "*.java", "*.scala"], cwd=FORK)
        _WORD[name] = r.returncode == 0
    return _WORD[name]

_UP = {}
def upstream(prn):
    """(head, base, how, parsed diff, renames, Sources) of one upstream PR."""
    if prn in _UP: return _UP[prn]
    head = (sgit("rev-parse", f"refs/salp/pr/{prn}^{{commit}}") or "").strip()
    base, how = pr_base(head) if head else (None, "no_pr_ref")
    files = cc.parse_unified_diff((sgit("diff", "-U0", "-M", "--no-color", base, head) or "") if base else "")
    show = lambda rev: (lambda p: sgit("show", f"{rev}:{p}") if rev and p else None)   # noqa: E731
    src = Sources(upstream_after=show(head), upstream_before=show(base),
                  fork=lambda p: fgit("show", f"HEAD:{p}") if p else None,
                  upstream_type_files=lambda n: list(stems(base, lambda: sgit("ls-tree", "-r", "--name-only", base)).get(n, [])) if base and n else [],
                  fork_type_files=lambda n: list(stems("fork", lambda: fgit("ls-files")).get(n, [])) if n else [],
                  fork_has_word=fork_has_word)
    renames = find_renames(files, old_text=src.upstream_before) if files else []
    _UP[prn] = (head, base, how, files, renames, src)
    return _UP[prn]


# ------------------------------------------------------------------ run
doc = json.loads(Path(a.sample).read_text()); sample = doc["sample"] if isinstance(doc, dict) else doc
if a.part:
    i, n = map(int, a.part.split("/")); sample = [s for j, s in enumerate(sample) if j % n == i]
if a.only: sample = [s for s in sample if a.only in s["key"]]
done = {json.loads(l)["key"] for l in open(OUT)} if OUT.exists() else set()
todo = [s for s in sample if s["key"] not in done]
if a.limit: todo = todo[:a.limit]
old = {json.loads(l)["key"]: json.loads(l) for l in open(a.old_verdicts)} if Path(a.old_verdicts).exists() else {}
screen = {json.loads(l)["key"]: json.loads(l) for l in open(a.screen)} if a.screen else {}
dirty = (fgit("status", "--porcelain", "--untracked-files=no") or "").strip()
if dirty: sys.exit(f"fork clone is not clean:\n{dirty}")
class PromptLog:
    """Keeps every repair prompt (the loop persists a prompt only with a candidate)."""
    def __init__(self, inner):
        self.inner = inner; self.dir = None
        for k in ("name", "model", "params", "model_digest", "request_limit_chars", "request_limit_chars_is_explicit"):
            setattr(self, k, getattr(inner, k, None))
    @property
    def last_usage(self): return getattr(self.inner, "last_usage", None)
    def generate(self, request):
        if self.dir is not None:
            (self.dir / f"prompt-{request.request_hash[:16]}.txt").write_text(request.prompt)
        return self.inner.generate(request)

class DryBackend:
    name = "dry"; model = "none"; model_digest = None; params = {"num_ctx": 16384}
    request_limit_chars = None; request_limit_chars_is_explicit = False; last_usage = None
    def generate(self, request): return "CANNOT_ADAPT: needs nothing from nowhere (dry run)"

backend = None
if a.arm != "A":
    backend = PromptLog(DryBackend() if a.dry_run else CachedBackend(OllamaBackend(), OUTD / "gen-cache"))
    print(f"backend {backend.model} digest={backend.model_digest} params={backend.params}", flush=True)
PROV = run_provenance(None if a.dry_run else backend, driver="ctx_reprompt.py", arm=a.arm, k=2, echo_policy="retry")
print(f"arm {a.arm}: {len(todo)} to run ({len(done)} done)", flush=True)

for n, s in enumerate(todo, 1):
    key = s["key"]; t_case = time.time(); prn = re.search(r"/PR-(\d+)/", key).group(1)
    sap_dir = Path(a.salp_out) / key; case_dir = ATT / key.replace("/", "__"); case_dir.mkdir(parents=True, exist_ok=True)
    o = old.get(key, {}); scr = screen.get(key, {})
    row = {"key": key, "group": s.get("group"), "arm": a.arm, "stored_verdict": o.get("verdict"),
           "screen_blocking": scr.get("blocking"), "screen_classes": scr.get("classes")}
    try:
        stored = json.loads((Path(a.candidates) / (key.replace("/", "__") + ".json")).read_text())
        manifest = json.loads((sap_dir.parent / "pr.json").read_text())
        sap = json.loads((sap_dir / "sap.json").read_text()); path = sap["target_file"]; row["file"] = path
        ext = EXT.get(sap["source_file"].rsplit(".", 1)[-1], "java")
        # the 477 run's units and contexts: same stages, no model call, whole-file targets as they were
        res = run_case(sap_dir, manifest, StubBackend("x"), narrow_whole_file_targets=False)
        if len(res.units) != len(stored["units"]):
            raise RuntimeError(f"unit count differs from the stored candidate: {len(res.units)} vs {len(stored['units'])}")
        units, src_texts = [], []
        for i, (u, su) in enumerate(zip(res.units, stored["units"])):
            ent = u.context.target_localization.get("function")
            art = next((x for x in u.context.program_context if x.role == "target" and x.entity == ent), None)
            orig = recover(art) if art else ""
            if ent != su["entity"] or orig != su["target_original"] or su.get("outcome") != "completion":
                raise RuntimeError(f"unit {i} ({ent}) does not match the stored candidate")
            def expand(c, art=art):
                for e in (art.placeholders if art else []):
                    c = c.replace(PLACEHOLDER.format(n=e.ph_id), e.original_text, 1)
                return c
            gen = GenerationRecord(case_id=u.request.case_id, outcome="completion", candidate=su["candidate_raw"],
                                   diagnostics=["stored 477 attempt-1 candidate; not regenerated"],
                                   usage=su.get("usage") or {}, request_hash=u.request.request_hash, exec_id="stored-477")
            st = UnitState(index=i, entity=ent, context=u.context, request=u.request, generation=gen, target_original=orig,
                           expand=expand, expected_placeholders=_expected_placeholders(u.context))
            st.art = art; units.append(st)
            fdir = sap_dir / ent
            src_texts.append(tuple((fdir / f"source.{side}.{ext}").read_text(errors="replace")
                                   if (fdir / f"source.{side}.{ext}").exists() else "" for side in ("before", "after")))
        row["units_info"] = [{"unit": u.index, "entity": u.entity, "kinds": list(unit_kinds(u.context)),
                              "placeholders": len(u.expected_placeholders or [])} for u in units]
        current = {}
        prompts = {}
        def on_attempt(attempt, arts):
            current["raw"] = [x["candidate_raw"] for x in arts]
            for x in arts:
                stem = f"attempt-{attempt:02d}.unit{x['unit']}"
                if attempt > 1: (case_dir / f"{stem}.prompt.txt").write_text(x["prompt"])
                (case_dir / f"{stem}.candidate.json").write_text(json.dumps(
                    {k: x[k] for k in ("unit", "entity", "candidate_raw", "candidate_expanded")}, indent=1))
        (case_dir / "target_original.json").write_text(json.dumps(
            [{"unit": u.index, "entity": u.entity, "target_original": u.target_original} for u in units], indent=1))
        validate = make_validator(key, path, units, lambda: current["raw"])
        if backend is not None: backend.dir = case_dir

        ctx_log = {}
        def repair_extras(i, validation):
            """Arm C: what the names the build could not resolve are (rcap.adaptctx)."""
            if validation.kind != "build": return {}
            vkey = validation.detail.get("val_key"); spans = validation.detail.get("unit_spans") or []
            log = (VC / f"{vkey}.build.log").read_text(errors="replace")
            spliced = (VC / f"{vkey}.spliced.txt").read_text()
            lines = spliced.splitlines(); prefix = str(FORK) + "/"
            croot = json.loads((VC / f"{vkey}.json").read_text()).get("root", str(FORK)).rstrip("/") + "/"
            errs = parse_compile_errors(log); mine = []
            for e in errs:
                e.rel = e.path[len(croot):] if e.path.startswith(croot) else (e.path[len(prefix):] if e.path.startswith(prefix) else e.path)
                if e.rel != path: continue
                if not e.body and 0 < e.line <= len(lines):
                    e.body = [lines[e.line - 1]] + ([" " * (e.col - 1) + "^"] if e.col else [])
                idx = next((j for j, (f, l) in enumerate(spans) if f <= e.line <= l), None)
                if idx == i or idx is None: mine.append(e)
            if not mine: mine = [e for e in errs if e.rel != path]      # only callers in other files fail
            missing = missing_symbols(mine) + import_symbols(mine)
            head, base, how, files, renames, src = upstream(prn)
            u = units[i]; before, after = src_texts[i]
            head_src = src.upstream_after(sap["source_file"])
            region = cc.UnitRegion(key=u.entity, source_file=sap["source_file"],
                                   after_span=cc.locate(after, head_src) if head_src else None,
                                   after_text=after, before_text=before)
            hints, owners = collections.defaultdict(list), {}
            for e in scr.get("errors", []):
                if e.get("symbol") and e.get("fork_equivalent"):
                    for line in e["fork_equivalent"][:3]:
                        h = f"fork equivalent found by the build-failure screen ({e.get('equiv_kind')}): {line}"
                        if h not in hints[e["symbol"]]: hints[e["symbol"]].append(h)
                if e.get("symbol") and e.get("owner"): owners.setdefault(e["symbol"], e["owner"])
            scs, omitted = build_symbol_context(
                missing, mine, files=files, region=region, unit_file=path, source_file=sap["source_file"], sources=src,
                scope_text=(current["raw"][i] or "") + "\n" + spliced, renames=renames, hints=hints, owners=owners)
            ctx_log[i] = {"missing": [m.__dict__ for m in missing], "upstream": {"head": head, "base": base, "base_how": how},
                          "renames": [r.label for r in renames][:20],
                          "symbols": [{"symbol": x.symbol, "owner": x.owner, "origin": x.origin, "upstream_file": x.upstream_file,
                                       "upstream_lines": x.upstream_lines, "complete": x.upstream_complete,
                                       "fork_members": len(x.fork_members), "hints": len(x.hints), "why": x.why,
                                       "missing_in_fork": x.missing_in_fork, "qualify": x.qualify} for x in scs],
                          "omitted": omitted}
            if not scs: return {}
            return {"symbol_context": scs, "symbol_context_omitted": omitted, "decline_prefill": a.arm != "C2"}

        if a.arm == "A":
            raw = [u.generation.candidate for u in units]; current["raw"] = raw
            v = validate(1, [u.expand(raw[i]) for i, u in enumerate(units)])
            row.update({"final_verdict": v.verdict, "error_summary": v.summary, "validation": v.detail,
                        "validation_seconds": v.seconds, "units_with_errors": v.units_with_errors,
                        "matches_stored_verdict": v.verdict == o.get("verdict")})
            (case_dir / "attempt-01.feedback.txt").write_text(v.feedback + "\n")
        else:
            rec = run_reprompt_loop(units, backend, validate, k=2, echo_policy="retry", on_attempt=on_attempt,
                                    repair_extras=repair_extras if a.arm in ("C", "C2") else None)
            row.update(rec); row["symbol_context"] = ctx_log
            # fidelity of what each unit ends with (attempt 2 when it produced a candidate, else attempt 1)
            last = current.get("raw") or [u.generation.candidate for u in units]
            fid = []
            for i, u in enumerate(units):
                before, after = src_texts[i]
                f = judge(u.target_original, u.expand(last[i]), before, after, rename_context="isolation")
                f1 = judge(u.target_original, u.expand(u.generation.candidate), before, after, rename_context="isolation")
                fid.append({"unit": i, "label": f.label, "recall": f.recall, "extra_share": f.extra_share,
                            "novel_share": f.novel_share, "reasons": f.reasons[:4], "evidence": f.evidence[:4],
                            "attempt1_label": f1.label, "attempt1_recall": f1.recall,
                            "changed_by_reprompt": last[i] != u.generation.candidate})
            row["fidelity"] = fid
    except Exception as exc:
        row.update({"stop_reason": "refused", "final_verdict": "NO_CANDIDATE", "error": f"{type(exc).__name__}: {exc}"[:400],
                    "traceback": traceback.format_exc()[-1200:]})
        subprocess.run(["git", "checkout", "-q", "--", "."], cwd=FORK)
    row["seconds"] = round(time.time() - t_case, 1)
    row["provenance"] = PROV
    with OUT.open("a") as fh: fh.write(json.dumps(row) + "\n")
    att = " ".join(str(x.get("verdict")) for x in row.get("attempts", [])) or row.get("final_verdict")
    print(f"[{n}/{len(todo)}] {key}: stored={row['stored_verdict']} -> {att} | stop={row.get('stop_reason')} "
          f"{row.get('error', '')[:120]} ({row['seconds']}s)", flush=True)
print("ALL DONE", flush=True)
