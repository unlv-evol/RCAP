#!/usr/bin/env python3
"""Re-prompt loop over a sample of SAPs: generate -> splice -> build -> test -> feed back.

Attempt 1 is the ordinary RCAP generation (rcap.pipeline.run_case). Every
attempt is spliced into the fork clone and validated exactly as validate.py
does (same Gradle tasks and flags, same baseline rule: only failures that do
not occur on the unmodified tree count). A BUILD_FAIL or TEST_FAIL is turned
into feedback (rcap.feedback) and a fresh single-turn repair request
(rcap.reprompt), up to --k attempts in total. Unlike validate.py the FULL
build/test output is kept, per attempt, next to the prompt and the candidate.

  reprompt.py --sample sample.json --salp-out DIR --fork DIR --out results.jsonl
              --baseline baseline-tests.json [--k 10] [--only SUBSTR] [--limit N]

`sample.json`: [{"old_key": ..., "new_dir": "<pair>/PR-N/sap-...", "group": ...}].
Resumable by old_key. The fork file is restored after every attempt. The
baseline cache is read and, for a class it lacks, extended in place — pass a
private copy. Nothing outside --out's directory, the cache and the fork
clone's working tree is written.
"""
import argparse, glob, hashlib, json, os, re, subprocess, sys, time, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from rcap.backend_ollama import OllamaBackend
from rcap.feedback import extract_build_feedback, extract_test_feedback, parse_junit_failures
from rcap.pipeline import _expected_placeholders, run_case
from rcap.provenance import run_provenance
from rcap.reduction_program import PLACEHOLDER, recover
from rcap.reprompt import UnitState, Validation, run_reprompt_loop
from rcap.unitkind import shape_mismatch

RUN = Path("/home/adam/Documents/work/rcap-class-project/run477")
ap = argparse.ArgumentParser()
ap.add_argument("--sample", required=True); ap.add_argument("--salp-out", required=True)
ap.add_argument("--fork", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--baseline", required=True)
ap.add_argument("--k", type=int, default=10)
ap.add_argument("--echo-policy", default="feedback", choices=["feedback", "retry"])
ap.add_argument("--dialogue", action="store_true",
                help="one conversation per unit (rcap.reprompt dialogue mode) instead of fresh repair requests")
ap.add_argument("--attempts-dir", help="where per-attempt artifacts go (default: next to --out)")
ap.add_argument("--gradle-bin-file", default=str(RUN / "validate/gradle_bin.txt"))
ap.add_argument("--old-verdicts", default=str(RUN / "validate/verdicts-linkedin__kafka.v2.jsonl"))
ap.add_argument("--old-candidates", default=str(RUN / "validate/candidates"))
ap.add_argument("--only"); ap.add_argument("--limit", type=int)
a = ap.parse_args()

FORK = Path(a.fork).resolve(); OUT = Path(a.out); ATT = Path(a.attempts_dir) if a.attempts_dir else OUT.parent / ("attempts" if a.echo_policy == "feedback" else f"attempts-echo-{a.echo_policy}")
GRADLE = Path(a.gradle_bin_file).read_text().strip()
BASE = Path(a.baseline); baseline = json.loads(BASE.read_text()) if BASE.exists() else {}
TEST_TIMEOUT = int(os.environ.get("TEST_TIMEOUT", "1800")); BUILD_TIMEOUT = 1800

def sh(cmd, timeout):
    t = time.time()
    try:
        r = subprocess.run(cmd, cwd=FORK, capture_output=True, text=True, timeout=timeout, errors="replace")
        return r.returncode, r.stdout + r.stderr, round(time.time() - t, 1)   # FULL output
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"") + (e.stderr or b"")
        return "timeout", out.decode(errors="replace") if isinstance(out, bytes) else out, round(time.time() - t, 1)

def gradle(*tasks, timeout):
    # identical to validate.py: JDK 11 pinned per call, offline, static analysis and formatting excluded
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
    """Line of the spliced (placeholder-expanded) candidate -> line of the candidate
    as the model wrote it. A line inside expanded elided code maps to the
    placeholder's own line. Only valid when each placeholder occurs once."""
    spots = []
    for e in art.placeholders:
        ph = PLACEHOLDER.format(n=e.ph_id); pos = raw.find(ph)
        if pos < 0 or raw.count(ph) != 1: return lambda n: None
        spots.append((raw.count("\n", 0, pos) + 1, e.original_text.count("\n") - ph.count("\n")))
    spots.sort()
    def to_raw(n):
        shift = 0
        for raw_line, extra in spots:
            start = raw_line + shift            # expanded line where this placeholder begins
            if n < start: break
            if n <= start + extra: return raw_line
            shift += extra
        return n - shift
    return to_raw

def make_validator(case_dir, path, units, raws):
    fpath = FORK / path

    def validate(attempt, expanded):
        t0 = time.time(); tag = f"attempt-{attempt:02d}"; detail = {}
        subprocess.run(["git", "checkout", "-q", "--", path], cwd=FORK)
        try:
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
            fpath.write_text(text, encoding="utf-8")
            proj = project_of(path)
            rc, log, secs = gradle(f"{proj}:testClasses", timeout=BUILD_TIMEOUT)
            (case_dir / f"{tag}.build.log").write_text(log)
            detail["build"] = {"rc": rc, "seconds": secs, "log_chars": len(log)}
            if rc == "timeout":
                return Validation(verdict="INCONCLUSIVE", summary="build timeout", detail=detail)
            if rc != 0:
                mappers = [line_mapper(raws()[i], u.art) for i, u in enumerate(units)]
                fb = extract_build_feedback(log, path, spans, root=str(FORK),
                                            unit_line=lambda i, n: mappers[i](n), edited_source=text)
                detail["feedback"] = {"errors_total": fb.n_total, "errors_shown": fb.n_shown,
                                      "chars": len(fb.text), "mappable": fb.mappable}
                (case_dir / f"{tag}.feedback.txt").write_text(fb.text + "\n")
                return Validation(verdict="BUILD_FAIL", kind="build", feedback=fb.text,
                                  unit_feedback=fb.per_unit, units_with_errors=fb.units_with_errors,
                                  summary=fb.summary, detail=detail)
            tproj, tcls, how = find_test(path); detail["test_class"] = tcls; detail["test_how"] = how
            if not tcls:
                how = how if how.startswith("no_") else "abstract test with no subclass"
                return Validation(verdict="INCONCLUSIVE", summary=f"build ok; {how}", detail=detail)
            bkey = f"{tproj}|{','.join(tcls)}"
            if bkey not in baseline:
                fpath.write_text(orig, encoding="utf-8")
                baseline[bkey], _, blog = run_test(tproj, tcls); BASE.write_text(json.dumps(baseline, indent=1))
                (case_dir / "baseline.test.log").write_text(blog)
                fpath.write_text(text, encoding="utf-8")
            b = baseline[bkey]; t, failures, tlog = run_test(tproj, tcls)
            (case_dir / f"{tag}.test.log").write_text(tlog)
            new = sorted(set(t["failed"]) - set(b["failed"]))
            detail["test"] = t; detail["baseline"] = {k: b[k] for k in ("rc", "ran", "failed")}
            detail["new_failures"] = new
            (case_dir / f"{tag}.tests.json").write_text(json.dumps(
                {"ran": t["ran"], "new_failures": new,
                 "failures": [f.__dict__ for f in failures]}, indent=1))
            if b["rc"] == "timeout" or b["ran"] == 0:
                return Validation(verdict="INCONCLUSIVE", summary="baseline test did not run cleanly", detail=detail)
            if t["rc"] == "timeout":
                return Validation(verdict="INCONCLUSIVE", summary="test timeout", detail=detail)
            if t["ran"] == 0 or new:
                fb = extract_test_feedback(failures, b["failed"], ran=t["ran"], gradle_log=tlog)
                detail["feedback"] = {"failures_total": fb.n_total, "groups_shown": fb.n_shown, "chars": len(fb.text)}
                (case_dir / f"{tag}.feedback.txt").write_text(fb.text + "\n")
                return Validation(verdict="TEST_FAIL", kind="test", feedback=fb.text, summary=fb.summary, detail=detail)
            return Validation(verdict="VALID", summary=f"{t['ran']} tests ran, no new failures", detail=detail)
        except Exception as exc:
            return Validation(verdict="INCONCLUSIVE", summary=f"harness error: {exc!r}"[:300], detail=detail)
        finally:
            subprocess.run(["git", "checkout", "-q", "--", path], cwd=FORK)

    def timed(attempt, expanded):
        t0 = time.time(); v = validate(attempt, expanded); v.seconds = round(time.time() - t0, 1)
        return v
    return timed

sample = json.loads(Path(a.sample).read_text())
if a.only: sample = [s for s in sample if a.only in s["old_key"]]
done = {json.loads(l)["key"] for l in open(OUT)} if OUT.exists() else set()
todo = [s for s in sample if s["old_key"] not in done]
if a.limit: todo = todo[:a.limit]
old = {}
if Path(a.old_verdicts).exists():
    old = {json.loads(l)["key"]: json.loads(l) for l in open(a.old_verdicts)}
OUT.parent.mkdir(parents=True, exist_ok=True)
class Recorded:
    """The backend, with every call kept: the messages sent and the reply exactly as
    the model wrote it (also when no code could be taken from it)."""
    def __init__(self, inner): self.inner = inner; self.calls = []
    def __getattr__(self, name): return getattr(self.inner, name)
    def generate(self, request, **kw):
        call = {"messages": [*(getattr(request, "history", None) or []), {"role": "user", "content": request.prompt}]}
        self.calls.append(call)
        call["reply"] = self.inner.generate(request, **kw)
        call["done_reason"] = getattr(self.inner, "last_done_reason", None)
        return call["reply"]

backend = Recorded(OllamaBackend())
PROV = run_provenance(backend, driver="reprompt.py", k=a.k, echo_policy=a.echo_policy, dialogue=a.dialogue)
print(f"backend {backend.model} digest={backend.model_digest} params={backend.params}; "
      f"k={a.k}; {len(todo)} to run ({len(done)} done)", flush=True)

for n, s in enumerate(todo, 1):
    key = s["old_key"]; t_case = time.time()
    sap_dir = Path(a.salp_out) / s["new_dir"]
    case_dir = ATT / key.replace("/", "__"); case_dir.mkdir(parents=True, exist_ok=True)
    o = old.get(key, {})
    row = {"key": key, "group": s.get("group"), "new_sap": s["new_dir"],
           "old_verdict": o.get("verdict"), "old_build_cause": o.get("build_cause"), "k": a.k}
    backend.calls = []
    try:
        manifest = json.loads((sap_dir.parent / "pr.json").read_text())
        sap = json.loads((sap_dir / "sap.json").read_text())
        row["file"] = sap["target_file"]
        t0 = time.time()
        res = run_case(sap_dir, manifest, backend)      # attempt 1: the normal pipeline, unchanged
        gen1 = round(time.time() - t0, 1)
        units = []
        for i, u in enumerate(res.units):
            ent = u.context.target_localization.get("function")
            art = next((x for x in u.context.program_context if x.role == "target" and x.entity == ent), None)
            def expand(c, art=art):
                for e in (art.placeholders if art else []):
                    c = c.replace(PLACEHOLDER.format(n=e.ph_id), e.original_text, 1)
                return c
            st = UnitState(index=i, entity=ent, context=u.context, request=u.request, generation=u.generation,
                           target_original=recover(art) if art else "", expand=expand,
                           expected_placeholders=_expected_placeholders(u.context))
            st.art = art; units.append(st)
        # did the new SALP change what the model is asked / what it answers?
        oc = Path(a.old_candidates) / (key.replace("/", "__") + ".json")
        if oc.exists():
            ocj = json.loads(oc.read_text())
            row["attempt1_prompt_matches_old"] = [
                hashlib.sha256(u.request.prompt.encode()).hexdigest() == ou.get("prompt_sha256")
                for u, ou in zip(units, ocj["units"])]
            row["attempt1_candidate_matches_old"] = [
                u.generation.candidate == ou.get("candidate_raw") for u, ou in zip(units, ocj["units"])]
        current = {}
        def on_attempt(attempt, arts):
            current["raw"] = [x["candidate_raw"] for x in arts]
            for x in arts:
                stem = f"attempt-{attempt:02d}.unit{x['unit']}"
                (case_dir / f"{stem}.prompt.txt").write_text(x["prompt"])
                (case_dir / f"{stem}.candidate.json").write_text(json.dumps(
                    {k: x[k] for k in ("unit", "entity", "candidate_raw", "candidate_expanded")}, indent=1))
        (case_dir / "target_original.json").write_text(json.dumps(
            [{"unit": u.index, "entity": u.entity, "target_original": u.target_original} for u in units], indent=1))
        rec = run_reprompt_loop(units, backend, make_validator(case_dir, sap["target_file"], units, lambda: current["raw"]),
                                k=a.k, echo_policy=a.echo_policy, on_attempt=on_attempt, dialogue=a.dialogue)
        if rec["attempts"]:
            for u in rec["attempts"][0]["units"]: u["gen_seconds"] = round(gen1 / max(1, len(units)), 1)
        row.update(rec)
    except Exception as exc:
        row.update({"stop_reason": "refused", "final_verdict": "NO_CANDIDATE", "first_success_attempt": None,
                    "attempts": [], "error": f"{type(exc).__name__}: {exc}"[:400],
                    "stage": getattr(exc, "stage", None), "traceback": traceback.format_exc()[-800:]})
    with (case_dir / "calls.jsonl").open("w") as fh:        # every model call of this case, in order
        for c in backend.calls: fh.write(json.dumps(c) + "\n")
    row["model_calls"] = len(backend.calls)
    row["seconds"] = round(time.time() - t_case, 1)
    row["provenance"] = PROV
    with OUT.open("a") as fh: fh.write(json.dumps(row) + "\n")
    print(f"[{n}/{len(todo)}] {key}: old={row['old_verdict']} -> "
          + " ".join(str(x.get('verdict')) for x in row["attempts"])
          + f" | stop={row['stop_reason']} first_success={row['first_success_attempt']} ({row['seconds']}s)", flush=True)
print("ALL DONE", flush=True)
