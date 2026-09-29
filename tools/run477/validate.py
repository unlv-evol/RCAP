#!/usr/bin/env python3
"""Splice -> build -> test each persisted RCAP candidate in its fork (SVRP-style funnel).

Per candidate SAP (validate/candidates/*.json), in the fork working copy at the
case's target commit:
  1. INTEGRATION  apply each unit in order: its original target text must occur
                  EXACTLY ONCE in the current file; it is replaced by the
                  placeholder-expanded candidate. Otherwise INTEGRATION_FAIL.
  2. BUILD        gradle :<project>:testClasses (main + test, Java + Scala).
  3. TEST         the file's own test class if it is a test, else <Class>Test in
                  the same project (else anywhere). Run with --tests; the same
                  test class is first run on the UNMODIFIED tree (cached), so only
                  NEW failures count against the candidate.
Verdicts: VALID | INTEGRATION_FAIL | BUILD_FAIL | TEST_FAIL | INCONCLUSIVE(reason).
The file is restored with git checkout after every case. Resumable by key.
Usage: validate.py <fork-worktree> <target_repo> [--limit N] [--only-key K]
"""
import json, os, re, subprocess, sys, time, glob, xml.etree.ElementTree as ET
from pathlib import Path
V = Path("/home/adam/Documents/work/rcap-class-project/run477/validate")
FORK = Path(sys.argv[1]).resolve(); REPO = sys.argv[2]
GRADLE = (V / "gradle_bin.txt").read_text().strip()
OUT = V / f"verdicts-{REPO.replace('/', '__')}.jsonl"
BASE = V / f"baseline-tests-{REPO.replace('/', '__')}.json"
TEST_TIMEOUT = int(os.environ.get("TEST_TIMEOUT", "1800")); BUILD_TIMEOUT = 1800

def sh(cmd, timeout):
    t = time.time()
    try:
        r = subprocess.run(cmd, cwd=FORK, capture_output=True, text=True, timeout=timeout, errors="replace")
        return r.returncode, (r.stdout + r.stderr)[-4000:], round(time.time() - t, 1)
    except subprocess.TimeoutExpired:
        return "timeout", "", round(time.time() - t, 1)

def gradle(*tasks, timeout):
    # Pin JDK 11 per invocation: ~/.gradle/gradle.properties points Gradle at JDK 17,
    # which this fork's Gradle 7.1.1 predates (and which excludes some suites).
    return sh([GRADLE, "--offline", "-q", "-Dorg.gradle.java.home=/usr/lib/jvm/java-11-openjdk-amd64",
               # 8 cores shared with the LLM run: cap test JVMs
               "-PmaxParallelForks=" + os.environ.get("MAX_FORKS", "3"), *tasks, "-x", "checkstyleMain", "-x", "checkstyleTest",
               "-x", "spotbugsMain", "-x", "spotbugsTest"], timeout)

def project_of(path):
    return ":" + path.split("/src/")[0].replace("/", ":") if "/src/" in path else None

def fqcn(path):
    """Fully-qualified class name from the file's own package clause(s): Kafka's
    core tests live under src/test/scala/unit/ and .../integration/, and those
    directory names are NOT part of the package. Scala may chain clauses."""
    try: src = (FORK / path).read_text(errors="replace")
    except OSError: src = ""
    pk = [m.group(1) for m in re.finditer(r"^\s*package\s+([\w.]+)\s*;?\s*$", src, re.M)]
    if pk: return ".".join(pk) + "." + Path(path).stem
    m = re.search(r"/src/\w+/(?:java|scala)/(.+)\.(?:java|scala)$", path)
    return m.group(1).replace("/", ".") if m else None

def concrete(rel):
    """An abstract test class runs only through its subclasses (e.g. ConsumerCoordinatorTest)."""
    src = (FORK / rel).read_text(errors="replace"); name = Path(rel).stem
    if not re.search(rf"abstract\s+class\s+{name}\b", src): return [fqcn(rel)]
    root = rel.split("/src/")[0]
    subs = [p for p in glob.glob(f"{FORK}/{root}/src/test/**/*.*", recursive=True)
            if p.endswith((".java", ".scala")) and re.search(rf"extends\s+{name}\b", Path(p).read_text(errors="replace"))]
    return [fqcn(os.path.relpath(p, FORK)) for p in subs]

def find_test(path):
    """(project, [fqcn...], how) of the test classes to run for this file."""
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
    args = [a for c in classes for a in ("--tests", c)]
    rc, log, secs = gradle(f"{proj}:test", *args, timeout=TEST_TIMEOUT)
    ran, failed = 0, []
    for f in glob.glob(f"{rdir}/*.xml"):
        root = ET.parse(f).getroot()
        for tc in root.iter("testcase"):
            ran += 1
            if tc.find("failure") is not None or tc.find("error") is not None:
                failed.append(f"{tc.get('classname')}.{tc.get('name')}")
    return {"rc": rc, "ran": ran, "failed": sorted(failed), "seconds": secs, "log_tail": log[-800:] if rc not in (0,) else ""}

done = {json.loads(l)["key"] for l in open(OUT)} if OUT.exists() else set()
baseline = json.loads(BASE.read_text()) if BASE.exists() else {}
cands = sorted(glob.glob(str(V / "candidates/*.json")))
if "--only-key" in sys.argv: cands = [c for c in cands if sys.argv[sys.argv.index("--only-key") + 1] in c]
todo = []
for c in cands:
    r = json.loads(Path(c).read_text())
    if r["target_repo"] == REPO and r["key"] not in done and r.get("outcome") == "completion": todo.append(r)
if "--limit" in sys.argv: todo = todo[:int(sys.argv[sys.argv.index("--limit") + 1])]
print(f"{REPO}: {len(todo)} candidates to validate ({len(done)} done)", flush=True)

for i, r in enumerate(todo, 1):
    t0 = time.time(); path = r["target_file"]; fpath = FORK / path
    row = {"key": r["key"], "file": path, "units": len(r["units"])}
    subprocess.run(["git", "checkout", "-q", "--", path], cwd=FORK)
    try:
        text = fpath.read_text(encoding="utf-8")
        orig = text; applied = []
        for u in r["units"]:
            n = text.count(u["target_original"] or "\0")
            if n == 1:
                text = text.replace(u["target_original"], u["candidate_expanded"], 1); applied.append("ok")
            else: applied.append(f"found_{n}")
        row["splice"] = applied
        if not any(a == "ok" for a in applied) or applied[0] != "ok":
            row["verdict"] = "INTEGRATION_FAIL"
        elif text == orig:
            row["verdict"] = "INCONCLUSIVE"; row["reason"] = "candidate identical to fork code (echo)"
        else:
            fpath.write_text(text, encoding="utf-8")
            proj = project_of(path)
            rc, log, secs = gradle(f"{proj}:testClasses", timeout=BUILD_TIMEOUT)
            row["build"] = {"rc": rc, "seconds": secs}
            if rc != 0:
                row["verdict"] = "BUILD_FAIL"; row["build"]["log_tail"] = "\n".join(
                    l for l in log.splitlines() if "error" in l.lower() or ".java:" in l or ".scala:" in l)[-1500:]
            else:
                tproj, tcls, how = find_test(path); row["test_class"] = tcls; row["test_how"] = how
                if not tcls:
                    how = how if how.startswith("no_") else "abstract test with no subclass"
                    row["verdict"] = "INCONCLUSIVE"; row["reason"] = f"build ok; {how}"
                else:
                    bkey = f"{tproj}|{','.join(tcls)}"
                    if bkey not in baseline:
                        fpath.write_text(orig, encoding="utf-8")
                        baseline[bkey] = run_test(tproj, tcls); BASE.write_text(json.dumps(baseline, indent=1))
                        fpath.write_text(text, encoding="utf-8")
                    b = baseline[bkey]; t = run_test(tproj, tcls)
                    row["test"] = {k: t[k] for k in ("rc", "ran", "failed", "seconds")}
                    row["baseline"] = {k: b[k] for k in ("rc", "ran", "failed")}
                    new = sorted(set(t["failed"]) - set(b["failed"]))
                    row["new_failures"] = new
                    if b["rc"] == "timeout" or b["ran"] == 0:
                        row["verdict"] = "INCONCLUSIVE"; row["reason"] = "baseline test did not run cleanly"
                    elif t["rc"] == "timeout":
                        row["verdict"] = "INCONCLUSIVE"; row["reason"] = "test timeout"
                    elif t["ran"] == 0:
                        row["verdict"] = "TEST_FAIL"; row["reason"] = "tests did not run with candidate"; row["test"]["log_tail"] = t["log_tail"]
                    elif new:
                        row["verdict"] = "TEST_FAIL"
                    else:
                        row["verdict"] = "VALID"
    except Exception as exc:
        row["verdict"] = "INCONCLUSIVE"; row["reason"] = f"harness error: {exc!r}"[:300]
    finally:
        subprocess.run(["git", "checkout", "-q", "--", path], cwd=FORK)
    row["seconds"] = round(time.time() - t0, 1)
    with OUT.open("a") as fh: fh.write(json.dumps(row) + "\n")
    print(f"[{i}/{len(todo)}] {r['key']}: {row['verdict']} {row.get('reason','')} ({row['seconds']}s)", flush=True)
print("ALL DONE", flush=True)
