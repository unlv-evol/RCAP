#!/usr/bin/env python3
"""PR-level validation: splice ALL of a PR's candidates into the fork at once, then build/test.

validate.py splices one SAP at a time, so a candidate whose main-code or test-code partner
lives in another SAP of the same PR fails to build even when the pair is right. Here, per PR
(multi-SAP PRs only by default):
  1. INTEGRATION  every unit of every completion candidate spliced exactly-once (as validate.py);
                  files whose first unit does not splice are left untouched and recorded.
  2. BUILD        one gradle call: :<proj>:testClasses for every touched project.
  3. TEST         the union of each touched file's test classes (validate.find_test rules),
                  one gradle test call per project, vs the cached per-class baseline.
Verdicts as validate.py, per PR. Every touched file is restored afterwards. Resumable by PR.
Usage: validate_pr.py <fork-worktree> <target_repo> [--all] [--only-pr N]
"""
import collections, json, os, re, sys, time
from pathlib import Path
import subprocess
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src")); sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import lang_of, screen_unit, unit_sources  # noqa: E402
from rcap.provenance import run_provenance  # noqa: E402
PROV = run_provenance(None, driver="validate_pr.py")
# validate.py's helpers (everything before its main loop); baseline keys "proj|Class" are shared
exec(compile(Path(__file__).with_name("validate.py").read_text().split("\ndone = ")[0], "validate.py", "exec"))
baseline = json.loads(BASE.read_text()) if BASE.exists() else {}
OUT_PR = V / f"verdicts-pr-{REPO.replace('/', '__')}.jsonl"

cands = collections.defaultdict(list)
for c in sorted(glob.glob(str(V / "candidates/*.json"))):
    r = json.loads(Path(c).read_text())
    if r["target_repo"] == REPO and r.get("outcome") == "completion":
        cands[re.search(r"/PR-(\d+)/", r["key"]).group(1)].append(r)
prs = sorted(p for p, rs in cands.items() if len(rs) > 1 or "--all" in sys.argv)
if "--only-pr" in sys.argv: prs = [sys.argv[sys.argv.index("--only-pr") + 1]]
done = {json.loads(l)["pr"] for l in open(OUT_PR)} if OUT_PR.exists() else set()
todo = [p for p in prs if p not in done]
print(f"{REPO}: {len(todo)} PRs to validate ({len(done)} done)", flush=True)

def restore(paths):
    if paths: subprocess.run(["git", "checkout", "-q", "--", *paths], cwd=FORK)

for i, pr in enumerate(todo, 1):
    t0 = time.time(); rs = cands[pr]; paths = [r["target_file"] for r in rs]
    row = {"pr": pr, "saps": [r["key"] for r in rs], "files": {}}
    restore(paths)
    try:
        spliced = {}
        for r in rs:
            p = r["target_file"]; text = (FORK / p).read_text(encoding="utf-8"); orig = text; applied = []
            for u in r["units"]:
                rej = screen_unit(u["target_original"] or "", u["candidate_expanded"], *unit_sources(V.parent / "salp-477-out" / r["key"], u.get("entity")), lang_of(p))
                if rej: applied.append(rej[0])
                elif text.count(u["target_original"] or "\0") == 1:
                    text = text.replace(u["target_original"], u["candidate_expanded"], 1); applied.append("ok")
                else: applied.append("miss")
            ok = applied[0] == "ok" and text != orig
            row["files"][p] = {"splice": applied, "applied": ok}
            if ok: spliced[p] = (orig, text)
        if not spliced:
            row["verdict"] = "INTEGRATION_FAIL"
        else:
            projs = sorted({project_of(p) for p in spliced})
            tests = collections.defaultdict(set)
            for p in spliced:
                tproj, tcls, how = find_test(p); row["files"][p]["test_how"] = how
                if tcls: tests[tproj].update(tcls)
            for p, (_, text) in spliced.items(): (FORK / p).write_text(text, encoding="utf-8")
            rc, log, secs = gradle(*[f"{pj}:testClasses" for pj in projs], timeout=BUILD_TIMEOUT)
            missing = [(tp, c) for tp, cs in tests.items() for c in sorted(cs) if f"{tp}|{c}" not in baseline]
            if rc == 0 and missing:
                # baselines only when the build passed, on the unmodified tree
                for p, (orig, _) in spliced.items(): (FORK / p).write_text(orig, encoding="utf-8")
                for tp, c in missing:
                    baseline[f"{tp}|{c}"] = run_test(tp, [c]); BASE.write_text(json.dumps(baseline, indent=1))
                for p, (_, text) in spliced.items(): (FORK / p).write_text(text, encoding="utf-8")
                rc2, _, _ = gradle(*[f"{pj}:testClasses" for pj in projs], timeout=BUILD_TIMEOUT)
                assert rc2 == 0, "rebuild after baseline failed"
            row["build"] = {"rc": rc, "seconds": secs, "projects": projs}
            if rc != 0:
                row["verdict"] = "BUILD_FAIL"; row["build"]["log_tail"] = "\n".join(
                    l for l in log.splitlines() if "error" in l.lower() or ".java:" in l or ".scala:" in l)[-3000:]
            elif not tests:
                row["verdict"] = "INCONCLUSIVE"; row["reason"] = "build ok; no test class for any file"
            else:
                row["tests"] = {}; new_all, bad_base, t_out = [], [], False
                for tproj, tcls in tests.items():
                    t = run_test(tproj, sorted(tcls))
                    b_failed = set(); runnable = []
                    for c in sorted(tcls):
                        b = baseline[f"{tproj}|{c}"]
                        if b["rc"] == "timeout" or b["ran"] == 0: bad_base.append(c)
                        else: runnable.append(c); b_failed |= set(b["failed"])
                    new = sorted(f for f in set(t["failed"]) - b_failed if any(f.startswith(c + ".") for c in runnable))
                    row["tests"][tproj] = {"classes": sorted(tcls), "rc": t["rc"], "ran": t["ran"], "new_failures": new,
                                           "baseline_unusable": [c for c in bad_base if c in tcls]}
                    new_all += new; t_out |= t["rc"] == "timeout"
                if t_out: row["verdict"] = "INCONCLUSIVE"; row["reason"] = "test timeout"
                elif new_all: row["verdict"] = "TEST_FAIL"
                elif len(bad_base) == sum(len(c) for c in tests.values()):
                    row["verdict"] = "INCONCLUSIVE"; row["reason"] = "no usable baseline test"
                else: row["verdict"] = "VALID"
    except Exception as exc:
        row["verdict"] = "INCONCLUSIVE"; row["reason"] = f"harness error: {exc!r}"[:300]
    finally:
        restore(paths)
    row["seconds"] = round(time.time() - t0, 1)
    row["provenance"] = PROV
    with OUT_PR.open("a") as fh: fh.write(json.dumps(row) + "\n")
    print(f"[{i}/{len(todo)}] PR-{pr} ({len(rs)} SAPs): {row['verdict']} {row.get('reason', '')} ({row['seconds']}s)", flush=True)
print("ALL DONE", flush=True)
