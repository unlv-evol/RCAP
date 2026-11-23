#!/usr/bin/env python3
"""Standalone re-validation of stored candidates with the merged splice (and, optionally, the
deterministic import post-pass). No model call.

validate.py spliced a SAP's units one after the other by exact text search; a unit that overlaps an
earlier one was dropped ("found_0"). Here every unit is an edit against the ORIGINAL file and the
units are merged at hunk level (rcap.splice). Otherwise the funnel is validate.py's:

  1. MERGE   rcap.splice.merge_units; a unit whose expanded candidate still carries an RCAP_PH_
             marker is not applied (placeholder_leak). Nothing applicable -> INTEGRATION_FAIL.
  2. IMPORTS (--fix-imports) rcap.imports.fix_imports against the upstream AFTER version of the file.
  3. BUILD   gradle :<project>:testClasses, same flags as validate.py.
  4. TEST    the file's test class(es) vs the cached baseline of the unmodified tree.

  revalidate.py --fork DIR --baseline FILE --keys keys.txt --out OUT.jsonl [--fix-imports]
                [--candidates DIR] [--repo-cache DIR]

One JSON row per key (resumable). Full build logs of failures go to <out stem>-logs/.
"""
import argparse, json, re, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from harness import Harness, lang_of, screen_unit, unit_sources  # noqa: E402
from rcap.provenance import run_provenance  # noqa: E402
from rcap.imports import fix_imports, unused_import_fails_build  # noqa: E402
from rcap.splice import NOT_APPLIED, UnitEdit, merge_units  # noqa: E402

RUN = Path("/home/adam/Documents/work/rcap-class-project/run477")
ap = argparse.ArgumentParser()
ap.add_argument("--fork", required=True); ap.add_argument("--baseline", required=True)
ap.add_argument("--keys", required=True, help="file with one candidate key per line")
ap.add_argument("--out", required=True)
ap.add_argument("--candidates", default=str(RUN / "validate/candidates"))
ap.add_argument("--fix-imports", action="store_true")
ap.add_argument("--repo-cache", default="/home/adam/Documents/work/rcap-class-project/feature-reprompt/repo-cache")
ap.add_argument("--source-repo", default="apache__kafka.git")
ap.add_argument("--gradle-bin-file", default=str(RUN / "validate/gradle_bin.txt"))
ap.add_argument("--limit", type=int)
ap.add_argument("--salp-out", default=str(RUN / "salp-477-out"), help="SAP directories (upstream texts for the duplicate check)")
a = ap.parse_args()
PROV = run_provenance(None, driver="revalidate.py", candidates=a.candidates)

H = Harness(a.fork, Path(a.gradle_bin_file).read_text().strip()); FORK = H.fork
OUT = Path(a.out); LOGS = OUT.with_name(OUT.stem + "-logs"); LOGS.mkdir(parents=True, exist_ok=True)
BASE = Path(a.baseline); baseline = json.loads(BASE.read_text()) if BASE.exists() else {}
SRC = Path(a.repo_cache) / a.source_repo
TEST_TIMEOUT = 1800; BUILD_TIMEOUT = 1800
FACTS = H.build_facts()

def upstream_after(key, source_file):
    prn = re.search(r"/PR-(\d+)/", key).group(1)
    r = subprocess.run(["git", "--git-dir", str(SRC), "show", f"refs/salp/pr/{prn}:{source_file}"], capture_output=True, text=True, errors="replace")
    return r.stdout if r.returncode == 0 else None

keys = [k.strip() for k in Path(a.keys).read_text().splitlines() if k.strip()]
done = {json.loads(l)["key"] for l in open(OUT)} if OUT.exists() else set()
todo = [k for k in keys if k not in done]
if a.limit: todo = todo[:a.limit]
dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=FORK, capture_output=True, text=True).stdout.strip()
if dirty: sys.exit(f"fork clone is not clean:\n{dirty}")
print(f"{len(todo)} to validate ({len(done)} done); fix_imports={a.fix_imports}; build facts {FACTS}", flush=True)

for i, key in enumerate(todo, 1):
    t0 = time.time()
    r = json.loads((Path(a.candidates) / (key.replace("/", "__") + ".json")).read_text())
    path = r["target_file"]; fpath = FORK / path
    row = {"key": key, "file": path, "units": len(r["units"]), "fix_imports": a.fix_imports,
           "provenance": {**PROV, "generation": r.get("provenance")}}
    try:
        orig = fpath.read_text(encoding="utf-8")
        edits, leaks = [], {}
        for n, u in enumerate(r["units"]):
            if u.get("outcome") != "completion" or not u.get("candidate_expanded"): continue
            b_, a_ = unit_sources(Path(a.salp_out) / key, u.get("entity"))
            rej = screen_unit(u["target_original"] or "", u["candidate_expanded"], b_, a_, lang_of(path))
            if rej: leaks[n] = rej; continue
            edits.append(UnitEdit(f"u{n}", u["target_original"] or "", u["candidate_expanded"]))
        m = merge_units(orig, edits, language=lang_of(path))
        row["splice"] = {k: um.as_dict() for k, um in m.units.items()}
        row["rejected"] = {f"u{n}": {"reason": rj[0], "evidence": rj[1][:3]} for n, rj in leaks.items()}
        row["splice_status"] = [(leaks[n][0] if n in leaks else m.units[f"u{n}"].status if f"u{n}" in m.units else "no_candidate")
                                for n in range(len(r["units"]))]
        if m.violations: raise RuntimeError("merge invariant violated: " + "; ".join(m.violations)[:240])
        text = m.text
        if a.fix_imports and m.changed:
            up = upstream_after(key, r["source_file"])
            lang = "scala" if path.endswith((".scala", ".sc")) else "java"
            if up is None: row["import_fix"] = {"error": "upstream after-version not readable"}
            else:
                fx = fix_imports(orig, text, up, lang, exists=lambda imp: H.import_exists(imp, path),
                                 remove_unused=unused_import_fails_build(lang, checkstyle=FACTS["checkstyle"],
                                                                         scala_fatal_warnings=FACTS["scala_fatal_warnings"]))
                row["import_fix"] = fx.as_dict(); text = fx.text
        if not edits or all(um.status in NOT_APPLIED for um in m.units.values()):
            row["verdict"] = "INTEGRATION_FAIL"
            row["reason"] = (" / ".join(sorted({rj[0] for rj in leaks.values()})).replace("_", " ") if leaks and not edits
                             else "no unit could be located")
        elif text == orig:
            row["verdict"] = "INCONCLUSIVE"; row["reason"] = "candidate identical to fork code (echo)"
        else:
            fpath.write_text(text, encoding="utf-8")
            proj = H.project_of(path)
            rc, log, secs = H.gradle(f"{proj}:testClasses", timeout=BUILD_TIMEOUT, keep_going=False)
            row["build"] = {"rc": rc, "seconds": secs}
            if rc != 0:
                row["verdict"] = "BUILD_FAIL"
                row["build"]["log_tail"] = "\n".join(l for l in log.splitlines() if "error" in l.lower() or ".java:" in l or ".scala:" in l)[-1500:]
                (LOGS / (key.replace("/", "__") + ".build.log")).write_text(log)
            else:
                tproj, tcls, how = H.find_test(path); row["test_class"] = tcls; row["test_how"] = how
                if not tcls:
                    how = how if how.startswith("no_") else "abstract test with no subclass"
                    row["verdict"] = "INCONCLUSIVE"; row["reason"] = f"build ok; {how}"
                else:
                    bkey = f"{tproj}|{','.join(tcls)}"
                    if bkey not in baseline:
                        fpath.write_text(orig, encoding="utf-8")
                        baseline[bkey] = H.run_test_summary(tproj, tcls, TEST_TIMEOUT); BASE.write_text(json.dumps(baseline, indent=1))
                        fpath.write_text(text, encoding="utf-8")
                    b = baseline[bkey]; t = H.run_test_summary(tproj, tcls, TEST_TIMEOUT)
                    row["test"] = {k: t[k] for k in ("rc", "ran", "failed", "seconds")}
                    row["baseline"] = {k: b[k] for k in ("rc", "ran", "failed")}
                    new = sorted(set(t["failed"]) - set(b["failed"])); row["new_failures"] = new
                    if b["rc"] == "timeout" or b["ran"] == 0:
                        row["verdict"] = "INCONCLUSIVE"; row["reason"] = "baseline test did not run cleanly"
                    elif t["rc"] == "timeout":
                        row["verdict"] = "INCONCLUSIVE"; row["reason"] = "test timeout"
                    elif t["ran"] == 0:
                        row["verdict"] = "TEST_FAIL"; row["reason"] = "tests did not run with candidate"; row["test"]["log_tail"] = t["log_tail"]
                    elif new: row["verdict"] = "TEST_FAIL"
                    else: row["verdict"] = "VALID"
    except Exception as exc:
        row["verdict"] = "INCONCLUSIVE"; row["reason"] = f"harness error: {exc!r}"[:300]
    finally:
        subprocess.run(["git", "checkout", "-q", "--", path], cwd=FORK)
    row["seconds"] = round(time.time() - t0, 1)
    with OUT.open("a") as fh: fh.write(json.dumps(row) + "\n")
    print(f"[{i}/{len(todo)}] {key}: {row['verdict']} {row.get('reason', '')} splice={row.get('splice_status')} "
          f"imports+{len(row.get('import_fix', {}).get('added', []))} ({row['seconds']}s)", flush=True)
print("ALL DONE", flush=True)
