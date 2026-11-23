#!/usr/bin/env python3
"""Re-splice each BUILD_FAIL candidate and rebuild it, keeping the FULL compiler log.

validate.py kept only lines containing "error"/".java:"/".scala:", which drops javac's
"symbol:" / "location:" continuation lines, so the missing symbol of a Java
"cannot find symbol" is not recoverable from verdicts-*.jsonl. This re-runs the exact
same splice + `gradle :<proj>:testClasses` and writes the whole output to
build-logs/<key>.log. The verdict must reproduce (rc != 0); a mismatch is recorded.
Resumable: skips keys whose log already exists.
Usage: rebuild_logs.py <fork-worktree> <target_repo> [--shard I/N]   (one shard per worktree)
"""
import json, os, subprocess, sys, time
from pathlib import Path
V = Path("/home/adam/Documents/work/rcap-class-project/run477/validate")
FORK = Path(sys.argv[1]).resolve(); REPO = sys.argv[2]
GRADLE = (V / "gradle_bin.txt").read_text().strip()
LOGS = V / "build-logs"; LOGS.mkdir(exist_ok=True)
OUT = V / f"rebuild-{REPO.replace('/', '__')}.jsonl"

def safe(key): return key.replace("/", "__")

verdicts = [json.loads(l) for l in open(V / f"verdicts-{REPO.replace('/', '__')}.jsonl")]
fails = {v["key"] for v in verdicts if v["verdict"] == "BUILD_FAIL"}
cands = {}
for p in (V / "candidates").glob("*.json"):
    r = json.loads(p.read_text())
    if r["key"] in fails: cands[r["key"]] = r
todo = [k for k in sorted(fails) if not (LOGS / f"{safe(k)}.log").exists()]
if "--shard" in sys.argv:
    si, sn = map(int, sys.argv[sys.argv.index("--shard") + 1].split("/"))
    todo = todo[si::sn]
print(f"{len(todo)} to rebuild ({len(fails) - len(todo)} done)", flush=True)
for i, key in enumerate(todo, 1):
    if (LOGS / f"{safe(key)}.log").exists(): continue
    r = cands[key]; path = r["target_file"]; fpath = FORK / path
    subprocess.run(["git", "checkout", "-q", "--", path], cwd=FORK)
    try:
        text = fpath.read_text(encoding="utf-8")
        for u in r["units"]:
            if text.count(u["target_original"] or "\0") == 1:
                text = text.replace(u["target_original"], u["candidate_expanded"], 1)
        fpath.write_text(text, encoding="utf-8")
        proj = ":" + path.split("/src/")[0].replace("/", ":")
        t = time.time()
        p = subprocess.run([GRADLE, "--offline", "-q", "-Dorg.gradle.java.home=/usr/lib/jvm/java-11-openjdk-amd64",
                            "-PmaxParallelForks=3", f"{proj}:testClasses", "-x", "checkstyleMain", "-x", "checkstyleTest",
                            "-x", "spotbugsMain", "-x", "spotbugsTest"],
                           cwd=FORK, capture_output=True, text=True, timeout=1800, errors="replace")
        (LOGS / f"{safe(key)}.log").write_text(p.stdout + p.stderr)
        row = {"key": key, "rc": p.returncode, "reproduced": p.returncode != 0, "seconds": round(time.time() - t, 1)}
    except Exception as exc:
        row = {"key": key, "rc": None, "reproduced": False, "error": repr(exc)[:300]}
    finally:
        subprocess.run(["git", "checkout", "-q", "--", path], cwd=FORK)
    with OUT.open("a") as fh: fh.write(json.dumps(row) + "\n")
    print(f"[{i}/{len(todo)}] {key}: rc={row['rc']} ({row.get('seconds')}s)", flush=True)
print("ALL DONE", flush=True)
