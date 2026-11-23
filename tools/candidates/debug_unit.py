"""Print the references and resolutions of one unit (calibration aid).

    debug_unit.py <unit id pair/PR/sap::fn> <order_pr ordered/ dir, or - to read the candidate on stdin> <sap root>
Repo cache: $SYMCHECK_REPO_CACHE (default ~/Documents/work/SALP/data/repos); indexes in ~/.cache/rcap-symindex.
"""
import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent)); sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
import symbol_check as sc
uid, ordered, root = sys.argv[1:4]
pair, pr, rest = uid.split("/", 2); sap, fn = rest.split("::")
c = sc.Contexts(os.environ.get("SYMCHECK_REPO_CACHE", os.path.expanduser("~/Documents/work/SALP/data/repos")), str(Path.home() / ".cache/rcap-symindex"), str(sc.DEFAULT_PR_BASES))
pdir = Path(root) / pair / pr
man = json.loads((pdir / "pr.json").read_text()); sapj = json.loads((pdir / sap / "sap.json").read_text())
if ordered != "-":
    f = Path(ordered) / "attempts" / pr / f"{sap}.{fn}.candidate.json"
    cand = json.loads(f.read_text())["candidate_expanded"]
    target = f.with_name(f.name.replace(".candidate.json", ".target_original.txt")).read_text()
else:
    cand = sys.stdin.read(); target = (pdir / sap / "functions" / fn / "target.java").read_text()
fork = c.fork(man["target_repo"]); prx = c.pr(pdir, man); fv = fork.file_view(sapj["target_file"])
print("target_file", sapj["target_file"], "indexed", fv is not None, "lib", fv and fv["lib"])
for name, role, ar, ao in sc.candidate_refs(cand, target):
    print(f"  {name:40} {role:7} {ar} {'arity-only' if ao else ''} ->", sc.resolve(name, role, ar, fork, c.jdk(), prx, fv, ao))
print(json.dumps(sc.check(cand, target, fork, c.jdk(), prx, sapj["target_file"]))[:600])
