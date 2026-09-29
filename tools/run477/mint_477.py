#!/usr/bin/env python3
"""Mint GACPD dirs + SAPs for the whole RePatch 477 dataset, all 10 repo pairs.

The earlier RCAP-vs-RePatch comparison covered only the 393 linkedin/kafka
cases. This drives the same path over every pair in the validation dataset.

Per case the dataset gives exact revisions, so the target state is bound to the
commit the RePatch study actually used rather than to a date: the cutoff passed
to make_gacpd is the commit DATE of `target_revision`, and the resolved commit is
then checked against `target_revision` and any mismatch recorded rather than
assumed away. `divergence` is the date of `base_revision` (the merge base);
it is metadata only here, since refactoring detection is off.

Usage: mint_477.py <out-dir> [--only-non-kafka] [--max-files N]
"""
import argparse, json, subprocess, sys, collections
from pathlib import Path

DATA = Path("/home/adam/Documents/work/validation-study/validation-data/dataset.jsonl")
CACHE = Path("/home/adam/Documents/work/SALP/data/repos")
RCAP = Path("/home/adam/Documents/work/rcap-class-project/rcap")
PY_BIN = RCAP / ".venv/bin/python"

def slug(url): return url.replace("https://github.com/", "").strip("/")
def clone_dir(s): return CACHE / (s.replace("/", "__") + ".git")

def git(repo, *a, check=False):
    r = subprocess.run(["git", "--git-dir", str(repo)] + list(a),
                       capture_output=True, text=True)
    if check and r.returncode: raise RuntimeError(r.stderr[:300])
    return r.stdout.strip()

def rc(repo, *a):
    return subprocess.run(["git", "--git-dir", str(repo)] + list(a),
                          capture_output=True, text=True).returncode

MAX_PLAUSIBLE_FILES = 60

def pr_base(clone, head, base_revision=None):
    """(base, how) for the PR's diff, or (None, why) when it cannot be determined.

    Two wrong answers were shipped before this one.

    * merge-base(head, HEAD) COLLAPSES to head for the 64 of 477 cases whose PR
      head has already landed in the mainline: empty diff, empty SAP.
    * The dataset's `base_revision` is NOT the PR's parent -- it is the FORK
      DIVERGENCE POINT, and for 257 of 280 usable apache/kafka cases it is not an
      ancestor of the head at all. `git diff base head` across divergent lines
      also reports, reversed, everything reachable from base but not head, so the
      diff is inflated and largely fabricated: PR 12289 goes from 1 real Java file
      to 196, and PR 12535 mints 24 files it never touches while missing all 4 it
      does.
    * `parents[0]` of a 2-parent head is invalid too: such a head is the
      contributor merging upstream INTO their branch, so diffing against it
      sweeps in everything upstream changed.

    What is correct is the fork point of the PR's own line of development.
    `base_revision` is accepted only so callers may pass it for cross-checking;
    it is never used as the base.
    """
    parents = git(clone, "rev-list", "--parents", "-1", head).split()[1:]
    if rc(clone, "merge-base", "--is-ancestor", head, "HEAD") != 0:
        return git(clone, "merge-base", head, "HEAD"), "merge_base"
    if len(parents) == 1:
        return parents[0], "first_parent"
    return None, "ambiguous_merge_head_already_landed"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--only-non-kafka", action="store_true")
    ap.add_argument("--max-files", type=int, default=100)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(DATA)]
    if a.only_non_kafka:
        rows = [r for r in rows if slug(r["target_repo"]) != "linkedin/kafka"]
    print(f"{len(rows)} case(s)")

    # 1. PR heads, batched per source repo (one network round trip each)
    need = collections.defaultdict(set)
    for r in rows:
        need[slug(r["source_repo"])].add(str(r["pr_number"]))
    for s, prs in sorted(need.items()):
        cd = clone_dir(s)
        if not cd.is_dir():
            print(f"  MISSING CLONE {s} -- skipping its {len(prs)} case(s)"); continue
        missing = [p for p in sorted(prs)
                   if not git(cd, "rev-parse", "--verify", "--quiet", f"refs/pull/{p}/head^{{commit}}")]
        if not missing:
            print(f"  {s}: all {len(prs)} PR head(s) present"); continue
        print(f"  {s}: fetching {len(missing)} PR head(s)...", flush=True)
        for i in range(0, len(missing), 150):
            chunk = missing[i:i+150]
            subprocess.run(["git", "--git-dir", str(cd), "fetch", "-q", "origin"]
                           + [f"+refs/pull/{p}/head:refs/pull/{p}/head" for p in chunk],
                           capture_output=True, text=True)
        still = [p for p in missing
                 if not git(cd, "rev-parse", "--verify", "--quiet", f"refs/pull/{p}/head^{{commit}}")]
        print(f"    got {len(missing)-len(still)}/{len(missing)}"
              + (f"; unavailable: {still[:8]}" if still else ""))

    # 2. make_gacpd per case
    stats = collections.Counter(); mism = []; big = []; per_pair = collections.Counter()
    log = []
    for i, r in enumerate(rows, 1):
        src, tgt, pr = slug(r["source_repo"]), slug(r["target_repo"]), str(r["pr_number"])
        sc, tc = clone_dir(src), clone_dir(tgt)
        if not (sc.is_dir() and tc.is_dir()):
            stats["no_clone"] += 1; continue
        if not git(sc, "rev-parse", "--verify", "--quiet", f"refs/pull/{pr}/head^{{commit}}"):
            stats["no_pr_head"] += 1; continue
        trev = r.get("target_revision"); brev = r.get("base_revision")
        cutoff = git(tc, "show", "-s", "--format=%cI", trev) if trev else ""
        if not cutoff:
            stats["no_target_rev_in_clone"] += 1; continue
        divergence = (git(sc, "show", "-s", "--format=%cI", brev) if brev else "") or cutoff
        # does that date actually resolve back to the recorded revision?
        got = git(tc, "rev-list", "-1", f"--before={cutoff}", "HEAD")
        if trev and got and not trev.startswith(got[:12]) and not got.startswith(trev[:12]):
            mism.append((r["case_id"], trev[:12], got[:12]))
        head = git(sc, "rev-parse", f"refs/pull/{pr}/head^{{commit}}")
        base, how = pr_base(sc, head)
        # cross-check only: does the dataset's own base_revision match ours?
        ds_base = r.get("base_revision")
        ds_agrees = bool(ds_base) and base is not None and ds_base.startswith(base[:12])
        base_rec = {"case_id": r["case_id"], "pair": f"{src} -> {tgt}", "pr": pr,
                    "patch_type": r.get("patch_type"), "base_how": how,
                    "head": head[:12], "base": (base or "")[:12],
                    "dataset_base_agrees": ds_agrees}
        if not base:
            stats[f"no_base:{how}"] += 1
            log.append({**base_rec, "outcome": f"no_base:{how}"}); continue
        stats[f"base_via_{how}"] += 1
        n_par = len([l for l in git(sc, "diff", "--name-only", base, head).splitlines()
                     if l.rsplit(".", 1)[-1].lower() in {"java", "scala", "sc"}])
        # No base is authoritative, so the guard applies to all of them: a diff
        # this large is far more likely to be a bad base than a real PR.
        if n_par > MAX_PLAUSIBLE_FILES:
            stats["skipped_implausible_diff"] += 1
            big.append((r["case_id"], how, n_par))
            log.append({**base_rec, "outcome": "skipped_implausible_diff",
                        "parseable_files": n_par}); continue
        p = subprocess.run([str(PY_BIN), str(RCAP/"tools/make_gacpd.py"),
                            "--cache", str(CACHE), "--out", a.out,
                            "--mainline", src, "--divergent", tgt, "--pr", pr,
                            "--cutoff", cutoff, "--divergence", divergence,
                            "--base", base,
                            "--max-files", str(a.max_files)],
                           capture_output=True, text=True)
        if p.returncode:
            log.append({**base_rec, "outcome": "make_gacpd_failed",
                        "err": p.stderr.strip()[-200:]})
            stats["make_gacpd_failed"] += 1
            if stats["make_gacpd_failed"] <= 3:
                print(f"  FAIL {r['case_id']}: {p.stderr.strip()[-200:]}")
            continue
        n_mo = p.stdout.count("  MO  ")
        stats["ok"] += 1; stats["mo_files"] += n_mo
        if n_mo == 0: stats["ok_but_no_mo"] += 1
        log.append({**base_rec, "outcome": "ok", "mo_files": n_mo,
                    "parseable_files": n_par})
        per_pair[f"{src} -> {tgt}"] += n_mo
        if i % 50 == 0: print(f"  [{i}/{len(rows)}] {dict(stats)}", flush=True)

    print(f"\n{dict(stats)}")
    print(f"\ncutoff-date/target_revision mismatches: {len(mism)}")
    for m in mism[:10]: print(f"  {m[0]}: recorded {m[1]} vs resolved {m[2]}")
    print(f"\nskipped for an implausible diff (>{MAX_PLAUSIBLE_FILES} parseable files): {len(big)}")
    for c, how, n in sorted(big, key=lambda x: -x[2]):
        print(f"  {c}: {n} files (base via {how})")
    print("\nMO files per pair:")
    for k, v in per_pair.most_common(): print(f"  {v:>5}  {k}")
    out_log = Path(a.out).parent / "mint_477_log.jsonl"
    with out_log.open("w") as fh:
        for rec in log: fh.write(json.dumps(rec) + "\n")
    print(f"\nper-case log -> {out_log}")

def _unused():
    pass

if __name__ == "__main__":
    main()
