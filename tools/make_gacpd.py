"""Generate a GACPD-format run directory from a real pull request.

Reproduces the layout SALP's ingester consumes (post GACPD-0.16, JSON records):

    <out>/<mainline>-<divergent>/<PR>_MO/
        pr_results.json
        MO/<flattened path>/
            results.json
            cmp/<File>.java              # divergent-repo file at the cutoff commit
            patch/<File>.patch           # whole-file unified diff of the PR's effect
            src/hunk_<n>_full_del.java   # @@ header + pre-change region
            src/hunk_<n>_full_add.java   # @@ header + post-change region
            src/hunk_<n>_context.java    # context lines only
            src/hunk_<n>_additions.java  # added lines only
            src/hunk_<n>_deletions.java  # deleted lines only

Reads only local bare clones (SALP's cache layout: <cache>/<owner>__<repo>.git),
so run it after the clones exist. Similarity percentages are fabricated
placeholder metadata (GACPD's clone-detector scores are not reproduced here)
and are marked as such in results.json via "generatedBy".

Usage:
  python tools/make_gacpd.py --cache SALP/data/repos --out SALP/data/gacpd \
      --mainline apache/kafka --divergent linkedin/kafka --pr 12535 \
      --cutoff 2026-06-30T16:37:26Z --divergence 2021-07-06T21:17:16Z
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

HUNK_HEADER = re.compile(r"^@@\s+-\d+(?:,\d+)?\s+\+\d+(?:,\d+)?\s+@@.*$")


def git(clone: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "--git-dir", str(clone), *args],
        capture_output=True, text=True, check=True)
    return result.stdout


def git_ok(clone: Path, *args: str) -> bool:
    return subprocess.run(["git", "--git-dir", str(clone), *args],
                          capture_output=True, check=False).returncode == 0


def split_hunks(patch: str) -> list[list[str]]:
    """The per-hunk line blocks of a unified diff (header line first in each)."""
    hunks: list[list[str]] = []
    for line in patch.splitlines():
        if HUNK_HEADER.match(line):
            hunks.append([line])
        elif hunks and line[:1] in (" ", "+", "-", "\\"):
            hunks[-1].append(line)
    return hunks


def side(hunk: list[str], keep: str) -> str:
    """Header + one side's plain lines ('-' side or '+' side, plus context)."""
    out = [hunk[0]]
    for line in hunk[1:]:
        if line[:1] == " " or line[:1] == keep:
            out.append(line[1:])
    return "\n".join(out) + "\n"


def only(hunk: list[str], kind: str, *, prefix_stripped: bool = True) -> str:
    lines = [line[1:] if prefix_stripped else line
             for line in hunk[1:] if line[:1] == kind]
    return "\n".join(lines) + ("\n" if lines else "")


def similarity_block(n: int) -> list[dict]:
    return [
        {"checkName": f"hunk_{n}_{mode}.java", "similarityPercent": 0, "tokenSize": ts}
        for ts in (50, 40) for mode in ("additions", "deletions")
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--mainline", required=True)
    ap.add_argument("--divergent", required=True)
    ap.add_argument("--pr", required=True)
    ap.add_argument("--cutoff", required=True)
    ap.add_argument("--divergence", required=True)
    ap.add_argument("--max-files", type=int, default=4)
    args = ap.parse_args()

    src_clone = args.cache / (args.mainline.replace("/", "__") + ".git")
    tgt_clone = args.cache / (args.divergent.replace("/", "__") + ".git")
    for clone in (src_clone, tgt_clone):
        if not clone.is_dir():
            raise SystemExit(f"missing bare clone: {clone}")

    pr_ref = f"refs/pull/{args.pr}/head"
    if not git_ok(src_clone, "rev-parse", "--verify", pr_ref):
        git(src_clone, "fetch", "origin", f"+{pr_ref}:{pr_ref}")
    head = git(src_clone, "rev-parse", pr_ref).strip()
    base = git(src_clone, "merge-base", head, "HEAD").strip()
    target_commit = git(tgt_clone, "rev-list", "-1", f"--before={args.cutoff}",
                        "HEAD").strip()
    print(f"PR head {head[:12]}  base {base[:12]}  target@cutoff {target_commit[:12]}")

    changed = [
        line.split("\t", 1) for line in
        git(src_clone, "diff", "--name-status", base, head).splitlines()
    ]
    java_modified = [p for s, p in changed if s == "M" and p.endswith(".java")]
    print(f"{len(java_modified)} modified .java files in the PR")

    pair = f"{args.mainline.replace('/', '_')}-{args.divergent.replace('/', '_')}"
    pr_dir = args.out / pair / f"{args.pr}_MO"

    pr_dir.mkdir(parents=True, exist_ok=True)
    (pr_dir / "pr_results.json").write_text(json.dumps({
        "classifiedPR": args.pr,
        "prTitle": f"PR {args.pr} (generated from local clone)",
        "prLocation": f"https://github.com/{args.mainline}/pull/{args.pr}",
        "repoDivergenceDate": args.divergence,
        "cutoffDate": args.cutoff,
        "generatedBy": "rcap tools/make_gacpd.py (not a real GACPD run)",
    }, indent=1), encoding="utf-8")

    minted = 0
    for path in java_modified:
        in_target = git_ok(tgt_clone, "cat-file", "-e", f"{target_commit}:{path}")
        classification = "MO" if in_target else "NA"
        if classification == "MO" and minted >= args.max_files:
            continue
        flattened = path.replace("/", "_").replace(".", "_")
        fdir = pr_dir / classification / flattened
        name = Path(path).name

        patch = git(src_clone, "diff", base, head, "--", path)
        hunks = split_hunks(patch)

        (fdir / "results.json").parent.mkdir(parents=True, exist_ok=True)
        (fdir / "results.json").write_text(json.dumps({
            "fileName": path,
            "classification": classification,
            "divergentPath": (
                f"Results/Repos_files/rcap_gen/{args.divergent}/{path}"
                if in_target else None),
            "divergentPathType": "resolved" if in_target else "unresolved",
            "divergentRepo": args.divergent,
            "mainline": args.mainline,
            "pr": args.pr,
            "similarityChecks": [c for i in range(len(hunks))
                                 for c in similarity_block(i + 1)],
            "generatedBy": "rcap tools/make_gacpd.py; similarity values are placeholders",
        }, indent=1), encoding="utf-8")

        (fdir / "patch").mkdir(exist_ok=True)
        (fdir / "patch" / f"{name}.patch").write_text(patch, encoding="utf-8")

        if in_target:
            (fdir / "cmp").mkdir(exist_ok=True)
            (fdir / "cmp" / name).write_text(
                git(tgt_clone, "show", f"{target_commit}:{path}"), encoding="utf-8")

        src = fdir / "src"
        src.mkdir(exist_ok=True)
        for i, hunk in enumerate(hunks, 1):
            (src / f"hunk_{i}_full_del.java").write_text(side(hunk, "-"), encoding="utf-8")
            (src / f"hunk_{i}_full_add.java").write_text(side(hunk, "+"), encoding="utf-8")
            (src / f"hunk_{i}_context.java").write_text(only(hunk, " "), encoding="utf-8")
            (src / f"hunk_{i}_additions.java").write_text(only(hunk, "+"), encoding="utf-8")
            (src / f"hunk_{i}_deletions.java").write_text(only(hunk, "-"), encoding="utf-8")

        if classification == "MO":
            minted += 1
        print(f"  {classification}  {path}  ({len(hunks)} hunk(s))")

    print(f"wrote {pr_dir}")


if __name__ == "__main__":
    main()
