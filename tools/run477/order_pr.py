#!/usr/bin/env python3
"""PR-level validation in dependency order (no re-prompting).

reprompt_pr.py --k 1 applied every candidate of a PR at once and dropped failing units until the
build settled: order-dependent, cascading (a half-applied rename took the whole rename out) and one
bad sibling masked every good unit. Here, per upstream PR:

  1. attempt 1 for every SAP (rcap.pipeline.run_case; --placeholder-retry adds the ONE targeted
     placeholder repair request; answers are replayed by prompt hash where they exist);
  2. change GROUPS = SAP units with a candidate + upstream hunks no unit covers that apply to the
     fork by exact context (MECHANICAL: declarations/fields outside units, message/*.json, ...);
  3. renames of the upstream change become ONE atomic group (declaration + every use). A rename
     that cannot be completed is not attempted: the fork's name is kept everywhere, and a unit
     whose whole change was the rename is NOTHING_TO_PORT, not a failure;
  4. a dependency graph (rcap.prorder): A needs B when A's added lines use a name B introduces
     (hard), or A is the test of B's file (soft). A name whose only provider is unavailable (no
     candidate, hunk does not apply, file absent) makes its users BLOCKED_BY(provider), unbuilt;
  5. groups are applied in topological layers, providers first, one incremental build per layer.
     Compile errors are charged within the layer by what each group CHANGED (rcap.blame): the
     group that wrote the error line, or the group that changed the declaration the error is
     about; anything ambiguous is isolated by building subsets of the layer, and every group
     taken out is tried once more on the kept state. A failed group's hard dependents are
     BLOCKED_BY(group) and never built; independent groups go on. Tests run on the compiled
     final state; charged groups are taken out with their dependents and the rest is re-tested.

All groups of a file are merged as edits against the ORIGINAL file (rcap.splice), so nested and
overlapping units are no longer dropped, and the import post-pass (rcap.imports) runs per file.

  order_pr.py --sample sample.json --salp-out DIR --fork DIR --baseline FILE --repo-cache DIR
              --out-dir DIR --tag ordered [--placeholder-retry] [--no-fix-imports] [--only PR-N]
One JSON row per PR in <out-dir>/<tag>/pr-results.jsonl; resumable by PR.
"""
import argparse, collections, json, re, subprocess, sys, time, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gencache import CachedBackend, load_replay  # noqa: E402
from harness import Harness, lang_of, screen_unit  # noqa: E402
from rcap import blame  # noqa: E402
from rcap import cochange as cc  # noqa: E402
from rcap import prorder as po  # noqa: E402
from rcap.backend_ollama import OllamaBackend  # noqa: E402
from rcap.feedback import (Span, TestFailure, _fallback, attribute_test_failures,  # noqa: E402
                           parse_compile_errors, parse_junit_failures)
from rcap.provenance import run_provenance  # noqa: E402
from rcap.imports import fix_imports, unused_import_fails_build  # noqa: E402
from rcap.pipeline import run_case  # noqa: E402
from rcap.reduction_program import PLACEHOLDER, recover  # noqa: E402
from rcap.reprompt import code_equal, is_echo  # noqa: E402
from rcap.splice import CONFLICT, NOT_APPLIED, UnitEdit, merge_units  # noqa: E402

RUN = Path("/home/adam/Documents/work/rcap-class-project/run477")
ap = argparse.ArgumentParser()
ap.add_argument("--sample", required=True); ap.add_argument("--salp-out", required=True)
ap.add_argument("--old-salp-out", default=str(RUN / "salp-477-out"))
ap.add_argument("--fork", required=True); ap.add_argument("--baseline", required=True)
ap.add_argument("--repo-cache", required=True); ap.add_argument("--source-repo", default="apache__kafka.git")
ap.add_argument("--out-dir", required=True); ap.add_argument("--tag", required=True)
ap.add_argument("--placeholder-retry", action="store_true", help="one targeted retry per placeholder violation (default off)")
ap.add_argument("--no-fix-imports", action="store_true"); ap.add_argument("--no-mech", action="store_true")
ap.add_argument("--gradle-bin-file", default=str(RUN / "validate/gradle_bin.txt"))
ap.add_argument("--old-candidates", default=str(RUN / "validate/candidates"))
ap.add_argument("--old-results", default=str(RUN / "results-477-v2.jsonl"))
ap.add_argument("--gen-cache", required=True); ap.add_argument("--gen-cache-ro", action="append", default=[])
ap.add_argument("--only", action="append"); ap.add_argument("--limit", type=int)
ap.add_argument("--max-bisect", type=int, default=40, help="subset builds per layer for isolating an ambiguous compile error")
ap.add_argument("--max-reinstate", type=int, default=30, help="re-tries per layer of groups taken out for a compile error")
ap.add_argument("--plan-only", action="store_true", help="stop after the dependency graph: no build, no test")
ap.add_argument("--no-regions", action="store_true", help="whole-file units are always asked for the whole file (as before rcap.regions)")
a = ap.parse_args()

H = Harness(a.fork, Path(a.gradle_bin_file).read_text().strip()); FORK = H.fork
OUTD = Path(a.out_dir).resolve() / a.tag; OUT = OUTD / "pr-results.jsonl"
SRC = Path(a.repo_cache).resolve() / a.source_repo
BASE = Path(a.baseline); baseline = json.loads(BASE.read_text()) if BASE.exists() else {}
TEST_TIMEOUT = 2400; BUILD_TIMEOUT = 2400
EXT = {"java": "java", "scala": "scala", "sc": "scala"}
FACTS = H.build_facts()


def sgit(*args):
    r = subprocess.run(["git", "--git-dir", str(SRC), *args], capture_output=True, text=True, errors="replace")
    return r.stdout if r.returncode == 0 else None

def slug(n): return re.sub(r"[^A-Za-z0-9]+", "-", n).strip("-")[:80]

def sap_dirs(pair, pr):
    """reprompt_pr.sap_dirs: SAPs the new mint merged by name collision are read from the old mint."""
    old_root = Path(a.old_salp_out) / pair / pr; new_root = Path(a.salp_out) / pair / pr
    olds = {}
    for d in sorted(old_root.glob("sap-*")):
        try: olds[d.name] = json.loads((d / "sap.json").read_text())["source_file"]
        except (OSError, KeyError, ValueError): pass
    groups = collections.defaultdict(list)
    for name, src in olds.items(): groups["sap-" + slug(src)].append(name)
    out, used = [], set()
    for name, src in sorted(olds.items()):
        nd = new_root / ("sap-" + slug(src))
        if len(groups["sap-" + slug(src)]) > 1: out.append((name, old_root / name, "old:collision"))
        elif nd.is_dir(): out.append((name, nd, "new")); used.add(nd.name)
        else: out.append((name, old_root / name, "old:absent_in_new"))
    for d in sorted(new_root.glob("sap-*")):
        if d.name not in used and d.name not in groups: out.append((d.name, d, "new:no_old_counterpart"))
    return out, old_root, new_root

def pr_base(head):
    parents = (sgit("rev-list", "--parents", "-1", head) or "").split()[1:]
    if subprocess.run(["git", "--git-dir", str(SRC), "merge-base", "--is-ancestor", head, "HEAD"]).returncode != 0:
        return (sgit("merge-base", head, "HEAD") or "").strip() or None, "merge_base"
    if len(parents) == 1: return parents[0], "first_parent"
    return None, "ambiguous_merge_head_already_landed"

def run_tests(proj, classes):
    rdir = FORK / proj.strip(":").replace(":", "/") / "build/test-results/test"
    for f in rdir.glob("*.xml"): f.unlink()
    rc, log, secs = H.gradle(f"{proj}:test", *[x for c in classes for x in ("--tests", c)], timeout=TEST_TIMEOUT)
    ran, failures = {}, []
    for f in sorted(rdir.glob("*.xml")):
        n, fl = parse_junit_failures([str(f)])
        ran[f.name[len("TEST-"):-len(".xml")]] = n; failures += fl
    return ran, failures, rc, log, secs


class U:
    """One SAP unit with a candidate."""
    def __init__(self, **kw): self.__dict__.update(kw)


def run_pr(pair, pr, backend):
    t_pr = time.time(); prn = pr.split("-")[1]
    adir = OUTD / "attempts" / pr; rdir = OUTD / "rounds" / pr
    adir.mkdir(parents=True, exist_ok=True); rdir.mkdir(parents=True, exist_ok=True)
    backend.replay = load_replay(a.old_candidates, pr); backend.sources.clear()
    saps, old_root, new_root = sap_dirs(pair, pr)
    row = {"pr": pr, "pair": pair, "placeholder_retry": a.placeholder_retry, "fix_imports": not a.no_fix_imports,
           "mech": not a.no_mech, "saps": [], "build_facts": FACTS, "provenance": PROV}
    tracked = H.tracked()

    # ---------------------------------------------------------------- the upstream diff
    head = (sgit("rev-parse", f"refs/salp/pr/{prn}^{{commit}}") or "").strip()
    base, how = pr_base(head) if head else (None, "no_pr_ref")
    files = cc.parse_unified_diff((sgit("diff", "-U0", "-M", "--no-color", base, head) if base else None) or "")
    row["upstream"] = {"head": head, "base": base, "base_how": how, "files": len(files), "hunks": sum(len(f.hunks) for f in files)}
    old_path = {f.new_path: f.old_path for f in files if f.new_path}
    src_text = lambda rev, path: sgit("show", f"{rev}:{path}") if rev and path else None  # noqa: E731
    fork_text = {}
    def original(path):
        if path not in fork_text:
            r = subprocess.run(["git", "show", f"HEAD:{path}"], cwd=FORK, capture_output=True, text=True)
            fork_text[path] = r.stdout if r.returncode == 0 else None
        return fork_text[path]

    old_rows = {}
    if Path(a.old_results).exists():
        for l in open(a.old_results):
            if f'/{pr}/' in l[:200]:
                r = json.loads(l); old_rows[r["sap"]] = r

    # ---------------------------------------------------------------- attempt 1 for every SAP
    units, regions, no_cand, upstream_file, generation = {}, [], {}, {}, {}
    t_gen = time.time()
    for name, sdir, origin in saps:
        srow = {"sap": name, "dir": str(sdir), "origin": origin, "units": [], "missing_units": []}
        row["saps"].append(srow)
        try:
            sap = json.loads((sdir / "sap.json").read_text())
            manifest = json.loads(((old_root if origin.startswith("old") else new_root) / "pr.json").read_text())
        except Exception as exc:
            srow["error"] = f"unreadable SAP: {exc!r}"[:300]; continue
        srow["source_file"] = sap["source_file"]; srow["target_file"] = sap["target_file"]
        upstream_file[sap["target_file"]] = sap["source_file"]
        ext = EXT.get(sap["source_file"].rsplit(".", 1)[-1], "java")
        head_src = src_text(head, sap["source_file"])
        base_src = src_text(base, old_path.get(sap["source_file"]) or sap["source_file"])
        fns = sap.get("functions", [])
        for fn in fns:
            fdir = sdir / "functions" / fn
            at = (fdir / f"source.after.{ext}").read_text(errors="replace") if (fdir / f"source.after.{ext}").exists() else ""
            bt = (fdir / f"source.before.{ext}").read_text(errors="replace") if (fdir / f"source.before.{ext}").exists() else ""
            regions.append(cc.UnitRegion(key=f"{name}::{fn}", source_file=sap["source_file"], old_source_file=old_path.get(sap["source_file"]),
                                         after_span=cc.locate(at, head_src) if head_src else None,
                                         before_span=cc.locate(bt, base_src) if base_src else None, after_text=at, before_text=bt))
        o = old_rows.get(name, {})
        backend.expected = list(zip(o.get("unit_outcomes") or [], o.get("prompt_chars") or []))
        try:
            res = run_case(sdir, manifest, backend, placeholder_retry=a.placeholder_retry, region_requests=not a.no_regions)
        except Exception as exc:
            srow["refused"] = {"stage": getattr(exc, "stage", None), "error": f"{type(exc).__name__}: {exc}"[:300]}
            for fn in fns: no_cand[f"{name}::{fn}"] = f"refused:{getattr(exc, 'stage', None)}"
            continue
        finally:
            backend.expected = []
        for i, u in enumerate(res.units):
            ent = u.context.target_localization.get("function"); fn = (ent or "").split("/")[-1]; key = f"{name}::{fn}"
            retry = u.placeholder_retry
            # what was sent, and under which limits: in every unit's record, whatever its outcome
            gen = {"request": ("regions" if u.region_requests else u.request.template_id),
                   "template": (u.region_requests[0].template_id if u.region_requests else u.request.template_id),
                   "prompt_chars": (sum(len(r.prompt) for r in u.region_requests) if u.region_requests else len(u.request.prompt)),
                   "whole_request_chars": len(u.request.prompt), "target_chars": len(u.context.transformation.get("target", "")),
                   "usage": dict(u.generation.usage), "limits": u.generation.limits, "capacity": u.generation.capacity,
                   "regions": u.generation.regions,
                   "region_plan": (u.region_plan.as_dict() if u.region_plan is not None and u.request.answer_kind != "function" else None)}
            generation[key] = gen
            if u.generation.outcome != "completion":
                outcome = u.generation.outcome; diags = list(u.generation.diagnostics)[:2]
                if outcome == "backend_error" and diags and diags[0].startswith("stored-477 outcome: "):
                    outcome = diags[0].split(": ", 1)[1]; diags = ["outcome of the 477 run for the identical prompt; not regenerated"]
                    if outcome == "limits_exceeded": gen["capacity"] = "stored_477_limits_exceeded"
                srow["missing_units"].append({"unit": key, "outcome": outcome, "diagnostics": diags, "placeholder_retry": retry,
                                              "capacity": gen["capacity"], "request": gen["request"]})
                no_cand[key] = outcome; continue
            art = next((x for x in u.context.program_context if x.role == "target" and x.entity == ent), None)
            raw = u.generation.candidate; exp = raw
            for e in (art.placeholders if art else []): exp = exp.replace(PLACEHOLDER.format(n=e.ph_id), e.original_text, 1)
            st = U(key=key, sap=name, file=sap["target_file"], index=i, fn=fn, sdir=str(sdir), raw=raw, expanded=exp, first_expanded=exp,
                   target_original=recover(art) if art else "", is_test="/src/test/" in sap["target_file"],
                   placeholder_retry=retry, status=None, detail={}, flags=[], template=u.generation.request_hash)
            units[key] = st; srow["units"].append(key)
            (adir / f"{name}.{fn}.target_original.txt").write_text(st.target_original)
            (adir / f"{name}.{fn}.prompt.txt").write_text(
                "\n\n".join(f"===== region request {n} of {len(u.region_requests)} =====\n{r.prompt}"
                            for n, r in enumerate(u.region_requests, 1)) if u.region_requests else u.request.prompt)
    for name, sdir, origin in saps:      # functions the pipeline never turned into a unit (null transformation per unit, ...)
        for r in regions:
            if r.key.startswith(name + "::") and r.key not in units and r.key not in no_cand: no_cand[r.key] = "no_unit:not_adaptable"
    row["attempt1_generation"] = {"seconds": round(time.time() - t_gen, 1), "sources": dict(backend.sources)}
    row["generation"] = generation
    region_of = {r.key: r for r in regions}
    cc.classify_hunks(files, regions)

    # ---------------------------------------------------------------- units that are never applied
    for u in units.values():
        r = region_of.get(u.key)
        rej = screen_unit(u.target_original, u.expanded, r.before_text if r else "", r.after_text if r else "", lang_of(u.file))
        if rej:
            # a leaked marker, the fork's own block a second time, or one function for a whole file
            u.status = rej[0].upper(); u.detail["rejected"] = rej[1][:3]
        elif is_echo(u.expanded, u.target_original): u.status = "ECHO"

    # ---------------------------------------------------------------- mechanical hunks (uncovered upstream change groups)
    sap_target = {s["source_file"]: s["target_file"] for s in row["saps"] if "source_file" in s}
    mech, cov, cand, hunk_file = {}, [], collections.defaultdict(list), {}
    for f in files:
        tgt = sap_target.get(f.new_path or "") or (f.new_path if f.new_path in tracked else None) \
            or (f.old_path if f.status == "R" and f.old_path in tracked else None)
        new_file = f.status == "A" and f.new_path not in tracked
        for h in f.hunks:
            hunk_file[h.hunk_id] = (f, tgt)
            c = {"hunk": h.hunk_id, "status": f.status, "coverage": h.coverage, "covered_by": h.covered_by,
                 "target": tgt or (f.new_path if new_file else None), "in_fork": tgt is not None, "summary": cc.summarize_hunk(h)[:160]}
            cov.append(c)
            if h.coverage != "none": continue
            if f.binary or f.status == "D": c["mech"] = "skipped:binary_or_delete"; continue
            target = tgt or (f.new_path if new_file else None)
            if target is None: c["mech"] = "skipped:target_file_absent_in_fork"; continue
            if "/src/" not in target: c["mech"] = "skipped:not_under_a_gradle_source_dir"; continue
            c["mech"] = "candidate"; cand[f.path].append((h, target, new_file))
    for path, items in cand.items():
        f = next(x for x in files if x.path == path)
        old_lines = (src_text(base, f.old_path or f.new_path) or "").splitlines() if f.status != "A" else []
        for h in cc.with_context([i[0] for i in items], old_lines):
            members = h.members or [h.hunk_id]
            target, new_file = items[0][1], items[0][2]
            old_block = "".join(ln[1:] + "\n" for ln in h.lines if ln[:1] in (" ", "-"))
            new_block = "".join(ln[1:] + "\n" for ln in h.lines if ln[:1] in (" ", "+"))
            mid = "mech:" + h.hunk_id
            m = {"id": mid, "hunk": h, "target": target, "new_file": new_file, "members": members, "old": old_block, "new": new_block,
                 "added": [ln[1:] for ln in h.lines if ln[:1] == "+"], "removed": [ln[1:] for ln in h.lines if ln[:1] == "-"],
                 "status": None, "detail": {}}
            # "applies cleanly" = its old block (3 context lines + removed lines) occurs exactly once in the fork file
            if a.no_mech: m["applies"] = False; why = "mechanical hunks disabled"
            elif new_file: m["applies"] = True; why = ""
            else:
                n = (original(target) or "").count(old_block) if old_block else 0
                m["applies"] = n == 1; why = f"context found {n} times in the fork file"
            mech[mid] = m
            for c in cov:
                if c["hunk"] in members:
                    c["mech_id"] = mid; c["mech"] = "applies_cleanly" if m["applies"] else "rejected:" + why
    row["coverage"] = {"hunks": cov, "counts": dict(collections.Counter(
        (c["coverage"] if c["coverage"] != "none" else "none:" + ("in_fork" if c["in_fork"] else "file_absent_in_fork")) for c in cov))}
    group_mech = {g: mid for mid, m in mech.items() for g in m["members"]}

    # ---------------------------------------------------------------- renames: atomic, or the fork's name is kept
    renames, atomic = [], {}
    for r in po.find_renames(files, old_text=lambda p: src_text(base, p)):
        info = {"old": r.old, "new": r.new, "hunks": r.hunks, "members": [], "complete": True, "why": [], "structural": False}
        members = []
        for hid in r.hunks:
            f, tgt = hunk_file[hid]
            h = next(x for x in f.hunks if x.hunk_id == hid)
            if tgt is None: continue                      # the file is not in the fork: nothing to rename there
            if h.covered_by:
                for k in h.covered_by:
                    if k in units and units[k].status is None and re.search(rf"(?<![\w$]){re.escape(r.new)}(?![\w$])", units[k].expanded):
                        members.append(k)
                    else:
                        info["complete"] = False
                        info["why"].append(f"unit {k} does not perform it ({no_cand.get(k) or (units[k].status if k in units else 'no unit') or 'candidate keeps the old name'})")
                        info.setdefault("culprits", []).append(k)
                if h.coverage != "full": info["complete"] = False; info["structural"] = True; info["why"].append(f"hunk {hid} only partly inside a unit")
            elif group_mech.get(hid) and mech[group_mech[hid]]["applies"]: members.append(group_mech[hid])
            else:
                info["complete"] = False; info["structural"] = True
                info["why"].append(f"use in hunk {hid} lies outside every unit and the hunk does not apply to the fork")
        if info["complete"]:
            # uses of the old name in fork files the upstream rename does not touch cannot be renamed by anyone
            touched_files = {(units[m].file if m in units else mech[m]["target"]) for m in members}
            elsewhere = [p.split(":", 1)[-1] for p in H.grep("-l", "-w", "-F", "-e", r.old, "HEAD", "--", "*.java", "*.scala")]
            elsewhere = [p for p in elsewhere if p not in touched_files]
            if elsewhere:
                info["complete"] = False; info["structural"] = True
                info["why"].append(f"`{r.old}` is also used in {len(elsewhere)} fork file(s) the rename does not reach, e.g. {elsewhere[0]}")
        info["members"] = sorted(set(members)); renames.append((r, info))
        if info["complete"] and len(info["members"]) > 1:
            for m in info["members"]: atomic.setdefault(m, []).append(r.label)

    def keep_fork_names(r, info, why):
        """The rename is not attempted: put the fork's name back in every member."""
        for mid in info["members"]:
            if mid in units:
                u = units[mid]; text, done = po.revert_rename(u.expanded, u.target_original, [r])
                if not done: continue
                u.expanded = text; u.flags.append(f"rename_reverted:{r.old}->{r.new}")
                if is_echo(text, u.target_original) or code_equal(text, u.target_original):
                    culprit = next(iter(info.get("culprits", [])), None)
                    if info["structural"] or culprit is None: u.status = "NOTHING_TO_PORT"
                    else: u.status = "BLOCKED_BY"; u.detail["blocked_by"] = culprit
                    u.detail.update(rename=r.label, why=why)
            else:
                m = mech[mid]; new = po.swap_word(m["new"], r.new, r.old)
                if new != m["new"]:
                    m["new"] = new; m["added"] = [po.swap_word(x, r.new, r.old) for x in m["added"]]
                    m["detail"]["rename_reverted"] = r.label
                    if m["new"] == m["old"]: m["applies"] = False; m["status"] = "NOTHING_TO_PORT"; m["detail"]["why"] = why
    for r, info in renames:
        if not info["complete"]:
            # also units outside the member list that wrote the new name on their own
            for k, u in units.items():
                if u.status is None and k not in info["members"]: info["members"].append(k)
            keep_fork_names(r, info, "; ".join(info["why"])[:400])
    row["renames"] = [{**info, "label": r.label} for r, info in renames]

    # ---------------------------------------------------------------- groups and the dependency graph
    def unit_group(u):
        added = po.added_lines(u.target_original, u.expanded)
        removed = po.added_lines(u.expanded, u.target_original)
        return po.Group(u.key, "unit", u.file, provides=po.introduced(added, removed, original(u.file)),
                        uses=po.identifiers(added), is_test=u.is_test)
    def mech_group(m):
        return po.Group(m["id"], "mech", m["target"], provides=po.introduced(m["added"], m["removed"], None if m["new_file"] else original(m["target"])),
                        uses=po.identifiers(m["added"]), is_test="/src/test/" in m["target"])
    groups = [unit_group(u) for u in units.values() if u.status is None] + [mech_group(m) for m in mech.values() if m["applies"]]
    # merge the members of a completed rename into one atomic group
    by_id = {g.gid: g for g in groups}; merged_into = {}
    for r, info in renames:
        ms = [m for m in info["members"] if m in by_id]
        if not info["complete"] or len(ms) < 2: continue
        roots = sorted({merged_into.get(m, m) for m in ms}); root = by_id[roots[0]]
        for other in roots[1:]:
            o = by_id[other]
            root.provides |= o.provides; root.uses |= o.uses; root.members = sorted(set(root.ids) | set(o.ids))
            for m in o.ids: merged_into[m] = root.gid
            groups.remove(o)
        root.members = sorted(set(root.ids)); root.atomic = "; ".join(filter(None, [root.atomic, r.label]))
        for m in root.members: merged_into[m] = root.gid
    gid_of = lambda m: merged_into.get(m, m)  # noqa: E731
    word_cache = {}
    def in_fork(name):
        if name not in word_cache: word_cache[name] = H.has_word(name)
        return word_cache[name]
    graph = po.build_graph(groups, in_fork=in_fork)
    order = po.layers(groups, graph)
    # names only an UNAVAILABLE provider introduces
    unavailable = {}
    for f in files:
        if f.binary: continue
        for h in f.hunks:
            mid = group_mech.get(h.hunk_id)
            if mid and mech[mid]["applies"]: continue
            f_, tgt = hunk_file[h.hunk_id]
            owners = [k for k in h.covered_by]
            if owners and all(k in units and units[k].status is None for k in owners): continue
            if owners: who = f"unit {owners[0]} ({no_cand.get(owners[0]) or (units[owners[0]].status if owners[0] in units else '?')})"
            elif tgt is None and f.status != "A": who = f"hunk {h.hunk_id} (file absent from the fork)"
            elif mid: who = f"hunk {h.hunk_id} ({mech[mid]['status'] or 'does not apply to the fork'})"
            else: who = f"hunk {h.hunk_id} (not applicable mechanically)"
            for n in po.introduced([t for _, t in h.added], [t for _, t in h.removed], None):
                unavailable.setdefault(n, who)
    provided_now = set().union(*[g.provides for g in groups]) if groups else set()
    row["graph"] = {"groups": len(groups), "layers": [len(x) for x in order],
                    "hard": {g: d for g, d in graph.hard.items()}, "soft": graph.soft,
                    "atomic": {g.gid: {"members": g.members, "why": g.atomic} for g in groups if g.members}}

    gstate = {g.gid: {"status": None, "detail": {}} for g in groups}
    def set_status(gid, status, **detail):
        gstate[gid]["status"] = status; gstate[gid]["detail"].update(detail)
    for g in groups:
        miss = sorted(n for n in g.uses if n in unavailable and n not in provided_now and n not in g.provides and not in_fork(n))
        if miss: set_status(g.gid, "BLOCKED_BY", blocked_by=unavailable[miss[0]], names=miss[:6], provider_available=False)

    if a.plan_only:
        row["plan"] = {"layers": order, "blocked_upfront": {g: s["detail"] for g, s in gstate.items() if s["status"]},
                       "unit_status": {k: u.status for k, u in units.items()}, "no_candidate": no_cand,
                       "mech": {m: (x["applies"], x["status"]) for m, x in mech.items()},
                       "provides": {g.gid: sorted(g.provides) for g in groups if g.provides}}
        return row

    # ---------------------------------------------------------------- applying a state
    touched = sorted({g.file for g in groups})
    def restore():
        tr = [p for p in touched if p in tracked]
        if tr: subprocess.run(["git", "checkout", "-q", "--", *tr], cwd=FORK)
        for p in touched:
            if p not in tracked and (FORK / p).exists(): (FORK / p).unlink()

    _decl = {}
    def decl_names(m):
        """Names a change declares OR un-declares on the lines it adds and removes: an error elsewhere
        that cannot resolve or apply one of them is charged to this change (rcap.blame)."""
        if m not in _decl:
            if m in units:
                u = units[m]
                # the lines it adds and removes, nothing else: a unit whose target is a whole file does not
                # "declare" the class by containing it, and an unchanged signature breaks no caller
                lines = po.added_lines(u.target_original, u.expanded) + po.added_lines(u.expanded, u.target_original)
            else: lines = mech[m]["added"] + mech[m]["removed"]
            _decl[m] = tuple(dict.fromkeys(cc.declared_names(lines)))
        return _decl[m]

    merge_log = {}
    rank = {g.gid: n for n, g in enumerate(groups)}
    layer_of = {gid: n for n, layer in enumerate(order) for gid in layer}
    last_applied = [blame.Applied()]
    def apply_state(gids):
        """Write the state; -> (spans by file, sources, per-member merge record). The state is written in
        one canonical order (layer, then the PR's own group order), whatever order it is asked for."""
        gids = sorted(dict.fromkeys(gids), key=lambda g: (layer_of.get(g, len(order)), rank.get(g, len(rank)), g))
        restore(); ids = [m for gid in gids for m in by_id[gid].ids]
        ap = last_applied[0] = blame.Applied()
        for gid in gids: ap.declares[gid] = frozenset(n for m in by_id[gid].ids for n in decl_names(m))
        byfile = collections.defaultdict(list); spans, sources, rec = collections.defaultdict(list), {}, {}
        for m in ids: byfile[units[m].file if m in units else mech[m]["target"]].append(m)
        for path, ms in byfile.items():
            orig = original(path)
            if orig is None:                              # a file the PR adds: mechanical, whole file
                text = "".join(mech[m]["new"] for m in ms if m in mech)
                (FORK / path).parent.mkdir(parents=True, exist_ok=True); (FORK / path).write_text(text, encoding="utf-8")
                sources[path] = text
                for m in ms: spans[path].append(Span(m, 1, text.count("\n") + 1, "mech", tuple(cc.declared_names(mech[m]["added"]))))
                ap.changed[path] = [(1, text.count("\n") + 1, tuple(sorted({gid_of(m) for m in ms})))]
                ap.spans[path] = [(1, text.count("\n") + 1, gid_of(m)) for m in ms]
                continue
            edits = [UnitEdit(m, units[m].target_original, units[m].expanded) if m in units
                     else UnitEdit(m, mech[m]["old"], mech[m]["new"], priority=-1) for m in ms]
            mr = merge_units(orig, edits, language=lang_of(path)); text = mr.text; shift_at = []
            for m in ms: rec[m] = mr.units[m].as_dict()
            if mr.violations:                             # never build a file the merge cannot account for
                for m in ms: rec[m] = {**rec[m], "status": "unspliceable", "reason": "merge invariant violated: " + "; ".join(mr.violations)[:200]}
            lang = EXT.get(path.rsplit(".", 1)[-1])
            if lang and not a.no_fix_imports and mr.changed and any(m in units for m in ms):
                up = src_text(head, upstream_file.get(path))
                if up:
                    fx = fix_imports(orig, text, up, lang, exists=lambda imp, p=path: H.import_exists(imp, p),
                                     remove_unused=unused_import_fails_build(lang, checkstyle=FACTS["checkstyle"], scala_fatal_warnings=FACTS["scala_fatal_warnings"]))
                    if fx.changed:
                        text = fx.text; shift_at = sorted(x["line"] for x in fx.added)
                        merge_log.setdefault("imports", {})[path] = fx.as_dict()
            sh = lambda n: n + sum(1 for x in shift_at if x <= n)  # noqa: E731
            (FORK / path).write_text(text, encoding="utf-8"); sources[path] = text
            names = {m: decl_names(m) for m in ms}
            for first, last, owners in mr.hunk_lines():   # the changed lines themselves, most specific owner first
                for o in owners[:1]: spans[path].append(Span(o, sh(first), sh(last), "unit" if o in units else "mech", names[o]))
                ap.changed.setdefault(path, []).append((sh(first), sh(last), tuple(sorted({gid_of(o) for o in owners}))))
            for m in sorted((m for m in ms if mr.units[m].new_span), key=lambda m: mr.units[m].new_span[1] - mr.units[m].new_span[0]):
                first, last = mr.lines_of(m); spans[path].append(Span(m, sh(first), sh(last), "unit" if m in units else "mech", names[m]))
                ap.spans.setdefault(path, []).append((sh(first), sh(last), gid_of(m)))
        ap.sources = sources
        return dict(spans), sources, rec

    builds = []
    def build(tag, gids):
        """Write and compile one state -> rcap.blame.BuildResult (errors carry repo-relative paths)."""
        spans, sources, rec = apply_state(gids); ap = last_applied[0]
        bad_merge = {}
        for m, r in rec.items():
            merge_log[m] = r
            if r["status"] in NOT_APPLIED or (r["status"] == CONFLICT and r["applied"] == 0):
                bad_merge[gid_of(m)] = f"{r['status']}: {r['reason']}"
        if bad_merge:
            builds.append({"tag": tag, "groups": len(gids), "rc": None, "seconds": 0, "unwritable": sorted(bad_merge)})
            return blame.BuildResult(None, applied=ap, unwritable=bad_merge)
        projs = sorted({H.project_of(p) for p in spans if H.project_of(p)})
        rc, log, secs = H.gradle(*[f"{p}:testClasses" for p in projs], timeout=BUILD_TIMEOUT)
        (rdir / f"{tag}.build.log").write_text(log)
        builds.append({"tag": tag, "groups": len(gids), "rc": rc, "seconds": secs, "projects": projs})
        print(f"    {pr} {tag}: {len(gids)} group(s) -> rc={rc} ({secs}s)", flush=True)
        if rc == 0: return blame.BuildResult(True, applied=ap)
        if rc == "timeout": return blame.BuildResult(False, applied=ap, note="build timeout")
        errors = parse_compile_errors(log); prefix = str(FORK).rstrip("/") + "/"
        for e in errors: e.rel = e.path[len(prefix):] if e.path.startswith(prefix) else e.path
        note = "" if errors else "build failed with no file:line compiler error: " + _fallback(log, 1500)[:400]
        builds[-1].update(errors=len(errors))
        return blame.BuildResult(False, errors=errors, applied=ap, note=note)

    attribution = []
    def settle(tag, base, layer_groups, li=None, prefix=""):
        """rcap.blame.settle_layer on top of `base`; statuses are set here. -> kept groups (sorted)."""
        out = blame.settle_layer(base, layer_groups, build, hard_dependents=graph.hard_dependents, tag=tag,
                                 max_isolation_builds=a.max_bisect, max_reinstate_builds=a.max_reinstate)
        attribution.append({"tag": tag, "groups": sorted(layer_groups), "kept": out.kept, "builds": out.builds,
                            "failed": out.failed, "evidence": out.evidence})
        for g, f in sorted(out.failed.items(), key=lambda kv: kv[1]["status"] == "BLOCKED_BY"):
            if g in accepted: accepted.remove(g)
            if f["status"] == "BLOCKED_BY":
                if gstate[g]["status"] is None:
                    set_status(g, "BLOCKED_BY", blocked_by=f["blocked_by"], names=graph.hard.get(g, {}).get(f["blocked_by"], []), provider_available=True)
                continue
            detail = {k: v for k, v in f.items() if k != "status"}
            detail["error"] = (prefix + str(detail.get("error", "")))[:300]
            if li is not None: detail["layer"] = li
            if str(f.get("how", "")).startswith("unattributed"): detail["unattributed"] = True
            fail(g, f["status"], **detail)
        return out.kept

    def fail(gid, status, **detail):
        """Mark a group and everything that cannot compile without it."""
        set_status(gid, status, **detail)
        for d in graph.hard_dependents(gid):
            if gstate[d]["status"] is None or d in accepted:
                if d in accepted: accepted.remove(d)
                set_status(d, "BLOCKED_BY", blocked_by=gid, names=graph.hard.get(d, {}).get(gid, []), provider_available=True)

    # ---------------------------------------------------------------- layers
    accepted = []
    try:
        for li, layer in enumerate(order, 1):
            for gid in layer:                             # hard dependents of anything that is not in
                if gstate[gid]["status"] is None:
                    lost = [p for p in graph.hard.get(gid, {}) if p not in accepted and p not in layer]
                    if lost: set_status(gid, "BLOCKED_BY", blocked_by=lost[0], names=graph.hard[gid][lost[0]], provider_available=True)
            live = [g for g in layer if gstate[g]["status"] is None]
            if live: accepted += settle(f"layer-{li:02d}", list(accepted), live, li=li)
            # a completed rename that failed as a whole: keep the fork's names and try its members once, as plain groups
            for gid in layer:
                g = by_id[gid]
                if g.atomic and gstate[gid]["status"] in ("BUILD_FAIL", "INTEGRATION_FAIL") and not gstate[gid]["detail"].get("retried"):
                    gstate[gid]["detail"]["retried"] = True
                    for r, info in renames:
                        if r.label in (g.atomic or ""):
                            info["complete"] = False; info["structural"] = True; info["why"].append("the completed rename did not build in the fork")
                            keep_fork_names(r, info, "the completed rename did not build in the fork: " + str(gstate[gid]["detail"].get("error"))[:200])
                    members = [m for m in g.ids if (m in units and units[m].status is None) or (m in mech and mech[m]["applies"])]
                    solo = []
                    for m in members:
                        ng = unit_group(units[m]) if m in units else mech_group(mech[m])
                        ng.gid = m + "#reverted"; ng.members = [m]; by_id[ng.gid] = ng; merged_into[m] = ng.gid
                        gstate[ng.gid] = {"status": None, "detail": {"rename_reverted": g.atomic}}; solo.append(ng.gid)
                    if solo:
                        for s_ in solo: layer_of[s_] = li - 1
                        accepted += settle(f"layer-{li:02d}.reverted", list(accepted), solo, li=li)

        # ---------------------------------------------------------------- tests on the compiled final state
        test_rounds, final_tests, unattr_tests, timed_out_classes, hit = 0, {}, [], set(), {}
        while accepted and test_rounds < 6:
            test_rounds += 1
            spans, sources, _ = apply_state(accepted)
            tests = collections.defaultdict(set); test_file_of, mapped_files, classes_of = {}, collections.defaultdict(list), {}
            for p in sorted(spans):
                tproj, tcls, thow = H.find_test(p); tcls = [c for c in tcls if c]; classes_of[p] = (tproj, tcls, thow)
                for c in tcls:
                    tests[tproj].add(c)
                    if thow == "self": test_file_of[c] = p
                    else: mapped_files[c].append(p)
            need = [(tp, c) for tp, cs in tests.items() for c in sorted(cs) if f"{tp}|{c}" not in baseline]
            if need:
                restore()
                for tp, c in need:
                    tf = test_file_of.get(c)
                    if tf and tf not in tracked: baseline[f"{tp}|{c}"] = {"rc": 0, "ran": 0, "failed": [], "seconds": 0, "new_class": True}
                    else:
                        ran, fl, brc, blog, bsecs = run_tests(tp, [c])
                        baseline[f"{tp}|{c}"] = {"rc": brc, "ran": sum(ran.values()), "failed": sorted(f.name for f in fl), "seconds": bsecs}
                        (rdir / f"baseline.{c}.test.log").write_text(blog)
                    BASE.write_text(json.dumps(baseline, indent=1))
                spans, sources, _ = apply_state(accepted)
            new, unusable, tsum = [], [], {}
            for tproj, tcls in sorted(tests.items()):
                usable = []
                for c in sorted(tcls):
                    b = baseline[f"{tproj}|{c}"]
                    (unusable if b["rc"] == "timeout" or (b["ran"] == 0 and not b.get("new_class")) else usable).append(c)
                if not usable: continue
                ran, failures, trc, tlog, tsecs = run_tests(tproj, usable)
                (rdir / f"tests-{test_rounds:02d}.{tproj.strip(':').replace(':', '-')}.log").write_text(tlog)
                if trc == "timeout": timed_out_classes |= set(usable)
                bfailed = {f for c in usable for f in baseline[f"{tproj}|{c}"]["failed"]}
                mine = [f for f in failures if f.name not in bfailed and any(f.name.startswith(c + ".") for c in usable)]
                if trc != "timeout":
                    for c in usable:
                        if ran.get(c, 0) == 0 and not any(f.name.startswith(c + ".") for f in mine):
                            mine.append(TestFailure(name=f"{c}.<class did not run>", kind="NoTestsRan",
                                                    message="the test class produced no result with these changes applied", frames=[]))
                new += mine
                tsum[tproj] = {"classes": usable, "rc": trc, "ran": ran, "seconds": tsecs, "new_failures": sorted(f.name for f in mine)}
                print(f"    {pr} tests-{test_rounds:02d} {tproj}: {len(usable)} class(es), {len(mine)} new failure(s) ({tsecs}s)", flush=True)
            final_tests = {"tests": tsum, "unusable": unusable, "classes_of": {p: list(v) for p, v in classes_of.items()}}
            charged = attribute_test_failures(new, spans, test_file_of_class=test_file_of, mapped_files=dict(mapped_files), sources=sources)
            hit = {}
            for key, rows in charged.items():
                hit.setdefault(gid_of(key), f"{len(rows)} new failing test(s); first: {rows[0][0].name}: {rows[0][0].message[:160]} ({rows[0][1]})")
            charged_ids = {id(f) for rows in charged.values() for f, _ in rows}
            unattr_tests = [f"{f.name}: {f.message[:160]}" for f in new if id(f) not in charged_ids]
            hit = {g: t for g, t in hit.items() if g in accepted}
            if not hit: break
            for g, t in hit.items():
                accepted.remove(g); fail(g, "TEST_FAIL", error=t[:300])
            if accepted:                                   # what is left must still compile together
                kept = settle(f"after-tests-{test_rounds:02d}", [], list(accepted), prefix="after a sibling was taken out: ")
                accepted[:] = [g for g in accepted if g in kept]
        exhausted = test_rounds >= 6 and bool(hit)
        row["tests"] = {"rounds": test_rounds, "final": final_tests, "unattributed": unattr_tests[:20], "exhausted": exhausted}
        merge_log.pop("imports", None)
        if accepted: apply_state(accepted)                # records (merge, imports) of the FINAL state
        for g in accepted:
            files_ = {(units[m].file if m in units else mech[m]["target"]) for m in by_id[g].ids}
            cls = [c for p in files_ for c in final_tests.get("classes_of", {}).get(p, (None, [], ""))[1]]
            usable = [c for c in cls if c not in final_tests.get("unusable", [])]
            if exhausted: set_status(g, "INCONCLUSIVE", reason="test rounds exhausted; final state not re-tested")
            elif unattr_tests: set_status(g, "TEST_FAIL_UNATTRIBUTED", error=unattr_tests[0][:300])
            elif any(c in timed_out_classes for c in usable): set_status(g, "INCONCLUSIVE", reason="test timeout")
            elif not usable: set_status(g, "INCONCLUSIVE", reason="build ok; no usable test class")
            else: set_status(g, "VALID_PR", test_classes=usable)
    finally:
        restore()

    # ---------------------------------------------------------------- per-unit and per-hunk outcome
    out_units = {}
    for key, u in units.items():
        gid = gid_of(key); st = gstate.get(gid, {"status": u.status, "detail": u.detail})
        status = u.status or st["status"]; detail = {**u.detail, **(st["detail"] if u.status is None else {})}
        g = by_id.get(gid)
        soft_missing = [p for p in graph.soft.get(gid, []) if p not in accepted]
        prod_missing = sorted({k for k, why in no_cand.items() if po.is_test_of(u.file, sap_target.get(region_of[k].source_file, "") or "")}
                              | {k for k, o in units.items() if o.status is not None and po.is_test_of(u.file, o.file)}) if u.is_test else []
        if status in ("BUILD_FAIL", "TEST_FAIL") and (soft_missing or prod_missing) and u.is_test:
            detail["model_status"] = status; detail["blocked_by"] = (soft_missing or prod_missing)[0]; detail["soft"] = True
            status = "BLOCKED_BY"
        provs = graph.hard_providers(gid) if g else set()
        out_units[key] = {
            "sap": u.sap, "file": u.file, "fn": u.fn, "is_test": u.is_test, "status": status, "detail": detail, "flags": u.flags,
            "in_final_state": gid in accepted, "group": gid, "atomic": g.atomic if g else None,
            "layer": next((n for n, layer in enumerate(order, 1) if gid in layer), None),
            "merge": merge_log.get(key), "placeholder_retry": u.placeholder_retry,
            "depends_on": sorted(provs), "depends_on_mech": sorted(p for p in provs if p.startswith("mech:")),
            "soft_providers_missing": soft_missing + prod_missing,
            "import_fix_in_file": bool(merge_log.get("imports", {}).get(u.file, {}).get("added")) and gid in accepted,
            "candidate_changed_by_driver": u.expanded != u.first_expanded}
        (adir / f"{u.sap}.{u.fn}.candidate.json").write_text(json.dumps(
            {"unit": key, "file": u.file, "candidate_raw": u.raw, "candidate_expanded": u.first_expanded,
             "candidate_applied": u.expanded, "status": status}, indent=1))
    out_mech = {}
    for mid, m in mech.items():
        gid = gid_of(mid); st = gstate.get(gid)
        status = m["status"] or (st["status"] if st else None) or ("NOT_APPLICABLE" if not m["applies"] else None)
        if status == "VALID_PR" or (gid in accepted): status = "MECHANICAL"
        out_mech[mid] = {"target": m["target"], "members": m["members"], "status": status, "in_final_state": gid in accepted,
                         "detail": {**m["detail"], **(st["detail"] if st else {})}, "merge": merge_log.get(mid),
                         "provides": sorted(by_id[gid].provides) if gid in by_id else [],
                         "needed_by": sorted(k for k, v in out_units.items() if mid in v["depends_on_mech"])}
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=FORK, capture_output=True, text=True).stdout.strip()
    row.update(units=out_units, mech_hunks=out_mech, no_candidate=no_cand, builds=builds, attribution=attribution, imports=merge_log.get("imports", {}),
               units_total=len(regions), units_with_candidate=len(units), final_groups=list(accepted),
               fork_clean_after=not dirty, seconds=round(time.time() - t_pr, 1))
    return row


# ------------------------------------------------------------------ main
sample = json.loads(Path(a.sample).read_text())
prs = [(sample["pair"], f"PR-{p}") for p in sample["prs"]]
if a.only: prs = [x for x in prs if x[1] in a.only]
OUTD.mkdir(parents=True, exist_ok=True)
done = {json.loads(l)["pr"] for l in open(OUT)} if OUT.exists() else set()
todo = [x for x in prs if x[1] not in done]
if a.limit: todo = todo[:a.limit]
dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=FORK, capture_output=True, text=True).stdout.strip()
if dirty: sys.exit(f"fork clone is not clean:\n{dirty}")
backend = CachedBackend(OllamaBackend(), a.gen_cache, a.gen_cache_ro,
                        regenerate={"placeholder_violation"} if a.placeholder_retry else ())
PROV = run_provenance(backend, driver="order_pr.py", k=1, placeholder_retry=a.placeholder_retry)
print(f"backend {backend.model} digest={PROV['model_digest']} params={backend.params}; rcap {PROV['rcap_commit']} placeholder_retry={a.placeholder_retry} "
      f"fix_imports={not a.no_fix_imports} mech={not a.no_mech}; {len(todo)} PR(s) to run ({len(done)} done)", flush=True)
for n, (pair, pr) in enumerate(todo, 1):
    try:
        row = run_pr(pair, pr, backend)
    except Exception as exc:
        row = {"pr": pr, "pair": pair, "error": f"{type(exc).__name__}: {exc}"[:400], "traceback": traceback.format_exc()[-2500:]}
        subprocess.run(["git", "checkout", "-q", "--", "."], cwd=FORK)
    with OUT.open("a") as fh: fh.write(json.dumps(row, default=str) + "\n")
    us = row.get("units", {})
    print(f"[{n}/{len(todo)}] {pr}: units {len(us)} of {row.get('units_total')} | " +
          " ".join(f"{k}={v}" for k, v in sorted(collections.Counter(u["status"] for u in us.values()).items())) +
          f" | mech kept {sum(1 for m in row.get('mech_hunks', {}).values() if m['in_final_state'])} | {row.get('seconds')}s {row.get('error', '')}", flush=True)
print("ALL DONE", flush=True)
