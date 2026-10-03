#!/usr/bin/env python3
"""PR-level re-prompt loop (round 2): apply ALL of a PR's candidates, build, test, attribute, re-prompt.

reprompt.py (round 1) splices ONE unit into ONE file and builds; most of its failures needed a
co-change of the same upstream PR that was never applied alongside. Here, per upstream PR:

  1. attempt 1 for EVERY SAP of the PR (rcap.pipeline.run_case, unchanged; stored candidates are
     replayed when the prompt hash matches, see --old-candidates / the generation cache);
  2. every completed unit is spliced into the fork clone together (optionally with the PR's hunks
     that no SAP unit covers, applied mechanically with `git apply`: --mech-hunks, default off);
  3. one Gradle build of the touched projects (--continue), then the mapped test classes against
     the cached per-class baseline of the unmodified fork;
  4. each diagnostic is charged to a (file, unit); charged units are re-prompted (rcap.prloop),
     units that cannot progress are dropped (reverted) and the combined state is validated again.

SAPs that produced no candidate (refused / limits / unparseable) are simply not applied and are
recorded as missing siblings. A deterministic pre-check (rcap.cochange) classifies a unit
NEEDS_COCHANGE, without a model call, when a name it cannot resolve is introduced by the upstream
PR outside the unit and that co-change is not applied.

  reprompt_pr.py --sample sample.json (a SAP list, or {"pair", "prs"}) --salp-out DIR --fork DIR --baseline baseline-tests.json
                 --repo-cache DIR --out-dir DIR --tag k10 [--k 10] [--mech-hunks] [--only PR-N]

One JSON row per PR in <out-dir>/<tag>/pr-results.jsonl; prompts/candidates under
<out-dir>/<tag>/attempts/, build and test logs under <out-dir>/<tag>/rounds/. Resumable by PR.
The fork clone is restored after every validation. Nothing outside --out-dir, the baseline cache
and the fork clone's working tree is written.
"""
import argparse, collections, glob, hashlib, json, os, re, subprocess, sys, time, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from rcap import cochange as cc
from rcap.backend_ollama import OllamaBackend
from rcap.feedback import (Span, TestFailure, _fallback, attribute_errors, attribute_test_failures,
                           member_signatures, missing_symbols, owner_types, parse_compile_errors,
                           parse_junit_failures, rank_members, render_errors_for,
                           render_test_failures)
from rcap.pipeline import _expected_placeholders, run_case
from rcap.prloop import PRUnit, PRValidation, UnitDiagnostic, run_pr_loop
from rcap.reduction_program import PLACEHOLDER, recover

RUN = Path("/home/adam/Documents/work/rcap-class-project/run477")
ap = argparse.ArgumentParser()
ap.add_argument("--sample", required=True); ap.add_argument("--salp-out", required=True)
ap.add_argument("--old-salp-out", default=str(RUN / "salp-477-out"),
                help="fallback SAP dirs for SAPs the new mint lost to name collisions")
ap.add_argument("--fork", required=True); ap.add_argument("--baseline", required=True)
ap.add_argument("--repo-cache", required=True, help="dir holding the source repo mirror(s)")
ap.add_argument("--source-repo", default="apache__kafka.git")
ap.add_argument("--out-dir", required=True); ap.add_argument("--tag", required=True)
ap.add_argument("--k", type=int, default=10)
ap.add_argument("--mech-hunks", action="store_true",
                help="also apply, with plain `git apply`, PR hunks that no SAP unit covers")
ap.add_argument("--gradle-bin-file", default=str(RUN / "validate/gradle_bin.txt"))
ap.add_argument("--old-candidates", default=str(RUN / "validate/candidates"))
ap.add_argument("--old-results", default=str(RUN / "results-477-v2.jsonl"))
ap.add_argument("--gen-cache", help="generation cache dir (default: <out-dir>/gen-cache); share it across rounds")
ap.add_argument("--only", action="append"); ap.add_argument("--limit", type=int)
a = ap.parse_args()

FORK = Path(a.fork).resolve(); OUTD = Path(a.out_dir).resolve() / a.tag; OUT = OUTD / "pr-results.jsonl"
GENCACHE = Path(a.gen_cache).resolve() if a.gen_cache else Path(a.out_dir).resolve() / "gen-cache"; GENCACHE.mkdir(parents=True, exist_ok=True)
SRC = Path(a.repo_cache).resolve() / a.source_repo
GRADLE = Path(a.gradle_bin_file).read_text().strip()
BASE = Path(a.baseline); baseline = json.loads(BASE.read_text()) if BASE.exists() else {}
TEST_TIMEOUT = int(os.environ.get("TEST_TIMEOUT", "2400")); BUILD_TIMEOUT = 2400
EXT = {"java": "java", "scala": "scala", "sc": "scala"}

def sha(t): return hashlib.sha256(t.encode()).hexdigest()

def sh(cmd, timeout, cwd=FORK, stdin=None):
    t = time.time()
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, errors="replace", input=stdin)
        return r.returncode, r.stdout + r.stderr, round(time.time() - t, 1)
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"") + (e.stderr or b"")
        return "timeout", out.decode(errors="replace") if isinstance(out, bytes) else out, round(time.time() - t, 1)

def gradle(*tasks, timeout):
    # identical flags to validate.py / reprompt.py, plus --continue so that every touched project
    # reports its own compile errors instead of stopping at the first failing one
    return sh([GRADLE, "--offline", "-q", "--continue", "-Dorg.gradle.java.home=/usr/lib/jvm/java-11-openjdk-amd64",
               "-PmaxParallelForks=" + os.environ.get("MAX_FORKS", "3"), *tasks, "-x", "checkstyleMain", "-x", "checkstyleTest",
               "-x", "spotbugsMain", "-x", "spotbugsTest", "-x", "spotlessScala"], timeout)

def sgit(*args):
    r = subprocess.run(["git", "--git-dir", str(SRC), *args], capture_output=True, text=True, errors="replace")
    return r.stdout if r.returncode == 0 else None

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
    """validate.py's rule: (project, [fqcn...], how) of the test classes to run for this file."""
    proj = project_of(path)
    if not path.endswith((".java", ".scala")): return proj, [], "not_code"
    if "/src/test/" in path: return proj, concrete(path), "self"
    name = Path(path).stem + "Test"; root = path.split("/src/")[0]
    same = [p for p in glob.glob(f"{FORK}/{root}/src/test/**/{name}.*", recursive=True) if p.endswith((".java", ".scala"))]
    hits = same or [p for p in glob.glob(f"{FORK}/**/src/test/**/{name}.*", recursive=True) if p.endswith((".java", ".scala"))]
    if len(hits) != 1: return proj, [], f"no_unique_test({len(hits)})"
    rel = os.path.relpath(hits[0], FORK)
    return project_of(rel), concrete(rel), "same_project" if same else "other_project"

def run_test(proj, classes):
    """-> ({class: tests ran}, [TestFailure], rc, log, seconds)"""
    rdir = FORK / proj.strip(":").replace(":", "/") / "build/test-results/test"
    for f in glob.glob(f"{rdir}/*.xml"): os.remove(f)
    args = [x for c in classes for x in ("--tests", c)]
    rc, log, secs = gradle(f"{proj}:test", *args, timeout=TEST_TIMEOUT)
    ran, failures = {}, []
    for f in sorted(glob.glob(f"{rdir}/*.xml")):
        n, fl = parse_junit_failures([f])
        ran[Path(f).name[len("TEST-"):-len(".xml")]] = n; failures += fl
    return ran, failures, rc, log, secs

def line_mapper(raw, art):
    """Line of the spliced (placeholder-expanded) candidate -> line of the candidate as the model wrote it."""
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

# ------------------------------------------------------------------ generation: replay + cache
class CachingBackend:
    """The real backend behind a prompt-hash cache. Generation is deterministic (temperature 0,
    fixed seed), so an identical prompt may be answered from (a) the candidates stored by the 477
    run or (b) this experiment's own cache; `sources` counts where each answer came from."""
    def __init__(self, inner, replay):
        self.inner = inner; self.replay = replay; self.sources = collections.Counter()
        for k in ("name", "model", "params", "model_digest", "request_limit_chars", "request_limit_chars_is_explicit"):
            setattr(self, k, getattr(inner, k))
        self.last_usage = None; self.last_source = None; self.expected = []
    def generate(self, request):
        h = sha(request.prompt); f = GENCACHE / f"{h}.json"
        if self.expected:
            # outcomes the 477 run recorded for this SAP's units, in order (no prompt hash was kept
            # for failures, so identity is checked by prompt length)
            outcome, chars = self.expected.pop(0)
            if outcome != "completion" and chars == len(request.prompt) and h not in self.replay and not f.exists():
                self.sources["stored-477-failure"] += 1
                raise RuntimeError("stored-477 outcome: " + outcome)
        if h in self.replay and self.replay[h].get("failed"):
            # The 477 run already got a non-candidate for this very prompt (e.g. a placeholder
            # violation after a multi-minute generation). Deterministic backend: not regenerated.
            self.sources["stored-477-failure"] += 1
            raise RuntimeError("stored-477 outcome: " + self.replay[h]["failed"])
        if h in self.replay:
            raw, usage, src = self.replay[h]["raw"], self.replay[h]["usage"], "stored-477"
        elif f.exists():
            d = json.loads(f.read_text())
            if d.get("error"):
                self.sources["cache-error"] += 1; raise RuntimeError(d["error"])
            raw, usage, src = d["raw"], d.get("usage"), "cache"
        else:
            try: raw = self.inner.generate(request)
            except Exception as exc:      # a timeout costs 10 minutes: pay for it once
                f.write_text(json.dumps({"error": f"{type(exc).__name__}: {exc}"[:300], "case_id": request.case_id}))
                self.sources["live-error"] += 1; raise
            usage = self.inner.last_usage; src = "live"
            f.write_text(json.dumps({"raw": raw, "usage": usage, "model": self.model, "params": self.params,
                                     "template_id": request.template_id, "case_id": request.case_id}))
        self.last_usage = usage or None; self.last_source = src; self.sources[src] += 1
        return raw

def load_replay(pr):
    out = {}
    for f in glob.glob(f"{a.old_candidates}/*__{pr}__*.json"):
        r = json.loads(Path(f).read_text())
        for u in r.get("units", []):
            if u.get("outcome") == "completion" and u.get("candidate_raw") and u.get("prompt_sha256"):
                out[u["prompt_sha256"]] = {"raw": "```\n" + u["candidate_raw"] + "```", "usage": u.get("usage") or None}
            elif u.get("prompt_sha256") and u.get("outcome") in ("placeholder_violation", "unparseable", "no_output", "limits_exceeded", "backend_error"):
                out[u["prompt_sha256"]] = {"failed": u["outcome"]}
    return out

# ------------------------------------------------------------------ SAP directories of a PR
def slug(n): return re.sub(r"[^A-Za-z0-9]+", "-", n).strip("-")[:80]

def sap_dirs(pair, pr):
    """[(short name, dir, origin)] for every SAP of the PR. The new mint names a SAP
    slug(source_path)[:80], which collides for sibling classes of a deep package and MERGES their
    files into one directory; such SAPs are read from the old mint instead (origin 'old:collision')."""
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
        if d.name not in used and len(groups.get(d.name, [])) <= 1 and d.name not in groups:
            out.append((d.name, d, "new:no_old_counterpart"))
    return out, old_root, new_root

# ------------------------------------------------------------------ the upstream PR as a diff
def pr_base(head):
    """mint_477.pr_base: the fork point of the PR's own line of development."""
    parents = (sgit("rev-list", "--parents", "-1", head) or "").split()[1:]
    if subprocess.run(["git", "--git-dir", str(SRC), "merge-base", "--is-ancestor", head, "HEAD"]).returncode != 0:
        return (sgit("merge-base", head, "HEAD") or "").strip() or None, "merge_base"
    if len(parents) == 1: return parents[0], "first_parent"
    return None, "ambiguous_merge_head_already_landed"

LS = None
def fork_files():
    global LS
    if LS is None:
        LS = collections.defaultdict(list)
        for p in subprocess.run(["git", "ls-files"], cwd=FORK, capture_output=True, text=True).stdout.splitlines():
            LS[Path(p).name].append(p)
    return LS

def type_file(owner, near):
    """Fork file declaring type `owner`, preferring the project of `near`."""
    hits = [p for ext in ("java", "scala") for p in fork_files().get(f"{owner}.{ext}", [])]
    if len(hits) > 1:
        same = [p for p in hits if p.split("/src/")[0] == near.split("/src/")[0]]
        main = [p for p in (same or hits) if "/src/main/" in p]
        hits = main or same or hits
    return hits[0] if len(hits) == 1 else None

def fork_context(missing, near, limit=14):
    """Signatures of the receiver type's members closest to each missing name, read from the
    working tree AS APPLIED (so an applied co-change's declaration shows up)."""
    out = []
    for m in missing:
        if len(out) >= limit: break
        todo = owner_types(m.owner)[:2]; seen = set()
        while todo:
            owner = todo.pop(0)
            rel = type_file(owner, near)
            if rel is None or owner in seen: continue
            seen.add(owner)
            src = (FORK / rel).read_text(errors="replace")
            close = rank_members(m.name, member_signatures(src, "scala" if rel.endswith(".scala") else "java"), limit=6)
            if close:
                out.append(f"Members of {owner} ({rel}) with a name close to `{m.name}`:")
                out += [f"  {s}" for s in close]
                break
            out.append(f"{owner} ({rel}) declares no member with a name close to `{m.name}`.")
            sup = re.search(rf"\b(?:class|interface|trait|object)\s+{re.escape(owner)}\b[^{{]*?\bextends\s+([A-Z][\w$]*)", src)
            if sup and len(seen) < 3: todo.insert(0, sup.group(1))     # then its direct supertype
    return out[:limit]

# ------------------------------------------------------------------ one PR
def run_pr(pair, pr, backend, sample_keys):
    t_pr = time.time(); prn = pr.split("-")[1]
    adir = OUTD / "attempts" / pr; rdir = OUTD / "rounds" / pr
    adir.mkdir(parents=True, exist_ok=True); rdir.mkdir(parents=True, exist_ok=True)
    backend.replay = load_replay(pr); backend.sources = collections.Counter()
    saps, old_root, new_root = sap_dirs(pair, pr)
    row = {"pr": pr, "pair": pair, "k": a.k, "mech_hunks": a.mech_hunks, "saps": [], "sample_keys": sample_keys}

    # ---- the upstream diff
    head = (sgit("rev-parse", f"refs/salp/pr/{prn}^{{commit}}") or "").strip()
    base, how = pr_base(head) if head else (None, "no_pr_ref")
    # zero context: one hunk per contiguous change group, the granularity coverage is decided at
    diff_text = sgit("diff", "-U0", "-M", "--no-color", base, head) if base else None
    files = cc.parse_unified_diff(diff_text or "")
    row["upstream"] = {"head": head, "base": base, "base_how": how, "files": len(files),
                       "hunks": sum(len(f.hunks) for f in files)}
    old_path = {f.new_path: f.old_path for f in files if f.new_path}
    def src_text(rev, path):
        return sgit("show", f"{rev}:{path}") if rev and path else None

    old_rows = {}
    if Path(a.old_results).exists():
        for l in open(a.old_results):
            if f'/{pr}/' in l[:200]:
                r = json.loads(l); old_rows[r["sap"]] = r
    # ---- attempt 1 for every SAP; regions for every function of every SAP
    units, regions, missing = [], [], []
    t_gen = time.time()
    for name, sdir, origin in saps:
        srow = {"sap": name, "dir": str(sdir), "origin": origin, "units": [], "missing_units": []}
        row["saps"].append(srow)
        try:
            sap = json.loads((sdir / "sap.json").read_text())
            manifest = json.loads(((old_root if origin.startswith("old") else new_root) / "pr.json").read_text())
        except Exception as exc:
            srow["error"] = f"unreadable SAP: {exc!r}"[:300]; missing.append({"sap": name, "reason": "unreadable"}); continue
        srow["source_file"] = sap["source_file"]; srow["target_file"] = sap["target_file"]
        ext = EXT.get(sap["source_file"].rsplit(".", 1)[-1], "java")
        head_src = src_text(head, sap["source_file"])
        base_src = src_text(base, old_path.get(sap["source_file"]) or sap["source_file"])
        for fn in sap.get("functions", []):
            fdir = sdir / "functions" / fn
            after = (fdir / f"source.after.{ext}"); before = (fdir / f"source.before.{ext}")
            at = after.read_text(errors="replace") if after.exists() else ""
            bt = before.read_text(errors="replace") if before.exists() else ""
            regions.append(cc.UnitRegion(
                key=f"{name}::{fn}", source_file=sap["source_file"],
                old_source_file=old_path.get(sap["source_file"]),
                after_span=cc.locate(at, head_src) if head_src else None,
                before_span=cc.locate(bt, base_src) if base_src else None, after_text=at, before_text=bt))
        o = old_rows.get(name, {})
        backend.expected = list(zip(o.get("unit_outcomes") or [], o.get("prompt_chars") or []))
        try:
            res = run_case(sdir, manifest, backend)          # attempt 1: the normal pipeline, unchanged
        except Exception as exc:
            srow["refused"] = {"stage": getattr(exc, "stage", None), "error": f"{type(exc).__name__}: {exc}"[:300]}
            missing.append({"sap": name, "reason": f"refused:{getattr(exc, 'stage', None)}", "functions": sap.get("functions", [])})
            continue
        for i, u in enumerate(res.units):
            ent = u.context.target_localization.get("function")
            fn = (ent or "").split("/")[-1]; key = f"{name}::{fn}"
            if u.generation.outcome != "completion":
                outcome = u.generation.outcome; diags = list(u.generation.diagnostics)[:2]
                if outcome == "backend_error" and diags and diags[0].startswith("stored-477 outcome: "):
                    outcome = diags[0].split(": ", 1)[1]; diags = ["outcome of the 477 run for the identical prompt; not regenerated"]
                srow["missing_units"].append({"unit": key, "outcome": outcome, "diagnostics": diags})
                missing.append({"sap": name, "unit": key, "reason": outcome}); continue
            art = next((x for x in u.context.program_context if x.role == "target" and x.entity == ent), None)
            def expand(c, art=art):
                for e in (art.placeholders if art else []):
                    c = c.replace(PLACEHOLDER.format(n=e.ph_id), e.original_text, 1)
                return c
            st = PRUnit(key=key, sap=name, file=sap["target_file"], index=i, entity=ent, context=u.context,
                        request=u.request, generation=u.generation, target_original=recover(art) if art else "",
                        expand=expand, expected_placeholders=_expected_placeholders(u.context),
                        is_test="/src/test/" in sap["target_file"])
            st.art = art; units.append(st); srow["units"].append(key)
        backend.expected = []
    row["attempt1_generation"] = {"seconds": round(time.time() - t_gen, 1), "sources": dict(backend.sources)}
    region_of = {r.key: r for r in regions}
    by_key = {u.key: u for u in units}
    cc.classify_hunks(files, regions)

    # ---- what the SAP units do not cover, and (optionally) mechanical application of it
    sap_target = {s["source_file"]: s["target_file"] for s in row["saps"] if "source_file" in s}
    tracked = {p for ps in fork_files().values() for p in ps}
    mech, cov, cand = {}, [], collections.defaultdict(list)
    for f in files:
        tgt = sap_target.get(f.new_path or "") or (f.new_path if f.new_path in tracked else None) \
            or (f.old_path if f.status == "R" and f.old_path in tracked else None)
        new_file = f.status == "A" and f.new_path not in tracked
        for h in f.hunks:
            c = {"hunk": h.hunk_id, "status": f.status, "coverage": h.coverage, "covered_by": h.covered_by,
                 "target": tgt or (f.new_path if new_file else None), "in_fork": tgt is not None,
                 "summary": cc.summarize_hunk(h)[:160]}
            cov.append(c)
            if h.coverage != "none": continue
            if f.binary or f.status == "D": c["mech"] = "skipped:binary_or_delete"; continue
            target = tgt or (f.new_path if new_file else None)
            if target is None: c["mech"] = "skipped:target_file_absent_in_fork"; continue
            if "/src/" not in target: c["mech"] = "skipped:not_under_a_gradle_source_dir"; continue
            c["mech"] = "candidate"; cand[f.path].append((h, target, new_file))
    for path, items in cand.items():
        # groups closer than the context width are merged, so that each mechanical hunk applies on its own
        f = next(x for x in files if x.path == path)
        old_lines = (src_text(base, f.old_path or f.new_path) or "").splitlines() if f.status != "A" else []
        for h in cc.with_context([i[0] for i in items], old_lines):
            members = h.members or [h.hunk_id]
            texts = [t for g in items if g[0].hunk_id in members for t in [x for _, x in g[0].added] + [x for _, x in g[0].removed]]
            mech[h.hunk_id] = {"hunk": h, "target": items[0][1], "new_file": items[0][2], "members": members,
                               "blocks": ["\n".join(t for _, t in g[0].added) for g in items if g[0].hunk_id in members],
                               "names": tuple(cc.declared_names(texts))}
            for c in cov:
                if c["hunk"] in members: c["mech_id"] = h.hunk_id
    row["coverage"] = {"hunks": cov, "counts": dict(collections.Counter(
        (c["coverage"] if c["coverage"] != "none" else "none:" + ("in_fork" if c["in_fork"] else "file_absent_in_fork")) for c in cov))}

    touched = sorted({u.file for u in units} | ({m["target"] for m in mech.values()} if a.mech_hunks else set()))
    created = sorted({m["target"] for m in mech.values() if m["new_file"]}) if a.mech_hunks else []

    def restore():
        tr = [p for p in touched if p in tracked]
        if tr: subprocess.run(["git", "checkout", "-q", "--", *tr], cwd=FORK)
        for p in created:
            if (FORK / p).exists(): (FORK / p).unlink()

    def apply_state(applied, mech_ids):
        """Write the combined state; -> (spans_by_file, sources, not_spliced, mech_failed).
        Mechanical hunks go in first, by context, on the fork's own text; units are then spliced by
        exact replacement of their original text. If a unit no longer splices because a mechanical
        hunk changed its text, the UNIT wins: that file's mechanical hunks are taken out."""
        restore(); not_spliced, mech_failed, spans, sources = {}, {}, collections.defaultdict(list), {}
        byfile = collections.defaultdict(list)
        for key in applied: byfile[by_key[key].file].append(by_key[key])
        mech_of = collections.defaultdict(list)
        for mid in mech_ids:
            m = mech[mid]
            rc, out, _ = sh(["git", "apply", "--whitespace=nowarn", "-"], 60, stdin=m["hunk"].patch(m["target"]))
            if rc != 0: mech_failed[mid] = "no longer applies: " + (out.strip().splitlines() or ["?"])[0][:200]
            else: mech_of[m["target"]].append(mid)
        def splice(text, us):
            miss = {}
            for u in sorted(us, key=lambda x: (x.sap, x.index)):
                n = text.count(u.target_original or "\0")
                if n == 1: text = text.replace(u.target_original, applied[u.key], 1)
                else: miss[u.key] = f"original text found {n} times"
            return text, miss
        for path, us in byfile.items():
            text, miss = splice((FORK / path).read_text(encoding="utf-8"), us)
            if miss and mech_of.get(path):
                subprocess.run(["git", "checkout", "-q", "--", path], cwd=FORK)
                text0, miss0 = splice((FORK / path).read_text(encoding="utf-8"), us)
                if len(miss0) < len(miss):
                    for mid in mech_of.pop(path): mech_failed[mid] = "conflicts with a ported unit of the same file"
                    text, miss = text0, miss0
                else:                                   # not the mechanical hunks' fault: put them back
                    for mid in mech_of[path]: sh(["git", "apply", "--whitespace=nowarn", "-"], 60, stdin=mech[mid]["hunk"].patch(path))
            not_spliced.update(miss)
            (FORK / path).write_text(text, encoding="utf-8")
        for path in sorted(set(byfile) | set(mech_of)):
            if not (FORK / path).exists(): continue
            text = (FORK / path).read_text(encoding="utf-8", errors="replace"); sources[path] = text
            for u in byfile.get(path, []):
                if u.key in not_spliced: continue
                pos = text.find(applied[u.key])
                if pos < 0: continue
                first = text.count("\n", 0, pos) + 1
                names = tuple(dict.fromkeys(cc.declared_names([(u.target_original or "").split("{")[0].replace("\n", " "),
                                                               applied[u.key].split("{")[0].replace("\n", " ")])))
                spans[path].append(Span(u.key, first, first + applied[u.key].rstrip("\n").count("\n"), "unit", names))
            for mid in mech_of.get(path, []):
                m = mech[mid]
                if m["new_file"]: spans[path].append(Span(mid, 1, text.count("\n") + 1, "mech", m["names"])); continue
                found = [sp for sp in (cc.locate(b, text) for b in m["blocks"] if b.strip()) if sp]
                for sp in found or [(0, -1)]: spans[path].append(Span(mid, sp[0], sp[1], "mech", m["names"]))
        return dict(spans), sources, not_spliced, mech_failed

    # mechanical hunks: keep only those that apply cleanly to the unmodified fork, in order
    mech_ids = []
    if a.mech_hunks:
        restore()
        for mid, m in mech.items():
            rc, out, _ = sh(["git", "apply", "--whitespace=nowarn", "-"], 60, stdin=m["hunk"].patch(m["target"]))
            if rc == 0: mech_ids.append(mid)
            for c in cov:
                if c.get("mech_id") == mid:
                    c["mech"] = "applies_cleanly" if rc == 0 else "rejected:" + (out.strip().splitlines() or ["?"])[0][:160]
        restore()
    row["mech_candidates"] = len(mech); row["mech_applied_initially"] = list(mech_ids)
    row["mech_members"] = {mid: m["members"] for mid, m in mech.items()}
    group_mech = {g: mid for mid, m in mech.items() for g in m["members"]}

    mappers = {}
    def hunk_applied(h, applied_keys, mech_now):
        if group_mech.get(h.hunk_id) in mech_now: return True, "applied mechanically"
        if h.coverage == "full" and h.covered_by and all(k in applied_keys for k in h.covered_by):
            return True, "ported by " + ", ".join(h.covered_by)
        if h.covered_by: return False, "covering unit(s) not applied or hunk only partly covered: " + ", ".join(h.covered_by)
        return False, "no SAP unit covers this hunk"

    def evidence_for(u, diag, applied_keys, mech_now):
        names = [m["name"] for m in diag.missing]
        ev = cc.find_evidence(names, files, region_of.get(u.key), all_regions=regions)
        out = {}
        for name, items in ev.items():
            rows = []
            for e in items:
                if group_mech.get(e.hunk_id) in mech_now: ok, why = True, "applied mechanically"
                elif e.owner and e.owner in applied_keys:
                    has = re.search(rf"(?<![\w$]){re.escape(e.matched_as)}(?![\w$])", by_key[e.owner].expanded or "")
                    ok, why = bool(has), (f"ported by {e.owner}" if has else f"{e.owner} is applied but its candidate does not contain `{e.matched_as}`")
                elif e.owner: ok, why = False, f"its unit {e.owner} is not applied"
                else: ok, why = False, "no SAP unit covers this line"
                rows.append({"symbol": name, "file": e.file, "line": e.line, "text": e.text, "declaration": e.declaration,
                             "hunk": e.hunk_id, "available": ok, "why": why})
            out[name] = rows
        return out

    def precheck(u, diag, applied_keys, mech_now):
        ev = evidence_for(u, diag, applied_keys, mech_now)
        return [rows[0] for rows in ev.values() if rows and not any(r["available"] for r in rows)]

    def repair_extras(u, diag, applied_keys, mech_now):
        ev = evidence_for(u, diag, applied_keys, mech_now) if diag.kind == "build" else {}
        notes, omitted = cc.cochange_notes(
            files, region_of.get(u.key), lambda h: hunk_applied(h, applied_keys, mech_now),
            priority_hunks=[r["hunk"] for rows in ev.values() for r in rows],
            in_fork=lambda f: bool(sap_target.get(f.new_path or "") or f.new_path in tracked or f.old_path in tracked))
        return {"cochanges": notes, "cochanges_omitted": omitted, "allowed_symbols": cc.allowed_symbols(notes)}

    def on_attempt(u, attempt, art):
        stem = f"{u.sap}.{u.key.split('::')[1]}.attempt-{attempt:02d}"
        (adir / f"{stem}.prompt.txt").write_text(art["prompt"])
        (adir / f"{stem}.candidate.json").write_text(json.dumps(
            {"unit": u.key, "file": u.file, **{k: art.get(k) for k in ("candidate_raw", "candidate_expanded", "outcome", "diagnostics")},
             "source": backend.last_source if attempt > 1 else None}, indent=1))
        if attempt == 1:
            (adir / f"{u.sap}.{u.key.split('::')[1]}.target_original.txt").write_text(u.target_original)

    def validate(rnd, applied, mech_now):
        t0 = time.time(); tag = f"round-{rnd:02d}"; detail = {}
        if not applied and not mech_now:
            return PRValidation(verdict="NOTHING_APPLIED", summary="no unit and no mechanical hunk left to apply")
        try:
            spans, sources, not_spliced, mech_failed = apply_state(applied, mech_now)
            detail["spans"] = {p: [(s.key, s.first, s.last, s.kind) for s in ss] for p, ss in spans.items()}
            live = {k: v for k, v in applied.items() if k not in not_spliced}
            changed_paths = sorted(spans)
            projs = sorted({project_of(p) for p in changed_paths if project_of(p)})
            detail["projects"] = projs
            # what could not be written is reported and left out; the rest is validated as is
            mech_now = [m for m in mech_now if m not in mech_failed]
            if not live and not mech_now:
                return PRValidation(verdict="NOTHING_APPLIED", not_spliced=not_spliced, mech_not_applied=mech_failed,
                                    summary="nothing could be applied", detail=detail)
            rc, log, secs = gradle(*[f"{p}:testClasses" for p in projs], timeout=BUILD_TIMEOUT)
            (rdir / f"{tag}.build.log").write_text(log)
            detail["build"] = {"rc": rc, "seconds": secs, "log_chars": len(log)}
            if rc == "timeout":
                return PRValidation(verdict="INCONCLUSIVE", unit_verdicts={k: "INCONCLUSIVE" for k in live}, not_spliced=not_spliced,
                                    mech_not_applied=mech_failed, summary="build timeout", detail=detail)
            if rc != 0:
                for k in live: mappers[k] = line_mapper(by_key[k].raw or "", by_key[k].art)
                errors = attribute_errors(parse_compile_errors(log), spans, root=str(FORK), sources=sources,
                                          unit_line=lambda k, n: mappers[k](n) if k in mappers else n)
                diags, mdiag, unattr = {}, {}, []
                for e in errors:
                    if not e.keys: unattr.append(f"{e.rel}:{e.line}: {e.message}")
                    for key in e.keys:
                        if key in mech: mdiag.setdefault(key, f"compile error {e.rel}:{e.line}: {e.message} ({e.how})")
                if not errors:
                    text = _fallback(log, 1500)
                    json_mech = [m for m in mech_now if mech[m]["target"].endswith(".json")]
                    if json_mech and re.search(r"processMessages|MessageGenerator|generator", log):
                        for m in json_mech: mdiag[m] = "message generation failed: " + text[:200]
                    else: unattr.append("build failed with no file:line compiler error: " + text[:400])
                for k in live:
                    mine = [e for e in errors if k in e.keys]
                    if not mine: continue
                    u = by_key[k]; miss = missing_symbols(mine)
                    text = render_errors_for(k, errors, u.file)
                    (rdir / f"{tag}.feedback.{u.sap}.{k.split('::')[1]}.txt").write_text(text + "\n")
                    diags[k] = UnitDiagnostic(
                        kind="build", text=text, how=sorted({e.how for e in mine}),
                        summary=f"{len(mine)} compile error(s); first: {Path(mine[0].rel).name}:{mine[0].line}: {mine[0].message}"[:300],
                        missing=[m.__dict__ for m in miss], fork_context=fork_context(miss, u.file))
                detail["errors"] = {"total": len(errors), "unattributed": len(unattr),
                                    "by_how": dict(collections.Counter(e.how or "none" for e in errors))}
                return PRValidation(verdict="BUILD_FAIL", unit_diags=diags, mech_diags=mdiag, unattributed=unattr[:20],
                                    not_spliced=not_spliced, mech_not_applied=mech_failed,
                                    unit_verdicts={k: ("BUILD_FAIL" if k in diags else "BLOCKED") for k in live},
                                    summary=f"{len(errors)} compile error(s), {len(unattr)} unattributed", detail=detail)
            # ---- tests
            tests = collections.defaultdict(set); test_file_of, mapped_files, classes_of = {}, collections.defaultdict(list), {}
            for p in changed_paths:
                tproj, tcls, thow = find_test(p); tcls = [c for c in tcls if c]
                classes_of[p] = (tproj, tcls, thow)
                for c in tcls:
                    tests[tproj].add(c)
                    if thow == "self": test_file_of[c] = p
                    else: mapped_files[c].append(p)
            need = [(tp, c) for tp, cs in tests.items() for c in sorted(cs) if f"{tp}|{c}" not in baseline]
            if need:
                restore()
                for tp, c in need:
                    tf = test_file_of.get(c)
                    if tf and not (FORK / tf).exists():       # a test class the PR adds: nothing to compare with
                        baseline[f"{tp}|{c}"] = {"rc": 0, "ran": 0, "failed": [], "seconds": 0, "new_class": True}
                    else:
                        ran, fl, brc, blog, bsecs = run_test(tp, [c])
                        baseline[f"{tp}|{c}"] = {"rc": brc, "ran": sum(ran.values()), "failed": sorted(f.name for f in fl), "seconds": bsecs}
                        (rdir / f"baseline.{c}.test.log").write_text(blog)
                    BASE.write_text(json.dumps(baseline, indent=1))
                apply_state(applied, mech_now)
            new, unusable, timed_out, tsum = [], [], False, {}
            for tproj, tcls in sorted(tests.items()):
                usable = []
                for c in sorted(tcls):
                    b = baseline[f"{tproj}|{c}"]
                    if b["rc"] == "timeout" or (b["ran"] == 0 and not b.get("new_class")): unusable.append(c)
                    else: usable.append(c)
                if not usable: continue
                ran, failures, trc, tlog, tsecs = run_test(tproj, usable)
                (rdir / f"{tag}.test.{tproj.strip(':').replace(':', '-')}.log").write_text(tlog)
                timed_out |= trc == "timeout"
                bfailed = {f for c in usable for f in baseline[f"{tproj}|{c}"]["failed"]}
                mine = [f for f in failures if f.name not in bfailed and any(f.name.startswith(c + ".") for c in usable)]
                if trc != "timeout":
                    for c in usable:
                        if ran.get(c, 0) == 0 and not any(f.name.startswith(c + ".") for f in mine):
                            mine.append(TestFailure(name=f"{c}.<class did not run>", kind="NoTestsRan",
                                                    message="the test class produced no result: it did not run to completion with these changes applied",
                                                    frames=[]))
                new += mine
                tsum[tproj] = {"classes": usable, "rc": trc, "ran": ran, "seconds": tsecs, "new_failures": sorted(f.name for f in mine)}
            detail["tests"] = tsum; detail["baseline_unusable"] = unusable
            (rdir / f"{tag}.tests.json").write_text(json.dumps({"tests": tsum, "failures": [f.__dict__ for f in new]}, indent=1))
            charged = attribute_test_failures(new, spans, test_file_of_class=test_file_of, mapped_files=dict(mapped_files), sources=sources)
            diags, mdiag = {}, {}
            for key, rows in charged.items():
                if key in mech:
                    mdiag[key] = f"new test failure {rows[0][0].name} ({rows[0][1]})"; continue
                if key not in live: continue
                u = by_key[key]
                text = render_test_failures([f for f, _ in rows], unit_is_test=u.is_test, pr_context=True)
                (rdir / f"{tag}.feedback.{u.sap}.{key.split('::')[1]}.txt").write_text(text + "\n")
                diags[key] = UnitDiagnostic(kind="test", text=text, how=sorted({h for _, h in rows}),
                                            summary=f"{len(rows)} new failing test(s); first: {rows[0][0].name}: {rows[0][0].message[:160]}"[:300])
            charged_failures = {id(f) for rows in charged.values() for f, _ in rows}
            unattr = [f"{f.name}: {f.message[:160]}" for f in new if id(f) not in charged_failures]
            verdicts = {}
            for k in live:
                tproj, tcls, thow = classes_of.get(by_key[k].file, (None, [], "?"))
                ok_cls = [c for c in tcls if c not in unusable]
                verdicts[k] = ("TEST_FAIL" if k in diags else
                               "INCONCLUSIVE" if not ok_cls or timed_out else "VALID")
            detail["test_how"] = {p: v[2] for p, v in classes_of.items()}
            verdict = ("TEST_FAIL" if new else "INCONCLUSIVE" if timed_out or not tsum else "VALID")
            return PRValidation(verdict=verdict, unit_diags=diags, mech_diags=mdiag, unattributed=unattr[:20],
                                not_spliced=not_spliced, mech_not_applied=mech_failed,
                                unit_verdicts=verdicts, detail=detail,
                                summary=(f"{len(new)} new failing test(s)" if new else
                                         "test timeout" if timed_out else "build ok; no usable test class" if not tsum else
                                         f"{sum(sum(t['ran'].values()) for t in tsum.values())} tests ran, no new failures"))
        except Exception as exc:
            detail["traceback"] = traceback.format_exc()[-1500:]
            return PRValidation(verdict="INCONCLUSIVE", unit_verdicts={k: "INCONCLUSIVE" for k in applied},
                                summary=f"harness error: {exc!r}"[:300], detail=detail)
        finally:
            restore()

    def timed(rnd, applied, mech_now):
        t0 = time.time(); v = validate(rnd, applied, mech_now); v.seconds = round(time.time() - t0, 1)
        print(f"    {pr} round {rnd}: {len(applied)} unit(s) + {len(mech_now)} mech -> {v.verdict} | {v.summary} "
              f"| charged {len(v.unit_diags)} unit(s), {len(v.mech_diags)} mech, {len(v.unattributed)} unattributed ({v.seconds}s)", flush=True)
        return v

    row["missing_siblings"] = missing
    row["units_total"] = len(regions); row["units_with_candidate"] = len(units)
    t_loop = time.time(); backend.sources = collections.Counter()
    rec = run_pr_loop(units, backend, timed, k=a.k, mech=mech_ids, precheck=precheck,
                      repair_extras=repair_extras, on_attempt=on_attempt)
    restore()
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=FORK, capture_output=True, text=True).stdout.strip()
    row.update(rec)
    row["loop_seconds"] = round(time.time() - t_loop, 1); row["reprompt_generation_sources"] = dict(backend.sources)
    row["fork_clean_after"] = not dirty
    row["seconds"] = round(time.time() - t_pr, 1)
    return row

# ------------------------------------------------------------------ main
sample = json.loads(Path(a.sample).read_text())
if isinstance(sample, dict):
    # a PR sample ({"pair": ..., "prs": [...]}, round 3): every SAP of each PR, in the drawn order
    sample = [{"new_dir": f"{sample['pair']}/PR-{p}", "old_key": f"{sample['pair']}/PR-{p}"} for p in sample["prs"]]
prs = collections.OrderedDict()
for s in sample:
    pair, pr = s["new_dir"].split("/")[:2]
    prs.setdefault((pair, pr), []).append(s["old_key"])
if a.only: prs = collections.OrderedDict((k, v) for k, v in prs.items() if k[1] in a.only)
OUTD.mkdir(parents=True, exist_ok=True)
done = {json.loads(l)["pr"] for l in open(OUT)} if OUT.exists() else set()
todo = [(k, v) for k, v in prs.items() if k[1] not in done]
if a.limit: todo = todo[:a.limit]
dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=FORK, capture_output=True, text=True).stdout.strip()
if dirty: sys.exit(f"fork clone is not clean:\n{dirty}")
backend = CachingBackend(OllamaBackend(), {})
print(f"backend {backend.model} digest={backend.model_digest} params={backend.params}; k={a.k} mech={a.mech_hunks}; "
      f"{len(todo)} PR(s) to run ({len(done)} done)", flush=True)
for n, ((pair, pr), keys) in enumerate(todo, 1):
    try:
        row = run_pr(pair, pr, backend, keys)
    except Exception as exc:
        row = {"pr": pr, "pair": pair, "error": f"{type(exc).__name__}: {exc}"[:400], "traceback": traceback.format_exc()[-1500:]}
        subprocess.run(["git", "checkout", "-q", "--", "."], cwd=FORK)
    with OUT.open("a") as fh: fh.write(json.dumps(row, default=str) + "\n")
    us = row.get("units", {})
    print(f"[{n}/{len(todo)}] {pr}: attempt-1 PR verdict {row.get('attempt1_verdict')} -> final {row.get('final_verdict')} "
          f"stop={row.get('stop_reason')} rounds={row.get('rounds_run')} | units {len(us)} with a candidate of {row.get('units_total')}; "
          f"final VALID_PR {sum(1 for u in us.values() if u['final_verdict'] == 'VALID_PR')} | {row.get('seconds')}s {row.get('error', '')}", flush=True)
print("ALL DONE", flush=True)
