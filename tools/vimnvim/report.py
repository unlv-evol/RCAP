#!/usr/bin/env python3
"""Score every output of the Vim -> Neovim evaluation and write the tables.

  report.py --out DIR

Reads the frozen config, the stored baseline files (read-only) and whatever arm
files exist under DIR/arms (an arm still running is reported with its n).
Writes DIR/scored/outputs.jsonl (one row per scored output), DIR/scored/summary.json
(every number of the report) and DIR/tables.md (the tables REPORT.md quotes).
Calls no model. Compiles through DIR/compile's cache (a new output is compiled once).
"""
import argparse, collections, json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import (extract_code, flagged_baseline, load_cases, load_config, load_jsonl, nl, sha)  # noqa: E402
from compile_c import Compiler  # noqa: E402
from rcap import fidelity as fd  # noqa: E402
from rcap import ppathf_metrics as pm  # noqa: E402
from rcap.accept import placeholder_marks, rejection  # noqa: E402
from rcap.reprompt import code_equal  # noqa: E402
from rcap.scoring import score_unit  # noqa: E402

ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True)
ap.add_argument("--no-compile", action="store_true", help="use cached compile results only")
A = ap.parse_args()
OUT = Path(A.out); cfg = load_config(OUT); cases = load_cases(cfg)
SC = OUT / "scored"; SC.mkdir(exist_ok=True)
base = load_jsonl(cfg["stored"]["baseline"])
rerun = load_jsonl(cfg["stored"]["baseline_rerun"])
# the sliced test set is in ANOTHER ORDER than the full one, so its index-based case ids
# name other cases: both it and its stored results are matched by the target commit
_by_commit = {c["commit_id_target"]: cid for cid, c in cases.items()}
ppr = {_by_commit[r["commit_id_target"]]: r for r in load_jsonl(cfg["stored"]["ppathf_reduced"]).values()}
sliced = {_by_commit[c["commit_id_target"]]: c
          for c in json.loads(Path(cfg["stored"]["ppathf_reduced_test_set"]).read_text())}
assert len(ppr) == len(sliced) == len(cases)
FLAGGED = flagged_baseline(cfg, base)
sub = json.loads((OUT / "subset.json").read_text()) if (OUT / "subset.json").exists() else None
REACH = set(sub["reachable"]) if sub else set()
compiler = Compiler(cfg)

# ------------------------------------------------------------------ one output
_cache_file = SC / "eval-cache.json"
_cache = json.loads(_cache_file.read_text()) if _cache_file.exists() else {}


def evaluate(cid: str, text: str | None, earlier: tuple[str, ...] = ()) -> dict:
    """Metrics (aligned and raw), label, acceptance verdict and compile result of one output."""
    case = cases[cid]
    pre, dev = nl(case["func_before_target"]), nl(case["func_after_target"])
    key = sha(cid + "\0" + (text if text is not None else "\0NONE") + "\0" + "\0".join(earlier))
    hit = _cache.get(key)
    if hit is None:
        hit = {"aligned": pm.score_aligned(text, dev, pre)}
        if text is None:
            hit.update(raw=None, label="NO_CANDIDATE", accept=None)
        else:
            f_s, f_sp = nl(case["func_before_source"]), nl(case["func_after_source"])
            u = score_unit(pre, text, f_s, f_sp, language="c", rename_context="isolation", earlier=earlier)
            rej = rejection(text, pre, f_s, f_sp, language="c")
            hit.update(raw=pm.score(text, dev, pre), label=u.label,
                       reasons=(u.reasons or (u.fidelity.reasons if u.fidelity else []))[:2],
                       recall=u.fidelity.recall if u.fidelity else None,
                       novel_share=u.fidelity.novel_share if u.fidelity else None,
                       accept=rej[0] if rej else None, code_equal_fork=code_equal(text, pre),
                       marks=placeholder_marks(text)[:4])
        _cache[key] = hit
    out = dict(hit)
    out["compiles"] = None
    if cid in REACH and text is not None:
        if A.no_compile:
            site = compiler.site(cid, case)
            import hashlib
            k = hashlib.sha256(f"{compiler.commit}\0{site['rel']}\0{site['name']}\0{' '.join(compiler.extra)}\0{text}".encode()).hexdigest()
            f = compiler.cache / f"{k}.json"
            out["compiles"] = json.loads(f.read_text())["ok"] if f.exists() else None
        else:
            out["compiles"] = compiler.check(cid, case, text)["ok"]
    return out


def stored_text(r: dict | None) -> str | None:
    return nl(r["candidate"].strip()) if r and r.get("outcome") == "ok" and r.get("candidate") else None


def recover_ppathf(cid: str, text: str | None) -> str | None:
    if text is None:
        return None
    for idx, piece in sliced[cid]["removed_pieces_sliced_target"].items():
        text = text.replace(f"/* placeholder_{idx} */", piece, 1)
    return text


# ------------------------------------------------------------------ series
SERIES: dict[str, dict] = collections.OrderedDict()
ROWS: list[dict] = []


def add(name: str, texts: dict[str, str | None], *, step, desc, scope=None, meta=None, earlier=None):
    rows = {}
    for cid in (scope or list(cases)):
        if cid not in texts:
            continue
        e = evaluate(cid, texts[cid], tuple((earlier or {}).get(cid, ())))
        row = {"series": name, "case_id": cid, "has_output": texts[cid] is not None,
               "flagged": cid in FLAGGED, "reachable": cid in REACH, **e["aligned"],
               "raw_exact": e["raw"]["exact_match"] if e["raw"] else None,
               "raw_rel": e["raw"]["rel_distance"] if e["raw"] else None,
               "raw_gen": e["raw"]["gen_distance"] if e["raw"] else None,
               "label": e["label"], "accept": e["accept"], "compiles": e["compiles"],
               "recall": e.get("recall"), "novel_share": e.get("novel_share"),
               "code_equal_fork": e.get("code_equal_fork"), **((meta or {}).get(cid, {}))}
        rows[cid] = row
        ROWS.append(row)
    SERIES[name] = {"step": step, "desc": desc, "rows": rows}
    return rows


def agg(rows, cids=None) -> dict:
    rs = [r for c, r in rows.items() if cids is None or c in cids]
    a = pm.aggregate(rs)
    a["dummy"] = sum(1 for r in rs if r["use_dummy"])
    return a


def paired(rows_a, rows_b, cids=None) -> dict:
    common = [c for c in rows_a if c in rows_b and (cids is None or c in cids)]
    ea, eb = [rows_a[c]["exact_match"] for c in common], [rows_b[c]["exact_match"] for c in common]
    ra, rb = [rows_a[c]["rel_distance"] for c in common], [rows_b[c]["rel_distance"] for c in common]
    return {"n": len(common), "mcnemar": pm.mcnemar_exact(ea, eb), "sign_red": pm.sign_test(ra, rb),
            "wilcoxon_red": pm.wilcoxon(ra, rb)}


S: dict[str, object] = {"config_id": cfg["config_id"], "flagged": FLAGGED, "n_cases": len(cases)}

origin = add("origin", {c: nl(k["func_before_target"]) for c, k in cases.items()}, step=0,
             desc="Origin: the pre-patch fork function, unchanged")
b_rows = add("baseline", {c: stored_text(base.get(c)) for c in cases}, step=1,
             desc="Zero-shot baseline, stored outputs of 2026-09-20 (harness prompt)")
r_rows = add("baseline_rerun", {c: stored_text(rerun.get(c)) for c in cases}, step=1,
             desc="Zero-shot baseline, stored re-run of 2026-09-26")
p_rows = add("ppathf_reduced", {c: recover_ppathf(c, stored_text(ppr.get(c))) for c in cases}, step=2,
             desc="Zero-shot on PPatHF's OWN reduced inputs, stored outputs of 2026-09-20 (their placeholders, recovered)")

# prompt identity of the stored baseline: rebuild the harness prompt and compare lengths
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("stored_harness", cfg["stored"]["harness"])
    hz = importlib.util.module_from_spec(spec); spec.loader.exec_module(hz)
    same = sum(1 for c, k in cases.items() if len(hz.build_prompt(k, False)) == base[c]["prompt_chars"])
    S["baseline_prompt_identity"] = {"prompt_chars_equal": same, "of": len(cases),
                                     "prompt_sha256_of_all": sha("".join(hz.build_prompt(k, False) for k in cases.values()))}
except Exception as exc:  # noqa: BLE001
    S["baseline_prompt_identity"] = {"error": str(exc)}

ARM = {}
for arm in ("rcap_full", "rcap_reduced"):
    rows = load_jsonl(OUT / "arms" / f"{arm}.jsonl")
    ARM[arm] = rows
    if not rows:
        continue
    add(arm, {c: (r["candidate"] if r["outcome"] == "completion" else None) for c, r in rows.items()}, step=2,
        desc=f"RCAP request ({'with' if arm == 'rcap_reduced' else 'no'} program reduction): rejected or refused = Origin",
        meta={c: {"outcome": r["outcome"], "capacity": r["capacity"]} for c, r in rows.items()})
    add(arm + ".as_is", {c: r["as_is"] for c, r in rows.items()}, step=2,
        desc=f"RCAP request ({'with' if arm == 'rcap_reduced' else 'no'} program reduction): every answer scored as written",
        meta={c: {"outcome": r["outcome"], "capacity": r["capacity"]} for c, r in rows.items()})

LOOP = {}
for arm in ("loop_feedback", "loop_retry", "ctx_B", "ctx_C", "ctx_C2"):
    if arm not in cfg["arms"]:
        continue
    rows = {c: r for c, r in load_jsonl(OUT / "arms" / f"{arm}.jsonl").items() if "attempts" in r}
    LOOP[arm] = rows
    if not rows:
        continue
    K = cfg["arms"][arm]["k"]
    scope = list(rows)
    # every attempt that produced a candidate
    per_attempt: dict[str, list[tuple[int, str, str]]] = {}
    for c, r in rows.items():
        seq = []
        for at in r["attempts"]:
            if at.get("candidate"):
                seq.append((at["attempt"], at["candidate"], at.get("verdict")))
        per_attempt[c] = seq
    for k in range(1, K + 1):
        rule, final, best, meta = {}, {}, {}, {}
        for c, r in rows.items():
            seq = [x for x in per_attempt[c] if x[0] <= k]
            ok = next((x for x in seq if x[2] == "VALID"), None)
            rule[c] = ok[1] if ok else None
            final[c] = seq[-1][1] if seq else None
            scored = [(evaluate(c, t)["aligned"]["rel_distance"], n, t) for n, t, _ in seq]
            best[c] = min(scored)[2] if scored else None
            meta[c] = {"budget": k, "success_attempt": ok[0] if ok else None, "last_attempt": seq[-1][0] if seq else None}
        add(f"{arm}.rule@{k}", rule, step=3 if arm.startswith("loop") else 4, scope=scope, meta=meta,
            desc="delivered under the loop's rule: the first attempt that compiles, else the fork unchanged")
        add(f"{arm}.final@{k}", final, step=3 if arm.startswith("loop") else 4, scope=scope, meta=meta,
            desc="the last candidate produced, as written")
        add(f"{arm}.best@{k}", best, step=3 if arm.startswith("loop") else 4, scope=scope, meta=meta,
            desc="ORACLE: the attempt closest to the developer's port (uses the ground truth)")
    # each re-prompted attempt as its own output, with the earlier attempts (for REVERTED)
    texts, meta, earlier = {}, {}, {}
    for c, seq in per_attempt.items():
        for n, t, verdict in seq:
            key = f"{c}#a{n}"
            texts[key] = t
            meta[key] = {"attempt": n, "verdict": verdict}
            earlier[key] = tuple(x[1] for x in seq if x[0] < n)
    rows_att = {}
    for key, t in texts.items():
        c = key.split("#")[0]
        e = evaluate(c, t, earlier[key])
        row = {"series": f"{arm}.attempt", "case_id": c, "has_output": True, "flagged": c in FLAGGED,
               "reachable": True, **e["aligned"], "raw_exact": e["raw"]["exact_match"], "raw_rel": e["raw"]["rel_distance"],
               "raw_gen": e["raw"]["gen_distance"], "label": e["label"], "accept": e["accept"], "compiles": e["compiles"],
               "recall": e.get("recall"), "novel_share": e.get("novel_share"),
               "code_equal_fork": e.get("code_equal_fork"), **meta[key]}
        rows_att[key] = row; ROWS.append(row)
    SERIES[f"{arm}.attempt"] = {"step": 3 if arm.startswith("loop") else 4, "desc": "every attempt's candidate", "rows": rows_att}

# ------------------------------------------------------------------ headline
ALL = set(cases); CLEAN = ALL - set(FLAGGED)


def embed(arm: str, kind: str) -> dict | None:
    """The stored baseline with the subset's cases replaced by an arm's output."""
    name = f"{arm}.{kind}@{cfg['arms'][arm]['k']}"
    if name not in SERIES:
        return None
    rows = dict(b_rows); rows.update(SERIES[name]["rows"])
    return rows


def gated(rows: dict) -> dict:
    """`rows` with every reachable output that does NOT compile replaced by Origin."""
    return {c: (origin[c] if (r["reachable"] and r["compiles"] is False) else r) for c, r in rows.items()}


head = []


def headline(label, rows, paper="", note=""):
    if rows is None:
        return
    done = set(rows)
    head.append({"row": label, "n_done": len(done), "all": agg(rows), "clean": agg(rows, CLEAN & done),
                 "paper": paper, "note": note})


headline("0. Origin (pre-patch function unchanged)", origin, "Origin: 0 (0.0%) / 26.60 / 1.00 (Table 1)")
headline("1. Zero-shot baseline (stored 2026-09-20)", b_rows,
         "StarCoder zero-shot @8k: 95 (30.6%) / 14.30 / 0.67; PPatHF (fine-tuned + reduction) @8k: 131 (42.3%) / 11.98 / 0.43 (Table 1)")
headline("1'. Zero-shot baseline, stored re-run 2026-09-26", r_rows, "same")
if "rcap_full" in SERIES:
    headline("2a. RCAP request, no reduction (rejected / refused = Origin)", SERIES["rcap_full"]["rows"],
             "PPatHF-r (fine-tuned, no reduction) @8k: 115 (37.1%) / 14.42 / 0.56 (Table 2)")
    headline("2a'. same, every answer scored as written", SERIES["rcap_full.as_is"]["rows"])
if "rcap_reduced" in SERIES:
    headline("2b. RCAP request + RCAP program reduction (rejected / refused = Origin)", SERIES["rcap_reduced"]["rows"],
             "PPatHF-f (NOT fine-tuned, their reduction) @8k: 115 of 290 completed (39.7%) / 10.35 / 0.46, completed cases only (Table 3)")
    headline("2b'. same, every answer scored as written", SERIES["rcap_reduced.as_is"]["rows"])
headline("2c. Zero-shot on PPatHF's own reduced inputs (stored 2026-09-20)", p_rows, "as 2b")
for arm, tag in (("loop_feedback", "3. Baseline + re-prompt loop k=10 (echo policy: feedback)"),
                 ("loop_retry", "3'. Baseline + re-prompt loop k=10 (echo policy: retry)"),
                 ("ctx_B", "4B. Baseline + ONE re-prompt, errors only"),
                 ("ctx_C", "4C. Baseline + ONE re-prompt, with context of the missing names"),
                 ("ctx_C2", "4C2 (EXPLORATORY, added after 4C was seen). As 4C, no ready-made decline line")):
    if LOOP.get(arm):
        headline(f"{tag}: loop rule (first compiling attempt, else Origin)", embed(arm, "rule"),
                 note=f"subset n={len(LOOP[arm])}; the other {len(cases) - len(LOOP[arm])} cases are step 1 unchanged")
        headline(f"{tag}: last candidate as written", embed(arm, "final"),
                 note=f"subset n={len(LOOP[arm])}")
S["headline"] = head

# ------------------------------------------------------------------ step 1 details
S["step1"] = {"outcomes": dict(collections.Counter(r["outcome"] for r in base.values())),
              "rerun_vs_baseline": paired(b_rows, r_rows), "rerun_vs_baseline_clean": paired(b_rows, r_rows, CLEAN),
              "red_bands": dict(collections.Counter(pm.red_band(r["rel_distance"]) for r in b_rows.values())),
              "src_zero": sum(1 for r in origin.values() if r["src_zero"]),
              "accept_post_hoc": dict(collections.Counter(r["accept"] or "accepted" for r in b_rows.values() if r["has_output"]))}

# ------------------------------------------------------------------ step 2 details
st2 = {}
for arm in ("rcap_full", "rcap_reduced"):
    rows = ARM[arm]
    if not rows:
        continue
    oc = collections.Counter(r["outcome"] + ("/" + r["capacity"] if r["capacity"] else "") for r in rows.values())
    rej = [c for c, r in rows.items() if r["outcome"] in ("placeholder_violation", "duplicated_block")]
    marker = [c for c in rej if rows[c]["outcome"] == "placeholder_violation"]
    kinds = collections.Counter()
    for c in marker:
        rep = rows[c].get("placeholder_report") or {}
        kinds["missing" if rep.get("missing") else ""] += 1
        kinds["duplicated" if rep.get("duplicated") else ""] += 1
        kinds["unknown_or_malformed" if rep.get("unknown") else ""] += 1
    kinds.pop("", None)
    asis = SERIES[arm + ".as_is"]["rows"]
    red = [r["reduction"] for r in rows.values()]
    st2[arm] = {
        "n_done": len(rows), "outcomes": dict(oc),
        "rejected_by_acceptance": len(rej), "placeholder_violation": len(marker), "violation_kinds": dict(kinds),
        "duplicated_block": sum(1 for c in rej if rows[c]["outcome"] == "duplicated_block"),
        "leak_after_expansion": sum(1 for r in rows.values() if r.get("marks_after_expansion")),
        "rejected_as_is": agg(asis, set(rej)) if rej else None,
        "rejected_as_is_bands": dict(collections.Counter(pm.red_band(asis[c]["rel_distance"]) for c in rej)),
        "pipeline_vs_as_is": paired(SERIES[arm]["rows"], asis),
        "vs_baseline": paired(b_rows, SERIES[arm]["rows"]),
        "vs_baseline_both_answered": paired(b_rows, SERIES[arm]["rows"],
                                            {c for c, r in rows.items() if r["outcome"] == "completion" and b_rows[c]["has_output"]}),
        "reduction": {"cases_with_placeholders": sum(1 for x in red if sum(x["placeholders"].values())),
                      "cases_with_target_placeholders": sum(1 for x in red if x["placeholders"]["target"]),
                      "nodes_before": sum(x["nodes_before"] for x in red), "nodes_after": sum(x["nodes_after"] for x in red),
                      "chars_before": sum(x["chars_before"] for x in red), "chars_after": sum(x["chars_after"] for x in red)},
        "prompt_chars": sum(r["prompt_chars"] for r in rows.values()),
        "input_tokens": sum((r["usage"] or {}).get("input_tokens", 0) for r in rows.values()),
        "output_tokens": sum((r["usage"] or {}).get("output_tokens", 0) for r in rows.values()),
        "seconds": sum(r["seconds"] for r in rows.values()),
        "model": sorted({(r["provenance"].get("model"), r["provenance"].get("model_digest"), r["provenance"].get("rcap_commit"))
                         for r in rows.values()}),
    }
if ARM["rcap_full"] and ARM["rcap_reduced"]:
    both = set(ARM["rcap_full"]) & set(ARM["rcap_reduced"])
    answered = {c for c in both if ARM["rcap_full"][c]["outcome"] == "completion" and ARM["rcap_reduced"][c]["outcome"] == "completion"}
    st2["reduced_vs_full"] = {"paired_n": len(both),
                              "pipeline": paired(SERIES["rcap_full"]["rows"], SERIES["rcap_reduced"]["rows"], both),
                              "as_is": paired(SERIES["rcap_full.as_is"]["rows"], SERIES["rcap_reduced.as_is"]["rows"], both),
                              "both_completed": paired(SERIES["rcap_full"]["rows"], SERIES["rcap_reduced"]["rows"], answered),
                              "both_completed_agg": {"full": agg(SERIES["rcap_full"]["rows"], answered),
                                                     "reduced": agg(SERIES["rcap_reduced"]["rows"], answered)},
                              "refused_full_only": sorted(c for c in both if ARM["rcap_full"][c]["outcome"] == "limits_exceeded"
                                                          and ARM["rcap_reduced"][c]["outcome"] != "limits_exceeded"),
                              "refused_reduced_only": sorted(c for c in both if ARM["rcap_reduced"][c]["outcome"] == "limits_exceeded"
                                                             and ARM["rcap_full"][c]["outcome"] != "limits_exceeded"),
                              "prompt_identical": sum(1 for c in both if ARM["rcap_full"][c]["prompt_sha256"] == ARM["rcap_reduced"][c]["prompt_sha256"])}
S["step2"] = st2
S["ppathf_reduced_vs_baseline"] = paired(b_rows, p_rows)

# ------------------------------------------------------------------ steps 3 and 4
if sub:
    S["subset"] = {"reachable": len(sub["reachable"]), "gate": sub["gate"], "attempt1_fails": len(sub["attempt1_fails"]),
                   "no_attempt1": sub["no_attempt1"], "tree_commit": sub["tree_commit"], "flags": sub["flags"],
                   "origin_compiles": sum(1 for c in sub["reachable"] if origin[c]["compiles"]),
                   "baseline_compiles": sum(1 for c in sub["reachable"] if b_rows[c]["compiles"])}
st3 = {}
for arm, rows in LOOP.items():
    if not rows:
        continue
    K = cfg["arms"][arm]["k"]
    scope = set(rows)
    att = SERIES[f"{arm}.attempt"]["rows"]
    a1 = {c: att[f"{c}#a1"] for c in rows}
    curve = []
    for k in range(1, K + 1):
        curve.append({"budget": k,
                      "ran": sum(1 for r in rows.values() if r["attempts_run"] >= k),
                      "passed_by": sum(1 for r in rows.values() if r["first_success_attempt"] and r["first_success_attempt"] <= k),
                      "rule": agg(SERIES[f"{arm}.rule@{k}"]["rows"]), "final": agg(SERIES[f"{arm}.final@{k}"]["rows"]),
                      "best": agg(SERIES[f"{arm}.best@{k}"]["rows"])})
    fin = SERIES[f"{arm}.final@{K}"]["rows"]; rule = SERIES[f"{arm}.rule@{K}"]["rows"]
    # the key question: a re-prompt made the build pass
    fixed = []
    for c, r in rows.items():
        s = r["first_success_attempt"]
        if s and s > 1:
            x = att[f"{c}#a{s}"]
            fixed.append({"case_id": c, "attempt": s, "src_distance": a1[c]["src_distance"],
                          "a1_gen": a1[c]["gen_distance"], "a1_red": a1[c]["rel_distance"], "a1_label": a1[c]["label"],
                          "fix_gen": x["gen_distance"], "fix_red": x["rel_distance"], "fix_exact": x["exact_match"],
                          "fix_label": x["label"], "fix_echo": x["echo"],
                          "direction": "closer" if x["gen_distance"] < a1[c]["gen_distance"] else
                                       "further" if x["gen_distance"] > a1[c]["gen_distance"] else "same"})
    # every case: last candidate vs attempt 1
    moved = collections.Counter()
    for c in rows:
        d = fin[c]["gen_distance"] - a1[c]["gen_distance"]
        moved["closer" if d < 0 else "further" if d > 0 else "same"] += 1
    # cases that end on the pre-patch function
    reverted = [c for c in rows if fin[c].get("code_equal_fork") or fin[c]["echo"]]
    echo_any = [c for c, r in rows.items() if any(a.get("verdict") in ("ECHO", "REVERTED") or a.get("echo") for a in r["attempts"][1:])]
    st3[arm] = {
        "n": len(rows), "k": K, "echo_policy": cfg["arms"][arm].get("echo_policy"),
        "stop_reasons": dict(collections.Counter(r["stop_reason"] for r in rows.values())),
        "attempts_run": dict(collections.Counter(r["attempts_run"] for r in rows.values())),
        "verdicts_by_attempt": {str(k): dict(collections.Counter(a["verdict"] for r in rows.values() for a in r["attempts"] if a["attempt"] == k))
                                for k in range(1, K + 1) if any(r["attempts_run"] >= k for r in rows.values())},
        "generations": sum(max(0, r["attempts_run"] - 1) for r in rows.values()),
        "curve": curve, "fixed": fixed,
        "fixed_summary": dict(collections.Counter(f["direction"] for f in fixed)),
        "fixed_exact": sum(1 for f in fixed if f["fix_exact"]),
        "fixed_agg": ({
            "a1_aed": sum(f["a1_gen"] for f in fixed) / len(fixed), "fix_aed": sum(f["fix_gen"] for f in fixed) / len(fixed),
            "a1_red": sum(f["a1_red"] for f in fixed) / len(fixed), "fix_red": sum(f["fix_red"] for f in fixed) / len(fixed),
            "a1_red_median": sorted(f["a1_red"] for f in fixed)[len(fixed) // 2],
            "fix_red_median": sorted(f["fix_red"] for f in fixed)[len(fixed) // 2],
            "fix_red_lt_1": sum(1 for f in fixed if f["fix_red"] < 1), "fix_red_lt_02": sum(1 for f in fixed if f["fix_red"] < 0.2),
            "fix_is_pre_patch": sum(1 for f in fixed if f["fix_echo"]),
            "sign": pm.sign_test([f["a1_gen"] for f in fixed], [f["fix_gen"] for f in fixed]),
            "max_step_further": max([f["fix_gen"] - f["a1_gen"] for f in fixed if f["fix_gen"] > f["a1_gen"]] or [0]),
        } if fixed else None),
        "fixed_labels": dict(collections.Counter(f["fix_label"] for f in fixed)),
        "final_vs_attempt1_moved": dict(moved),
        "attempt1_as_is": agg(a1), "attempt1_gated": agg({c: origin[c] for c in rows}),
        "rule_vs_attempt1": paired(a1, rule), "final_vs_attempt1": paired(a1, fin),
        "rule_vs_origin": paired({c: origin[c] for c in rows}, rule),
        "ends_on_pre_patch": {"cases": reverted, "n": len(reverted), "final": agg(fin, set(reverted)) if reverted else None,
                              "attempt1": agg(a1, set(reverted)) if reverted else None},
        "any_echo_or_reversion": len(echo_any),
        "declined": [(c, (r.get("needs_cochange") or [{}])[0].get("line")) for c, r in rows.items() if r["stop_reason"] == "needs_cochange"],
        "labels_final": dict(collections.Counter(fin[c]["label"] for c in rows)),
        "labels_attempt1": dict(collections.Counter(a1[c]["label"] for c in rows)),
        "model": sorted({(r["provenance"].get("model"), r["provenance"].get("model_digest"), r["provenance"].get("rcap_commit"))
                         for r in rows.values()}),
    }
S["loops"] = st3
def ctx_compare(other: str) -> dict | None:
    if not (LOOP.get("ctx_B") and LOOP.get(other)):
        return None
    common = set(LOOP["ctx_B"]) & set(LOOP[other])
    Bf, Cf = SERIES["ctx_B.final@2"]["rows"], SERIES[f"{other}.final@2"]["rows"]
    Br, Cr = SERIES["ctx_B.rule@2"]["rows"], SERIES[f"{other}.rule@2"]["rows"]
    pb = [bool(LOOP["ctx_B"][c]["first_success_attempt"]) for c in sorted(common)]
    pc = [bool(LOOP[other][c]["first_success_attempt"]) for c in sorted(common)]
    return {"arm": other, "n": len(common), "pass": pm.mcnemar_exact(pb, pc), "B_pass": sum(pb), "C_pass": sum(pc),
            "final": paired(Bf, Cf, common), "rule": paired(Br, Cr, common),
            "B": {"final": agg(Bf, common), "rule": agg(Br, common)}, "C": {"final": agg(Cf, common), "rule": agg(Cr, common)},
            "B_last": dict(collections.Counter(LOOP["ctx_B"][c]["attempts"][-1]["verdict"] for c in common)),
            "C_last": dict(collections.Counter(LOOP[other][c]["attempts"][-1]["verdict"] for c in common)),
            "v3_used": sum(1 for c in common if any(u.get("template_id") == "rcap-reprompt-v3-context"
                                                    for a in LOOP[other][c]["attempts"] for u in a["units"])),
            "per_case": [{"case_id": c, "B": [a["verdict"] for a in LOOP["ctx_B"][c]["attempts"]][-1],
                          "C": [a["verdict"] for a in LOOP[other][c]["attempts"]][-1],
                          "a1_red": SERIES["ctx_B.attempt"]["rows"][f"{c}#a1"]["rel_distance"],
                          "B_red": Bf[c]["rel_distance"], "C_red": Cf[c]["rel_distance"],
                          "B_exact": Bf[c]["exact_match"], "C_exact": Cf[c]["exact_match"],
                          "B_label": Bf[c]["label"], "C_label": Cf[c]["label"]} for c in sorted(common)]}


S["ctx"] = [x for x in (ctx_compare("ctx_C"), ctx_compare("ctx_C2")) if x]

# ------------------------------------------------------------------ step 5: labels against the ground truth
LABEL_SETS = collections.OrderedDict()
LABEL_SETS["step 1: zero-shot baseline"] = [r for r in b_rows.values() if r["has_output"]]
for arm in ("rcap_full", "rcap_reduced"):
    if arm + ".as_is" in SERIES:
        LABEL_SETS[f"step 2: {arm}, answers as written"] = [r for r in SERIES[arm + ".as_is"]["rows"].values() if r["has_output"]]
for arm in ("loop_feedback", "loop_retry", "ctx_B", "ctx_C", "ctx_C2"):
    if f"{arm}.attempt" in SERIES:
        LABEL_SETS[f"step {'3' if arm.startswith('loop') else '4'}: {arm}, re-prompted attempts (attempt >= 2)"] = [
            r for r in SERIES[f"{arm}.attempt"]["rows"].values() if r["attempt"] >= 2]


def label_tables(rs: list[dict]) -> dict:
    by = collections.defaultdict(lambda: collections.Counter())
    for r in rs:
        by[r["label"]]["n"] += 1
        by[r["label"]]["exact"] += bool(r["raw_exact"])
        by[r["label"]][pm.red_band(r["raw_rel"])] += 1
    n = len(rs)

    def pr(pred, truth):
        tp = sum(1 for r in rs if pred(r) and truth(r)); p = sum(1 for r in rs if pred(r)); t = sum(1 for r in rs if truth(r))
        return {"tp": tp, "predicted": p, "true": t, "precision": tp / p if p else None, "recall": tp / t if t else None}
    faithful = lambda r: r["label"] == fd.FAITHFUL_CODE  # noqa: E731
    near = collections.Counter()
    for r in rs:
        if faithful(r) and not r["raw_exact"]:
            g = r["raw_gen"]
            near["1-2 tokens" if g <= 2 else "3-5 tokens" if g <= 5 else "6-20 tokens" if g <= 20 else "> 20 tokens"] += 1
    return {"n": n, "by_label": {k: dict(v) for k, v in sorted(by.items(), key=lambda kv: -kv[1]["n"])},
            "faithful_not_exact_distance": dict(near),
            "faithful_vs_exact": pr(faithful, lambda r: bool(r["raw_exact"])),
            "faithful_vs_red02": pr(faithful, lambda r: r["raw_rel"] < 0.2),
            "faithful_vs_red1": pr(faithful, lambda r: r["raw_rel"] < 1),
            "bad_label_vs_red_ge1": pr(lambda r: r["label"] in (fd.FORK_REGRESSION, fd.WEAKENED, fd.DUPLICATED_BLOCK, fd.PLACEHOLDER_LEAK),
                                       lambda r: r["raw_rel"] >= 1),
            "nochange_is_echo": pr(lambda r: r["label"] in (fd.NO_CHANGE, "REVERTED", fd.COMMENT_ONLY), lambda r: bool(r["echo"]))}


def compile_tables(rs: list[dict]) -> dict:
    rs = [r for r in rs if r["compiles"] is not None]
    out = {"n": len(rs)}
    for flag in (True, False):
        g = [r for r in rs if r["compiles"] is flag]
        reds = sorted(r["raw_rel"] for r in g)
        out["compiles" if flag else "fails"] = {
            "n": len(g), "exact": sum(1 for r in g if r["raw_exact"]),
            "red_lt_02": sum(1 for r in g if r["raw_rel"] < 0.2), "red_ge_1": sum(1 for r in g if r["raw_rel"] >= 1),
            "red_median": (reds[len(reds) // 2] if len(reds) % 2 else (reds[len(reds) // 2 - 1] + reds[len(reds) // 2]) / 2) if reds else None,
            "red_mean": sum(reds) / len(reds) if reds else None,
            "faithful": sum(1 for r in g if r["label"] == fd.FAITHFUL_CODE),
            "faithful_exact": sum(1 for r in g if r["label"] == fd.FAITHFUL_CODE and r["raw_exact"]),
            "faithful_red02": sum(1 for r in g if r["label"] == fd.FAITHFUL_CODE and r["raw_rel"] < 0.2),
            "faithful_red1": sum(1 for r in g if r["label"] == fd.FAITHFUL_CODE and r["raw_rel"] >= 1),
            "echo": sum(1 for r in g if r["echo"]),
            "labels": dict(collections.Counter(r["label"] for r in g))}
    try:
        from scipy.stats import fisher_exact, mannwhitneyu
        c, f = out["compiles"], out["fails"]
        out["fisher_exact_p"] = float(fisher_exact([[c["exact"], c["n"] - c["exact"]], [f["exact"], f["n"] - f["exact"]]])[1])
        a = [r["raw_rel"] for r in rs if r["compiles"]]; b = [r["raw_rel"] for r in rs if not r["compiles"]]
        out["mannwhitney_red_p"] = float(mannwhitneyu(a, b, alternative="two-sided")[1]) if a and b else None
    except ImportError:
        pass
    return out


# the labels applied to the DEVELOPERS' OWN ports: what a correct port is called
dev_rows = add("developer", {c: nl(k["func_after_target"]) for c, k in cases.items()}, step=5,
               desc="the developer's port itself (ground truth), labelled like any output")
S["labels_on_developer_ports"] = {
    "n": len(dev_rows), "labels": dict(collections.Counter(r["label"] for r in dev_rows.values())),
    "accept": dict(collections.Counter(r["accept"] or "accepted" for r in dev_rows.values())),
    "not_faithful": sorted(c for c, r in dev_rows.items() if r["label"] != fd.FAITHFUL_CODE)[:400]}

def _split(dev_faithful: bool) -> dict:
    cs = [c for c, r in dev_rows.items() if (r["label"] == fd.FAITHFUL_CODE) is dev_faithful]
    out = [b_rows[c] for c in cs if b_rows[c]["has_output"]]
    f = [r for r in out if r["label"] == fd.FAITHFUL_CODE]
    return {"cases": len(cs), "outputs": len(out), "exact": sum(1 for r in out if r["raw_exact"]),
            "labelled_faithful": len(f), "faithful_exact": sum(1 for r in f if r["raw_exact"]),
            "faithful_red02": sum(1 for r in f if r["raw_rel"] < 0.2), "faithful_red_ge1": sum(1 for r in f if r["raw_rel"] >= 1),
            "src_distance_median": sorted(dev_rows[c]["src_distance"] for c in cs)[len(cs) // 2] if cs else None}


S["labels_on_developer_ports"]["baseline_split"] = {"developer_port_faithful": _split(True),
                                                    "developer_port_not_faithful": _split(False)}
S["labels"] = {name: label_tables(rs) for name, rs in LABEL_SETS.items() if rs}
S["compile_vs_truth"] = {name: compile_tables(rs) for name, rs in LABEL_SETS.items() if rs}
pooled, seen = [], set()
for rs in LABEL_SETS.values():
    for r in rs:
        k = (r["case_id"], r["raw_gen"], r["label"], r["raw_exact"])
        if k not in seen:
            seen.add(k); pooled.append(r)
S["labels"]["pooled (distinct outputs of all steps)"] = label_tables(pooled)
S["compile_vs_truth"]["pooled (distinct outputs of all steps)"] = compile_tables(pooled)
# lenient flags (the earlier compile check), for the record
# the earlier compile check's flags (no -Werror=implicit-function-declaration), for the record
if not A.no_compile:
    lenient = Compiler(cfg, strict=False)
    l_reach = [c for c, k in cases.items() if lenient.reachable(c, k)[0]]
    l_fail = [c for c in l_reach if b_rows[c]["has_output"]
              and not lenient.check(c, cases[c], stored_text(base[c]))["ok"]]
    S["lenient"] = {"reachable": len(l_reach), "attempt1_fails": len(l_fail),
                    "reference_needs_undeclared_function": sorted(set(l_reach) - REACH),
                    "fails_only_strict": sorted(set(sub["attempt1_fails"]) - set(l_fail)) if sub else None}
import re as _re
S["ppathf_reduced_markers_left"] = sum(1 for c in cases if (t := recover_ppathf(c, stored_text(ppr.get(c)))) and _re.search(r"placeholder_\d+", t))
S["ppathf_reduced_markers_dropped"] = sum(
    1 for c in cases if (t := stored_text(ppr.get(c)))
    and any(f"/* placeholder_{i} */" not in t for i in sliced[c]["removed_pieces_sliced_target"]))

_cache_file.write_text(json.dumps(_cache))
with (SC / "outputs.jsonl").open("w") as fh:
    for r in ROWS:
        fh.write(json.dumps(r) + "\n")
(SC / "summary.json").write_text(json.dumps(S, indent=1, default=str))

# ------------------------------------------------------------------ tables.md
L: list[str] = []


def f2(x, nd=2):
    return "–" if x is None else f"{x:.{nd}f}"


def acc(a):
    return "–" if not a["n"] else f"{a['exact']}/{a['n']} ({100 * a['accuracy']:.1f}%)"


def pv(p):
    return "–" if p is None else ("<0.001" if p < 0.001 else f"{p:.3f}")


def pair_txt(p):
    m, s, w = p["mcnemar"], p["sign_red"], p["wilcoxon_red"]
    return (f"n={p['n']}: exact +{m['gain']}/−{m['loss']} (McNemar exact p={pv(m['p'])}); "
            f"RED lower in {s['lower']}, higher in {s['higher']}, equal in {s['ties']} (sign test p={pv(s['p'])}; "
            f"Wilcoxon p={pv(w['p'])})")


L += ["## T1. Headline", "",
      f"All {len(cases)} cases, aligned protocol (a missing or incomplete output is the pre-patch function). "
      f"`clean` = without the {len(FLAGGED)} cases the truncation erratum flags. RED median is ours, not the paper's.", "",
      "| step / arm | n | Accuracy | AED | RED (mean) | RED median | scored as Origin | Accuracy, clean n | AED clean | RED clean | paper |",
      "|---|---|---|---|---|---|---|---|---|---|---|"]
for h in head:
    a, c = h["all"], h["clean"]
    L.append(f"| {h['row']}{' — ' + h['note'] if h['note'] else ''} | {a['n']} | {acc(a)} | {f2(a['aed'])} | {f2(a['red'], 3)} | "
             f"{f2(a['red_median'], 3)} | {a['dummy']} | {acc(c)} | {f2(c['aed'])} | {f2(c['red'], 3)} | {h['paper']} |")
L += ["", f"Flagged cases ({len(FLAGGED)}): " + "; ".join(f"`{c}` {w}" for c, w in sorted(FLAGGED.items())), ""]
s1 = S["step1"]
L += ["## T2. Step 1", "", f"- stored outcomes: {s1['outcomes']}; RED bands (310, aligned): {s1['red_bands']}",
      f"- cases where the pre-patch function already equals the developer's on tokens (src_distance = 0): {s1['src_zero']}",
      f"- re-run vs baseline, {pair_txt(s1['rerun_vs_baseline'])}",
      f"- acceptance rules applied post hoc to the stored outputs: {s1['accept_post_hoc']}",
      f"- stored prompt identity: {S['baseline_prompt_identity']}", ""]
if st2:
    L += ["## T3. Step 2: RCAP request, with and without the program cut", ""]
    for arm, d in st2.items():
        if arm == "reduced_vs_full":
            continue
        r = d["reduction"]
        L += [f"**{arm}** (n done {d['n_done']}/{len(cases)}; model/digest/commit {d['model']})", "",
              f"- outcomes: {d['outcomes']}",
              f"- rejected by the acceptance rules: {d['rejected_by_acceptance']} (placeholder_violation {d['placeholder_violation']} "
              f"{d['violation_kinds']}, duplicated_block {d['duplicated_block']}); a marker left after expansion in {d['leak_after_expansion']} answers",
              f"- the rejected answers scored as written: {acc(d['rejected_as_is']) if d['rejected_as_is'] else '–'}, "
              f"AED {f2(d['rejected_as_is']['aed']) if d['rejected_as_is'] else '–'}, RED {f2(d['rejected_as_is']['red'], 3) if d['rejected_as_is'] else '–'} "
              f"(as Origin they are RED 1.000 by construction); bands {d['rejected_as_is_bands']}",
              f"- as written vs rejected = Origin, {pair_txt(d['pipeline_vs_as_is'])}",
              f"- vs stored baseline (all), {pair_txt(d['vs_baseline'])}",
              f"- vs stored baseline (both answered), {pair_txt(d['vs_baseline_both_answered'])}",
              f"- program cut: {r['cases_with_placeholders']} cases with a placeholder ({r['cases_with_target_placeholders']} in the target); "
              f"AST nodes {r['nodes_before']} -> {r['nodes_after']} ({100 * (r['nodes_after'] / r['nodes_before'] - 1):+.1f}%); "
              f"function chars {r['chars_before']} -> {r['chars_after']} ({100 * (r['chars_after'] / r['chars_before'] - 1):+.1f}%)",
              f"- request chars {d['prompt_chars']}; input tokens {d['input_tokens']}; output tokens {d['output_tokens']}; wall {d['seconds'] / 3600:.2f} h", ""]
    if "reduced_vs_full" in st2:
        d = st2["reduced_vs_full"]
        L += [f"**reduced vs full** (paired n={d['paired_n']}; {d['prompt_identical']} cases have a byte-identical request in both arms)", "",
              f"- rejected / refused = Origin, {pair_txt(d['pipeline'])}",
              f"- as written, {pair_txt(d['as_is'])}",
              f"- cases both arms completed, {pair_txt(d['both_completed'])}; full {acc(d['both_completed_agg']['full'])} AED "
              f"{f2(d['both_completed_agg']['full']['aed'])} RED {f2(d['both_completed_agg']['full']['red'], 3)}; reduced "
              f"{acc(d['both_completed_agg']['reduced'])} AED {f2(d['both_completed_agg']['reduced']['aed'])} RED {f2(d['both_completed_agg']['reduced']['red'], 3)}",
              f"- capacity: refused without the cut and answered with it: {len(d['refused_full_only'])}; the reverse: {len(d['refused_reduced_only'])}", ""]
L += [f"Stored zero-shot on PPatHF's own reduced inputs vs stored baseline, {pair_txt(S['ppathf_reduced_vs_baseline'])}; "
      f"answers that dropped or rewrote one of THEIR placeholders (the elided code is then lost): {S['ppathf_reduced_markers_dropped']}; "
      f"with one left after recovery: {S['ppathf_reduced_markers_left']}", ""]
if sub:
    u = S["subset"]
    L += ["## T4. The compile check and the subset", "",
          f"- gate (the developer's function compiles in tree {u['tree_commit'][:10]}): {u['gate']} -> reachable {u['reachable']}",
          f"- of the reachable: the pre-patch function compiles in {u['origin_compiles']}, the stored attempt 1 in {u['baseline_compiles']}, "
          f"fails in {u['attempt1_fails']}; no attempt 1 (refused / incomplete): {len(u['no_attempt1'])}", f"- flags added: {u['flags']}",
          f"- with the earlier check's flags (implicit function declarations only warn): {S.get('lenient')}", ""]
for arm, d in st3.items():
    L += [f"## T5. {arm} (n={d['n']}, k={d['k']}, echo policy {d['echo_policy']})", "",
          f"- stop reasons: {d['stop_reasons']}; attempts run: {d['attempts_run']}; repair generations: {d['generations']}",
          f"- verdicts by attempt: {d['verdicts_by_attempt']}",
          f"- attempt 1 as written: {acc(d['attempt1_as_is'])}, AED {f2(d['attempt1_as_is']['aed'])}, RED {f2(d['attempt1_as_is']['red'], 3)} "
          f"(median {f2(d['attempt1_as_is']['red_median'], 3)}); attempt 1 under the loop's rule (none compiles = Origin): AED {f2(d['attempt1_gated']['aed'])}, RED 1.000",
          "", "| budget k | still running at k | passed by k | RULE: Accuracy | AED | RED | FINAL as written: Accuracy | AED | RED | RED median | BEST so far (oracle): Accuracy | AED | RED |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for c in d["curve"]:
        L.append(f"| {c['budget']} | {c['ran']} | {c['passed_by']} | {acc(c['rule'])} | {f2(c['rule']['aed'])} | {f2(c['rule']['red'], 3)} | "
                 f"{acc(c['final'])} | {f2(c['final']['aed'])} | {f2(c['final']['red'], 3)} | {f2(c['final']['red_median'], 3)} | "
                 f"{acc(c['best'])} | {f2(c['best']['aed'])} | {f2(c['best']['red'], 3)} |")
    L += ["", f"- rule@k vs attempt 1 as written, {pair_txt(d['rule_vs_attempt1'])}",
          f"- final@k vs attempt 1 as written, {pair_txt(d['final_vs_attempt1'])}",
          f"- rule@k vs Origin, {pair_txt(d['rule_vs_origin'])}",
          f"- last candidate vs attempt 1 (token distance to the developer's port): {d['final_vs_attempt1_moved']}",
          f"- cases with an echo / reversion at some re-prompt: {d['any_echo_or_reversion']}; cases whose LAST candidate is the pre-patch function: "
          f"{d['ends_on_pre_patch']['n']} {d['ends_on_pre_patch']['cases']}"
          + (f"; their attempt 1 was {acc(d['ends_on_pre_patch']['attempt1'])}, AED {f2(d['ends_on_pre_patch']['attempt1']['aed'])}, "
             f"RED {f2(d['ends_on_pre_patch']['attempt1']['red'], 3)}; their last candidate RED {f2(d['ends_on_pre_patch']['final']['red'], 3)}"
             if d['ends_on_pre_patch']['n'] else ""),
          f"- declines (CANNOT_ADAPT): {d['declined']}",
          f"- labels, attempt 1: {d['labels_attempt1']}; last candidate: {d['labels_final']}", "",
          f"**A re-prompt made the build pass in {len(d['fixed'])} cases**: {d['fixed_summary']}; exact {d['fixed_exact']}; labels {d['fixed_labels']}", ""]
    if d["fixed"]:
        g = d["fixed_agg"]
        L += [f"On these {len(d['fixed'])}: token distance to the developer's port, mean {f2(g['a1_aed'])} at attempt 1 -> {f2(g['fix_aed'])} at the passing attempt; "
              f"RED mean {f2(g['a1_red'], 3)} -> {f2(g['fix_red'], 3)}, median {f2(g['a1_red_median'], 3)} -> {f2(g['fix_red_median'], 3)}; "
              f"sign test on the distance p={pv(g['sign']['p'])}; the passing attempt has RED < 1 in {g['fix_red_lt_1']}, RED < 0.2 in {g['fix_red_lt_02']}, "
              f"is the pre-patch function in {g['fix_is_pre_patch']}; the largest step AWAY from the developer's port is {g['max_step_further']} tokens.", ""]
        L += ["| case | passes at attempt | dev. change (tokens) | attempt 1: distance / RED / label | passing attempt: distance / RED / label | exact | direction |",
              "|---|---|---|---|---|---|---|"]
        for f in d["fixed"]:
            L.append(f"| {f['case_id']} | {f['attempt']} | {f['src_distance']} | {f['a1_gen']} / {f2(f['a1_red'], 3)} / {f['a1_label']} | "
                     f"{f['fix_gen']} / {f2(f['fix_red'], 3)} / {f['fix_label']} | {'yes' if f['fix_exact'] else 'no'} | {f['direction']} |")
        L.append("")
for d in S["ctx"]:
    X = d["arm"].replace("ctx_", "")
    L += [f"## T6. One re-prompt: errors only (B) vs with context ({X}{', EXPLORATORY' if X != 'C' else ''}), n={d['n']}", "",
          f"- compiles after the re-prompt: B {d['B_pass']}, {X} {d['C_pass']} (McNemar exact: {X} only {d['pass']['gain']}, B only {d['pass']['loss']}, p={pv(d['pass']['p'])}); "
          f"the v3 request was used in {d['v3_used']} of {d['n']} {X} cases (a case with no name to describe gets the v2 request)",
          f"- outcome of the re-prompt: B {d['B_last']}; {X} {d['C_last']}",
          "", "| arm | RULE: Accuracy | AED | RED | LAST as written: Accuracy | AED | RED | RED median |", "|---|---|---|---|---|---|---|---|"]
    for x, name in (("B", "B"), ("C", X)):
        L.append(f"| {name} | {acc(d[x]['rule'])} | {f2(d[x]['rule']['aed'])} | {f2(d[x]['rule']['red'], 3)} | {acc(d[x]['final'])} | "
                 f"{f2(d[x]['final']['aed'])} | {f2(d[x]['final']['red'], 3)} | {f2(d[x]['final']['red_median'], 3)} |")
    L += ["", f"- {X} vs B, rule: {pair_txt(d['rule'])}", f"- {X} vs B, last as written: {pair_txt(d['final'])}", "",
          f"| case | attempt 1 RED | B: verdict / RED / label | {X}: verdict / RED / label |", "|---|---|---|---|"]
    for x in d["per_case"]:
        L.append(f"| {x['case_id']} | {f2(x['a1_red'], 3)} | {x['B']} / {f2(x['B_red'], 3)}{' exact' if x['B_exact'] else ''} / {x['B_label']} | "
                 f"{x['C']} / {f2(x['C_red'], 3)}{' exact' if x['C_exact'] else ''} / {x['C_label']} |")
    L.append("")
L += ["## T7. Labels against the ground truth", ""]
dl = S["labels_on_developer_ports"]
L += [f"**The developers' own ports, labelled like any output** (n={dl['n']}): {dl['labels']}; acceptance rules: {dl['accept']}. "
      "A label other than FAITHFUL_CODE here means: the developer did NOT reproduce the upstream change as it stands "
      "(adapted it, ported part of it), or the label is wrong.", "",
      "| cases where the developer's own port is ... | cases | step-1 outputs | exact | labelled FAITHFUL_CODE | of those: exact | RED < 0.2 | RED >= 1 | median size of the developer's change (tokens) |",
      "|---|---|---|---|---|---|---|---|---|"]
for name, key in (("labelled FAITHFUL_CODE", "developer_port_faithful"), ("labelled anything else", "developer_port_not_faithful")):
    x = dl["baseline_split"][key]
    L.append(f"| {name} | {x['cases']} | {x['outputs']} | {x['exact']} | {x['labelled_faithful']} | {x['faithful_exact']} | "
             f"{x['faithful_red02']} | {x['faithful_red_ge1']} | {x['src_distance_median']} |")
L.append("")
for name, d in S["labels"].items():
    L += [f"**{name}** (n={d['n']} outputs)", "", "| label | n | exact | " + " | ".join(pm.RED_BANDS) + " |", "|---|---|---|---|---|---|---|"]
    for lab, v in d["by_label"].items():
        L.append(f"| {lab} | {v['n']} | {v.get('exact', 0)} | " + " | ".join(str(v.get(b, 0)) for b in pm.RED_BANDS) + " |")
    for key, txt in (("faithful_vs_exact", "FAITHFUL_CODE -> exact match"), ("faithful_vs_red02", "FAITHFUL_CODE -> RED < 0.2"),
                     ("faithful_vs_red1", "FAITHFUL_CODE -> RED < 1 (closer than doing nothing)"),
                     ("bad_label_vs_red_ge1", "FORK_REGRESSION / WEAKENED / DUPLICATED_BLOCK / PLACEHOLDER_LEAK -> RED >= 1"),
                     ("nochange_is_echo", "NO_CHANGE / REVERTED / COMMENT_ONLY -> output equals the pre-patch function on tokens")):
        p = d[key]
        L.append(f"- {txt}: precision {p['tp']}/{p['predicted']} = {f2(p['precision'], 3)}, recall {p['tp']}/{p['true']} = {f2(p['recall'], 3)}")
    L.append(f"- FAITHFUL_CODE outputs that are NOT exact, by token distance to the developer's port: {d['faithful_not_exact_distance']}")
    L.append("")
L += ["## T8. Does `compiles` predict closeness (reachable cases only)", "",
      "| outputs | compiles: n | exact | RED < 0.2 | RED >= 1 | RED median | compiles AND FAITHFUL_CODE: n (exact / RED < 0.2 / RED >= 1) | fails: n | exact | RED < 0.2 | RED >= 1 | RED median | Fisher p (exact) | Mann-Whitney p (RED) |",
      "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
for name, d in S["compile_vs_truth"].items():
    if not d["n"]:
        continue
    c, f = d["compiles"], d["fails"]
    L.append(f"| {name} | {c['n']} | {c['exact']} | {c['red_lt_02']} | {c['red_ge_1']} | {f2(c['red_median'], 3)} | {c['faithful']} ({c['faithful_exact']} / {c['faithful_red02']} / {c['faithful_red1']}) | "
             f"{f['n']} | {f['exact']} | {f['red_lt_02']} | {f['red_ge_1']} | {f2(f['red_median'], 3)} | {pv(d.get('fisher_exact_p'))} | {pv(d.get('mannwhitney_red_p'))} |")
L.append("")
(OUT / "tables.md").write_text("\n".join(L) + "\n")
print(f"scored {len(ROWS)} rows, {len(SERIES)} series -> {OUT / 'tables.md'}")

# ------------------------------------------------------------------ REPORT.md
# REPORT.prose.md (hand-written, in the results directory) holds the text; the
# markers <!--T1--> .. <!--T8-->, <!--STATUS--> and <!--STEP2--> are filled from
# the numbers above, so the report can be re-rendered when a running arm ends.
import datetime
import re

prose = OUT / "REPORT.prose.md"
if prose.exists():
    sections: dict[str, list[str]] = {}
    cur = None
    for line in L:
        m = re.match(r"^## (T\d)\. (.*)$", line)
        if m:
            cur = m.group(1)
            sections.setdefault(cur, []).append(f"**{m.group(2)}**")
            continue
        if cur:
            sections[cur].append(line)
    n_arm = {arm: len(ARM[arm]) for arm in ARM}
    n_loop = {arm: len(LOOP[arm]) for arm in LOOP}
    want_loop = len(sub["attempt1_fails"]) if sub else 0
    pending = [f"{a} {n}/{len(cases)}" for a, n in n_arm.items() if n < len(cases)] + \
              [f"{a} {n}/{want_loop}" for a, n in n_loop.items() if n < want_loop]
    status = (f"Rendered {datetime.datetime.now().isoformat(timespec='seconds')} by `tools/vimnvim/report.py`. "
              f"Arms complete: " + ", ".join([f"{a} {n}/{len(cases)}" for a, n in n_arm.items() if n == len(cases)]
                                             + [f"{a} {n}/{want_loop}" for a, n in n_loop.items() if n == want_loop and n]) + ". "
              + ("**Still running or not started: " + ", ".join(pending) + ".** Every number of an incomplete arm below is over the cases "
                 "it has finished and will change; re-render with the command in section 12." if pending else "All arms are complete."))

    def step2_text() -> str:
        if not st2:
            return "_Step 2 has not produced a row yet (the chain runs it after steps 3 and 4)._"
        out = []
        for arm in ("rcap_full", "rcap_reduced"):
            d = st2.get(arm)
            if not d:
                out.append(f"- `{arm}`: not started.")
                continue
            a = agg(SERIES[arm]["rows"]); x = agg(SERIES[arm + ".as_is"]["rows"])
            cap = sum(v for k, v in d["outcomes"].items() if k.startswith("limits_exceeded"))
            out.append(
                f"- `{arm}` ({d['n_done']}/{len(cases)} cases): {acc(a)} exact, AED {f2(a['aed'])}, RED {f2(a['red'], 3)} "
                f"(median {f2(a['red_median'], 3)}) with rejected and refused answers left as the fork's code; "
                f"{acc(x)}, AED {f2(x['aed'])}, RED {f2(x['red'], 3)} with every answer scored as written. "
                f"Capacity refusals: {cap}. The acceptance rules reject {d['rejected_by_acceptance']} answers "
                f"(placeholder {d['placeholder_violation']}, duplicated block {d['duplicated_block']}); scored as written those "
                f"rejected answers are {acc(d['rejected_as_is']) if d['rejected_as_is'] else '–'} exact with RED "
                f"{f2(d['rejected_as_is']['red'], 3) if d['rejected_as_is'] else '–'} against 1.000 as Origin.")
        d = st2.get("reduced_vs_full")
        if d:
            m, b = d["pipeline"]["mcnemar"], d["both_completed"]["mcnemar"]
            verdict = ("no significant difference" if m["p"] >= 0.05 else
                       ("the program cut is significantly BETTER" if m["gain"] > m["loss"] else "the program cut is significantly WORSE"))
            out.append(
                f"- reduced vs full, paired on {d['paired_n']} cases: exact +{m['gain']}/−{m['loss']}, McNemar exact p={pv(m['p'])} "
                f"({verdict} at 0.05); on the {d['both_completed']['n']} cases both arms completed: +{b['gain']}/−{b['loss']}, p={pv(b['p'])}. "
                f"Capacity: {len(d['refused_full_only'])} cases are refused without the cut and answered with it, "
                f"{len(d['refused_reduced_only'])} the other way round. {d['prompt_identical']} requests are byte-identical in both arms "
                "(nothing to cut).")
        return "\n".join(out)

    text = prose.read_text()
    text = re.sub(r"<!--(T\d)-->", lambda m: "\n".join(sections.get(m.group(1), ["_not available yet_"])).rstrip(), text)
    text = text.replace("<!--STATUS-->", status).replace("<!--STEP2-->", step2_text())
    (OUT / "REPORT.md").write_text(text)
    print("wrote", OUT / "REPORT.md")
