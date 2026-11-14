"""The candidates of a run_477.py RCAP run, recovered without a model.

run_477.py rows record outcomes and sizes, not the candidate text; the answers live in the generation cache
(by prompt hash). This replays rcap.pipeline.run_case for every SAP of the rows file with the same overrides
(edit evidence, relevance filter, unit allowlist), num_ctx and mode, answering every request from the cache
only (RCAP_CACHE_ONLY=1; a miss is reported as outcome `cache_miss`, never sent to a model: the inner backend
points at port 9). For every unit sent it returns the expanded candidate (RCAP's placeholders put back, as
order_pr.py does before splicing) and the fork's original function.

    from rcap_candidates import replay
    replay(rows.jsonl, sap_root, [gen-cache dirs], "full_rcap", overrides_file, edit_evidence=None)
"""
from __future__ import annotations

import collections
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "tools" / "run477"))


def replay(rows_path: Path, sap_root: Path, gen_caches, mode: str, overrides=None, edit_evidence=None,
           log=print):
    os.environ["RCAP_CACHE_ONLY"] = "1"
    from gencache import CachedBackend  # noqa: E402
    from rcap import runoverrides  # noqa: E402
    from rcap.backend_ollama import OllamaBackend  # noqa: E402
    from rcap.config import EvaluationMode, ExecutionConfig  # noqa: E402
    from rcap.pipeline import run_case  # noqa: E402
    from rcap.reduction_program import PLACEHOLDER, recover  # noqa: E402

    if not gen_caches:
        raise SystemExit("rcap configs need --gen-cache (the run's answer cache)")
    ov = runoverrides.load(overrides or (HERE / "cand32k-overrides.json"))
    if edit_evidence is not None:
        ov.edit_evidence = edit_evidence
    cfg = ov.apply(ExecutionConfig(mode=EvaluationMode(mode)))
    rows = [json.loads(l) for l in open(rows_path) if l.strip()]
    variants = collections.Counter(r.get("variant") for r in rows)
    if ov.variant and set(variants) - {ov.variant, None}:
        log(f"WARNING: rows carry variant(s) {dict(variants)} but the replay runs as {ov.variant}: "
            "prompts may differ, expect cache misses")
    num_ctx = next((r["provenance"].get("num_ctx") for r in rows if r.get("provenance")), None) or 32768
    backends = {}
    out = {}
    stats = collections.Counter()
    for r in rows:
        if r.get("outcome") == "not_in_allowlist" or str(r.get("outcome", "")).startswith("REFUSED"):
            continue
        model = (r.get("provenance") or {}).get("model", "qwen3-coder:30b")
        if model not in backends:
            inner = OllamaBackend(model=model, url="http://127.0.0.1:9", num_ctx=num_ctx)   # never called
            backends[model] = CachedBackend(inner, gen_caches[0], gen_caches[1:])
        be = backends[model]
        pr_dir = sap_root / r["pair"] / r["pr"]
        manifest = json.loads((pr_dir / "pr.json").read_text())
        allowed = ov.units_for(r["pair"], r["pr"], r["sap"])
        try:
            res = run_case(pr_dir / r["sap"], manifest, be, cfg, generate_units=allowed)
        except Exception as exc:  # recorded per unit below as the SAP's refusal
            stats[f"sap_refused:{type(exc).__name__}"] += 1
            continue
        for u in res.units:
            if u.generation.outcome == "not_in_allowlist":
                continue
            ent = u.context.target_localization.get("function") or (u.unit.entity if u.unit is not None else "")
            fn = (ent or "").split("/")[-1]
            key = f"{r['pair']}/{r['pr']}/{r['sap']}::{fn}"
            art = next((x for x in u.context.program_context if x.role == "target" and x.entity == ent), None)
            target = recover(art) if art is not None else None
            outcome = u.generation.outcome
            cand = None
            if outcome == "completion":
                cand = u.generation.candidate or ""
                for e in (art.placeholders if art else []):
                    cand = cand.replace(PLACEHOLDER.format(n=e.ph_id), e.original_text, 1)
            elif any("cache-only" in str(d) for d in (u.generation.diagnostics or [])):
                outcome = "cache_miss"
            stats[outcome] += 1
            out[key] = {"candidate": cand, "outcome": outcome, "target": target}
    log(f"replayed {len(out)} units: {dict(stats)}; answer sources {dict(sum((b.sources for b in backends.values()), collections.Counter()))}")
    return out
