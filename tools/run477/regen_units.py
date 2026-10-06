#!/usr/bin/env python3
"""Regenerate the candidates of given SAPs with the CURRENT pipeline and keep them.

Same code path and backend settings as run_persist.py (the 477 run), for a list of keys instead of
all MO completions, and with what the unit-kind fix decides recorded per unit: the kinds of the
payloads, whether a whole-file target was narrowed to the source function's counterpart, the
request template and the generation outcome (including `shape_mismatch`).

  regen_units.py --keys FILE --out-dir DIR [--salp-out DIR] [--gen-cache DIR] [--no-narrow]

`FILE`: one key per line, or JSON ({"keys": [...]} or a list). Writes DIR/candidates/<key>.json in
run_persist.py's format (revalidate.py reads it with --candidates) and one row per SAP to
DIR/gen.jsonl. Resumable by candidate file. Nothing else is written.
"""
import argparse, hashlib, json, sys, time, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gencache import CachedBackend  # noqa: E402
from rcap.backend_ollama import OllamaBackend  # noqa: E402
from rcap.pipeline import run_case  # noqa: E402
from rcap.reduction_program import PLACEHOLDER, recover  # noqa: E402
from rcap.request import unit_kinds  # noqa: E402

RUN = Path("/home/adam/Documents/work/rcap-class-project/run477")
ap = argparse.ArgumentParser()
ap.add_argument("--keys", required=True); ap.add_argument("--out-dir", required=True)
ap.add_argument("--salp-out", default=str(RUN / "salp-477-out"))
ap.add_argument("--gen-cache"); ap.add_argument("--no-narrow", action="store_true")
ap.add_argument("--limit", type=int)
a = ap.parse_args()

raw = Path(a.keys).read_text()
try:
    keys = json.loads(raw); keys = keys["keys"] if isinstance(keys, dict) else keys
except ValueError:
    keys = [k.strip() for k in raw.splitlines() if k.strip()]
OUT = Path(a.out_dir); CAND = OUT / "candidates"; CAND.mkdir(parents=True, exist_ok=True)
fname = lambda k: CAND / (k.replace("/", "__") + ".json")
todo = [k for k in keys if not fname(k).exists()]
if a.limit: todo = todo[:a.limit]
backend = OllamaBackend()
num_ctx = backend.params.get("num_ctx")
if a.gen_cache: backend = CachedBackend(backend, a.gen_cache)
print(f"backend {backend.model} digest={backend.model_digest} params={backend.params}; "
      f"{len(todo)} to run ({len(keys) - len(todo)} done); narrow={not a.no_narrow}", flush=True)


def expand(candidate, art):
    for e in art.placeholders:
        candidate = candidate.replace(PLACEHOLDER.format(n=e.ph_id), e.original_text, 1)
    return candidate


for i, key in enumerate(todo, 1):
    sap_dir = Path(a.salp_out) / key; t = time.time()
    manifest = json.loads((sap_dir.parent / "pr.json").read_text())
    sap = json.loads((sap_dir / "sap.json").read_text())
    rec = {"key": key, "target_repo": manifest["target_repo"], "target_file": sap["target_file"],
           "source_file": sap["source_file"], "units": []}
    try:
        res = run_case(sap_dir, manifest, backend, narrow_whole_file_targets=not a.no_narrow)
        rec["outcome"] = next((u.generation.outcome for u in res.units if u.generation.outcome != "completion"), "completion")
        for u in res.units:
            ent = u.context.target_localization.get("function")
            art = next((x for x in u.context.program_context if x.role == "target" and x.entity == ent), None)
            cand = u.generation.candidate; usage = dict(u.generation.usage or {})
            total = usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
            rec["units"].append({
                "entity": ent, "outcome": u.generation.outcome, "diagnostics": list(u.generation.diagnostics)[:3],
                "kinds": dict(zip(("source_before", "source_after", "target"), unit_kinds(u.context))),
                "narrowing": u.context.target_localization.get("narrowing"),
                "template_id": u.request.template_id, "answer_kind": u.request.answer_kind,
                "prompt_chars": len(u.request.prompt),
                "prompt_sha256": hashlib.sha256(u.request.prompt.encode()).hexdigest(),
                "target_reduced": u.context.transformation.get("target"),
                "target_original": recover(art) if art else None,
                "candidate_raw": cand,
                "candidate_expanded": expand(cand, art) if (cand and art) else cand,
                "placeholders": [e.ph_id for e in art.placeholders] if art else [],
                "usage": usage,
                # input + output beyond the window means the runtime shifted the context while
                # answering: recorded as a flag, not enforced
                "tokens_exceed_window": bool(num_ctx and total > num_ctx)})
    except Exception as exc:
        rec["outcome"] = f"REFUSED:{type(exc).__name__}"; rec["error"] = traceback.format_exc()[-600:]
    rec["seconds"] = round(time.time() - t, 1)
    fname(key).write_text(json.dumps(rec, indent=1))
    with (OUT / "gen.jsonl").open("a") as fh:
        fh.write(json.dumps({k: v for k, v in rec.items() if k != "units"} | {"units": [
            {k: v for k, v in u.items() if k not in ("target_reduced", "target_original", "candidate_raw", "candidate_expanded")}
            for u in rec["units"]]}) + "\n")
    print(f"[{i}/{len(todo)}] {key}: {rec['outcome']} "
          + " ".join(f"{u['kinds']['target']}/{u['template_id']}/{u['outcome']}" for u in rec["units"])
          + f" ({rec['seconds']}s)", flush=True)
print("ALL DONE", flush=True)
