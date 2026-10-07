#!/usr/bin/env python3
"""Evaluation C: the single placeholder retry on SAPs the 477 run rejected as placeholder_violation.

Per SAP: the normal pipeline with run_case(placeholder_retry=True). Attempt 1 is regenerated (the
477 run did not keep rejected answers), then ONE repair request per violating unit. Writes
<out-dir>/candidates/<key>.json in run_persist.py's format (so revalidate.py can validate the new
completions) plus the retry record per unit, and one summary row per SAP in <out-dir>/retry.jsonl.

  placeholder_retry.py --keys keys.txt --out-dir DIR [--gen-cache DIR] [--salp-out DIR]
Resumable by key.
"""
import argparse, hashlib, json, sys, time, traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from gencache import CachedBackend  # noqa: E402
from rcap.backend_ollama import OllamaBackend  # noqa: E402
from rcap.pipeline import run_case  # noqa: E402
from rcap.provenance import run_provenance  # noqa: E402
from rcap.reduction_program import PLACEHOLDER, recover  # noqa: E402

RUN = Path("/home/adam/Documents/work/rcap-class-project/run477")
ap = argparse.ArgumentParser()
ap.add_argument("--keys", required=True); ap.add_argument("--out-dir", required=True)
ap.add_argument("--gen-cache"); ap.add_argument("--salp-out", default=str(RUN / "salp-477-out"))
ap.add_argument("--limit", type=int)
a = ap.parse_args()
OUTD = Path(a.out_dir); CAND = OUTD / "candidates"; CAND.mkdir(parents=True, exist_ok=True)
OUT = OUTD / "retry.jsonl"
keys = [k.strip() for k in Path(a.keys).read_text().splitlines() if k.strip()]
done = {json.loads(l)["key"] for l in open(OUT)} if OUT.exists() else set()
todo = [k for k in keys if k not in done]
if a.limit: todo = todo[:a.limit]
backend = CachedBackend(OllamaBackend(), a.gen_cache or OUTD / "gen-cache")
PROV = run_provenance(backend, driver="placeholder_retry.py", k=1, placeholder_retry=True)
print(f"backend {backend.model} digest={backend.model_digest} params={backend.params}; {len(todo)} to run ({len(done)} done)", flush=True)

def expand(c, art):
    for e in (art.placeholders if art else []):
        c = c.replace(PLACEHOLDER.format(n=e.ph_id), e.original_text, 1)
    return c

t0 = time.time()
for i, key in enumerate(todo, 1):
    sdir = Path(a.salp_out) / key; t = time.time()
    rec = {"key": key, "units": [], "provenance": PROV}; row = {"key": key, "provenance": PROV}
    try:
        manifest = json.loads((sdir.parent / "pr.json").read_text()); sap = json.loads((sdir / "sap.json").read_text())
        rec.update(target_repo=manifest["target_repo"], target_file=sap["target_file"], source_file=sap["source_file"])
        backend.sources.clear()
        res = run_case(sdir, manifest, backend, placeholder_retry=True)
        outs = [u.generation.outcome for u in res.units]
        rec["outcome"] = "completion" if all(o == "completion" for o in outs) else next(o for o in outs if o != "completion")
        for u in res.units:
            ent = u.context.target_localization.get("function")
            art = next((x for x in u.context.program_context if x.role == "target" and x.entity == ent), None)
            cand = u.generation.candidate
            rec["units"].append({
                "entity": ent, "outcome": u.generation.outcome, "diagnostics": list(u.generation.diagnostics)[:3],
                "prompt_sha256": hashlib.sha256(u.request.prompt.encode()).hexdigest(), "prompt_chars": len(u.request.prompt),
                "target_original": recover(art) if art else None, "candidate_raw": cand,
                "candidate_expanded": expand(cand, art) if cand else cand,
                "placeholders": [e.ph_id for e in art.placeholders] if art else [],
                "usage": dict(u.generation.usage), "placeholder_retry": u.placeholder_retry})
        retried = [u["placeholder_retry"] for u in rec["units"] if u["placeholder_retry"]]
        row.update(outcome=rec["outcome"], units=len(outs), unit_outcomes=outs, retries=len(retried),
                   retries_fixed=sum(1 for r in retried if r["fixed"]),
                   retry_outcomes=[r["retry_outcome"] for r in retried],
                   first_reports=[r["report"] for r in retried], sources=dict(backend.sources))
    except Exception as exc:
        rec["outcome"] = row["outcome"] = f"REFUSED:{type(exc).__name__}"; rec["error"] = traceback.format_exc()[-600:]
    row["seconds"] = rec["seconds"] = round(time.time() - t, 1)
    (CAND / (key.replace("/", "__") + ".json")).write_text(json.dumps(rec, indent=1))
    with OUT.open("a") as fh: fh.write(json.dumps(row) + "\n")
    el = time.time() - t0
    print(f"[{i}/{len(todo)}] {key}: {row['outcome']} retries={row.get('retries')} fixed={row.get('retries_fixed')} "
          f"({row['seconds']}s) | eta {el / i * (len(todo) - i) / 3600:.1f}h", flush=True)
print("ALL DONE", flush=True)
