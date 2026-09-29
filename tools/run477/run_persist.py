#!/usr/bin/env python3
"""Regenerate RCAP candidates for the MO completions of the 477 run and KEEP them.

Same code path and backend settings as run_477.py (deterministic at fixed seed),
restricted to SAPs relabelled MO whose outcome was `completion`. Per SAP it writes
candidates/<key>.json with, per unit: the fork file, the ORIGINAL target text
(placeholders recovered), the raw candidate and the candidate with its
placeholders expanded from the same recovery map -- everything the splice +
build + test step needs. Resumable by key.
"""
import json, sys, time, traceback, hashlib
from pathlib import Path
sys.path.insert(0, "/home/adam/Documents/work/rcap-class-project/rcap/src")
from rcap.backend_ollama import OllamaBackend
from rcap.pipeline import run_case
from rcap.reduction_program import recover, PLACEHOLDER

RUN = Path("/home/adam/Documents/work/rcap-class-project/run477")
OUT = RUN / "validate/candidates"; OUT.mkdir(parents=True, exist_ok=True)
lab = {json.loads(l)["key"]: json.loads(l) for l in open(RUN / "mo-relabel/labels_final.jsonl")}
keys = [k for k, v in lab.items() if v["label"] == "MO" and v["outcome"] == "completion"]
# linkedin/kafka targets first: 738 of 789, and the one fork with a validation baseline ready
keys.sort(key=lambda k: (not k.startswith("linkedinKafka-apacheKafka/"), k))
if "--limit" in sys.argv: keys = keys[:int(sys.argv[sys.argv.index("--limit") + 1])]
fname = lambda k: OUT / (k.replace("/", "__") + ".json")
todo = [k for k in keys if not fname(k).exists()]
backend = OllamaBackend()
print(f"backend {backend.model} digest={backend.model_digest}; {len(keys)} MO completions, {len(todo)} to run", flush=True)

def expand(candidate, art):
    for e in art.placeholders:
        candidate = candidate.replace(PLACEHOLDER.format(n=e.ph_id), e.original_text, 1)
    return candidate

t0 = time.time()
for i, key in enumerate(todo, 1):
    sap_dir = RUN / "salp-477-out" / key
    manifest = json.loads((sap_dir.parent / "pr.json").read_text())
    sap = json.loads((sap_dir / "sap.json").read_text())
    rec = {"key": key, "target_repo": manifest["target_repo"], "target_file": sap["target_file"],
           "source_file": sap["source_file"], "units": []}
    t = time.time()
    try:
        res = run_case(sap_dir, manifest, backend)
        rec["outcome"] = ("completion" if all(u.generation.outcome == "completion" for u in res.units)
                          else next(u.generation.outcome for u in res.units if u.generation.outcome != "completion"))
        for u in res.units:
            ent = u.context.target_localization.get("function")
            art = next((a for a in u.context.program_context if a.role == "target" and a.entity == ent), None)
            cand = u.generation.candidate
            rec["units"].append({
                "entity": ent, "outcome": u.generation.outcome,
                "prompt_sha256": hashlib.sha256(u.request.prompt.encode()).hexdigest(),
                "target_reduced": u.context.transformation.get("target"),
                "target_original": recover(art) if art else None,
                "candidate_raw": cand,
                "candidate_expanded": expand(cand, art) if (cand and art) else cand,
                "placeholders": [e.ph_id for e in art.placeholders] if art else [],
                "usage": getattr(u.generation, "usage", None)})
    except Exception as exc:
        rec["outcome"] = f"REFUSED:{type(exc).__name__}"; rec["error"] = traceback.format_exc()[-600:]
    rec["seconds"] = round(time.time() - t, 1)
    fname(key).write_text(json.dumps(rec, indent=1))
    el = time.time() - t0
    print(f"[{i}/{len(todo)}] {key}: {rec['outcome']} ({rec['seconds']}s) | eta {el/i*(len(todo)-i)/3600:.1f}h", flush=True)
print("ALL DONE", flush=True)
