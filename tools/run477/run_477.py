"""RCAP end-to-end over every SAP of the full RePatch 477 dataset, all repo pairs.

Resumable by design: it will run for days and across sessions, so every SAP's
row is flushed immediately and any SAP already present in the output is skipped
on restart. Kill and relaunch it freely.

Usage: run_477.py <salp-out-root> <results.jsonl> [--limit N]
"""
import json, sys, time, traceback, os
from pathlib import Path
sys.path.insert(0, "/home/adam/Documents/work/rcap-class-project/rcap/src")
from rcap.backend_ollama import OllamaBackend
from rcap.pipeline import run_case

root = Path(sys.argv[1]); out = Path(sys.argv[2])
limit = int(sys.argv[sys.argv.index("--limit") + 1]) if "--limit" in sys.argv else None
# --per-pair N samples N SAPs from EACH pair, so a smoke test covers every
# repository rather than N SAPs of whichever pair sorts first.
per_pair = int(sys.argv[sys.argv.index("--per-pair") + 1]) if "--per-pair" in sys.argv else None

done = set()
if out.exists():
    for line in out.open():
        line = line.strip()
        if line:
            try: done.add(json.loads(line)["key"])
            except Exception: pass

# pair / PR / sap
saps = []
for pair in sorted(p for p in root.iterdir() if p.is_dir()):
    for pr_dir in sorted(pair.glob("PR-*")):
        man = pr_dir / "pr.json"
        if not man.is_file(): continue
        for sap in sorted(p.name for p in pr_dir.glob("sap-*")):
            key = f"{pair.name}/{pr_dir.name}/{sap}"
            if key not in done:
                saps.append((key, pair.name, pr_dir, sap))
if per_pair:
    seen, sampled = {}, []
    for rec in saps:
        k = rec[1]
        if seen.get(k, 0) < per_pair:
            seen[k] = seen.get(k, 0) + 1
            sampled.append(rec)
    saps = sampled
if limit: saps = saps[:limit]

backend = OllamaBackend()
print(f"backend {backend.name}/{backend.model} digest={backend.model_digest} "
      f"limit={backend.request_limit_chars}", flush=True)
print(f"{len(done)} already recorded; {len(saps)} to run", flush=True)

t_start = time.time()
for i, (key, pair, pr_dir, sap) in enumerate(saps, 1):
    manifest = json.loads((pr_dir / "pr.json").read_text())
    row = {"key": key, "pair": pair, "pr": pr_dir.name, "sap": sap}
    t0 = time.time()
    try:
        res = run_case(pr_dir / sap, manifest, backend)
        row["units"] = len(res.units)
        row["unit_outcomes"] = [u.generation.outcome for u in res.units]
        row["prompt_chars"] = [len(u.request.prompt) for u in res.units]
        row["language"] = res.units[0].request.language if res.units else None
        row["outcome"] = ("completion"
                          if all(o == "completion" for o in row["unit_outcomes"])
                          else next(o for o in row["unit_outcomes"] if o != "completion"))
        row["diagnostics"] = [d for u in res.units
                              for d in (u.generation.diagnostics or [])][:3]
        row["has_package"] = res.package is not None
        row["candidate_chars"] = [len(u.generation.candidate or "") for u in res.units]
        row["usage"] = [getattr(u.generation, "usage", None) for u in res.units]
        # a completion that echoes the reduced target back is not an edit
        row["changed"] = [
            (u.generation.candidate or "").strip() != u.context.transformation["target"].strip()
            if u.generation.outcome == "completion" else None
            for u in res.units]
        row["null_entities"] = list(getattr(res.materialized, "null_entities", []) or [])
    except Exception as exc:
        row["outcome"] = f"REFUSED:{type(exc).__name__}"
        row["stage"] = getattr(exc, "stage", None)
        d = getattr(exc, "diagnostics", None)
        row["diagnostics"] = d[:3] if d else [str(exc)[:200]]
        if getattr(exc, "stage", None) is None:
            row["traceback"] = traceback.format_exc()[-500:]
    row["seconds"] = round(time.time() - t0, 1)
    with out.open("a") as fh:
        fh.write(json.dumps(row) + "\n")
    if i % 10 == 0 or row["outcome"] == "completion":
        el = time.time() - t_start
        rate = el / i
        print(f"[{i}/{len(saps)}] {key}: {row['outcome']} ({row['seconds']}s) "
              f"| {rate:.0f}s/sap, eta {(len(saps)-i)*rate/3600:.1f}h", flush=True)
print("ALL DONE", flush=True)
