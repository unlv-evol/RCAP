"""Mirror rcap.pipeline.run_case up to request synthesis, without the LLM.

run_case processes a composite case PER UNIT (section 13); building one context
for the whole case is the single-unit path and refuses composites by design, so
probing that way would report RCAP's refusal where none occurs.
"""
import json, sys, collections
from pathlib import Path
sys.path.insert(0, "/home/adam/Documents/work/rcap-class-project/rcap/src")
from rcap.intake import load_case
from rcap.selection import select
from rcap.reduction_semantic import reduce_semantic
from rcap.materialize import materialize
from rcap.reduction_program import reduce_program
from rcap.context import build_context
from rcap.coupling import detect_units
from rcap.request import synthesize

root = Path(sys.argv[1])
tally = collections.Counter(); rows = []
for pr_dir in sorted(root.glob("PR-*")):
    manifest = json.loads((pr_dir / "pr.json").read_text())
    for sap in sorted(p.name for p in pr_dir.glob("sap-*")):
        rec = {"pr": pr_dir.name, "sap": sap}
        try:
            case = load_case(pr_dir / sap, pr_manifest=manifest)
            rec["lang"] = case.language
            red = reduce_semantic(case, select(case))
            mat = materialize(case, red)
            prog = reduce_program(case, red, mat)
            coupling = detect_units(case)
            adaptable = mat.adaptable(red.materialize)
            units = [u for u in coupling.units if u.entity in adaptable]
            helpers = tuple(e for e in adaptable
                            if e not in {u.entity for u in units})
            reqs = []
            if len(units) <= 1:
                ctx = build_context(case, red, mat, prog,
                                    units[0] if units else None, (), helpers)
                reqs.append(synthesize(ctx))
            else:
                for u in units:
                    ctx = build_context(case, red, mat, prog, u,
                                        tuple(s for s in units if s is not u), helpers)
                    reqs.append(synthesize(ctx))
            rec["outcome"] = "request_ready"
            rec["units"] = len(reqs)
            rec["prompt_chars"] = [len(r.prompt) for r in reqs]
            rec["fence_lang"] = reqs[0].language
            # the backend's own admission limit (num_ctx*4)
            rec["over_limit"] = sum(1 for r in reqs if len(r.prompt) > 16384 * 4)
        except Exception as exc:
            rec["outcome"] = f"REFUSED:{type(exc).__name__}"
            rec["stage"] = getattr(exc, "stage", None)
            d = getattr(exc, "diagnostics", None)
            rec["diag"] = (d[0] if d else str(exc))[:160]
        tally[rec["outcome"]] += 1
        rows.append(rec)
print(json.dumps(tally, indent=1))
ready = [r for r in rows if r["outcome"] == "request_ready"]
print(f"\nSAPs reaching the LLM: {len(ready)}/{len(rows)}")
print(f"total units (prompts) : {sum(r['units'] for r in ready)}")
print(f"units over backend admission limit (65536 chars): {sum(r['over_limit'] for r in ready)}")
print(f"fence languages: {collections.Counter(r['fence_lang'] for r in ready)}")
print("\nrefusals by stage:")
for r in rows:
    if r["outcome"] != "request_ready":
        print(f"  {r['pr']}/{r['sap']}: {r['outcome']} @ {r.get('stage')} :: {r.get('diag','')[:100]}")
json.dump(rows, open(sys.argv[2], "w"), indent=1)
