#!/usr/bin/env python3
"""Step 2: RCAP's request pipeline on the benchmark, one arm, one deterministic run.

  run_arm.py --out DIR --arm rcap_full|rcap_reduced [--only SUBSTR] [--limit N] [--offline]

Per case: one unit (common.Unit) -> rcap.request.synthesize (rcap-request-v1,
fenced as C) -> rcap.generate.generate with the backend's declared limits, the
placeholder check and the duplicated-block check. `rcap_reduced` first cuts the
three function texts with RCAP's program reduction (RCAP_PH markers).

A row holds the pipeline's outcome AND, when the model answered, the answer as
it was written (`raw_code`, placeholder form) and with the target's
placeholders expanded (`as_is`), so a rejected answer can still be scored
"as is" next to "rejected = the fork stays unchanged".

Resumable by case_id; writes DIR/arms/<arm>.jsonl and DIR/gen-cache only.
"""
import argparse, json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import (Unit, extract_code, load_cases, load_config, load_jsonl, make_backend,  # noqa: E402
                    provenance, raw_answer, sha)
from rcap.accept import placeholder_marks  # noqa: E402
from rcap.generate import generate, placeholder_report  # noqa: E402
from rcap.request import synthesize  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True); ap.add_argument("--arm", required=True)
ap.add_argument("--only"); ap.add_argument("--limit", type=int); ap.add_argument("--offline", action="store_true")
a = ap.parse_args()
cfg = load_config(a.out); arm = cfg["arms"][a.arm]; assert arm["step"] == 2
cases = load_cases(cfg)
backend = make_backend(cfg, live=not a.offline)
prov = provenance(cfg, backend, arm=a.arm, arm_config=arm)
out = Path(a.out) / "arms" / f"{a.arm}.jsonl"; out.parent.mkdir(parents=True, exist_ok=True)
done = load_jsonl(out)
n = 0
with out.open("a") as fh:
    for i, (cid, case) in enumerate(cases.items(), 1):
        if cid in done or (a.only and a.only not in cid):
            continue
        if a.limit is not None and n >= a.limit:
            break
        n += 1
        t0 = time.time()
        unit = Unit(cid, case, bool(arm["program_reduction"]), arm.get("min_placeholder_lines", 3))
        request = synthesize(unit.context)
        gen = generate(request, backend, expected_placeholders=unit.expected_placeholders,
                       reject=unit.reject, parse_reference=unit.context.transformation["target"])
        cached = raw_answer(cfg, request.prompt)
        raw_code = extract_code(cached.get("raw")) if cached and not cached.get("error") else None
        if gen.capacity in ("request_too_large", "answer_cannot_fit_window", "input_truncated",
                            "window_overflow", "output_budget_exhausted", "generation_timeout"):
            # never scored "as is": what the runtime returned (if anything) was
            # written without the whole request, or cut
            as_is = None
        else:
            as_is = unit.expand(raw_code) if raw_code else None
        row = {
            "arm": a.arm, "case_id": cid, "outcome": gen.outcome, "capacity": gen.capacity,
            "diagnostics": gen.diagnostics[:4],
            "candidate_raw": gen.candidate,
            "candidate": unit.expand(gen.candidate) if gen.candidate else None,
            "raw_code": raw_code, "as_is": as_is,
            "placeholder_report": placeholder_report(raw_code, unit.expected_placeholders) if raw_code else None,
            "marks_after_expansion": placeholder_marks(as_is)[:6] if as_is else [],
            "template_id": request.template_id, "request_hash": request.request_hash,
            "prompt_sha256": sha(request.prompt), "prompt_chars": len(request.prompt),
            "usage": gen.usage, "limits": gen.limits, "answer_source": backend.last_source if raw_code else None,
            "reduction": unit.reduction_stats(), "seconds": round(time.time() - t0, 1),
            "provenance": prov,
        }
        fh.write(json.dumps(row) + "\n"); fh.flush()
        print(f"[{i}/{len(cases)}] {cid} {gen.outcome}{'/' + gen.capacity if gen.capacity else ''} "
              f"prompt={len(request.prompt)}c in={gen.usage.get('input_tokens')} out={gen.usage.get('output_tokens')} "
              f"t={row['seconds']}s", flush=True)
print("DONE", a.arm, dict(backend.sources))
