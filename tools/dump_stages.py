"""Run one SAP through RCAP stages 1-7 and write each stage's output as JSON.

Usage:
  .venv/bin/python tools/dump_stages.py <path-to-sap-dir> [--out DIR]

No LLM call is made (stages 8-9 need a backend); everything written is the
deterministic part of the pipeline, so re-running always reproduces it.
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rcap.context import build_context
from rcap.intake import load_case
from rcap.materialize import materialize
from rcap.reduction_program import reduce_program
from rcap.reduction_semantic import reduce_semantic
from rcap.request import synthesize
from rcap.selection import select


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("sap_dir", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    sap_dir = args.sap_dir.resolve()
    manifest = json.loads((sap_dir.parent / "pr.json").read_text())
    out = args.out or Path(__file__).resolve().parent.parent / "out" / "inspect" / sap_dir.name
    out.mkdir(parents=True, exist_ok=True)

    def dump(n: int, name: str, model) -> None:
        path = out / f"{n:02d}_{name}.json"
        path.write_text(model.model_dump_json(indent=1), encoding="utf-8")
        print(f"  wrote {path}")

    try:
        case = load_case(sap_dir, pr_manifest=manifest)
        dump(1, "intake_case", case)
        sel = select(case)
        dump(2, "selection", sel)
        red = reduce_semantic(case, sel)
        dump(3, "semantic_reduction", red)
        mat = materialize(case, red)
        dump(4, "materialization", mat)
        prog = reduce_program(case, red, mat)
        dump(5, "program_reduction", prog)
        ctx = build_context(case, red, mat, prog)
        dump(6, "context", ctx)
        req = synthesize(ctx)
        dump(7, "request", req)
        (out / "07_request_prompt.txt").write_text(req.prompt, encoding="utf-8")
        print(f"  wrote {out / '07_request_prompt.txt'}")
    except Exception as exc:
        stage = getattr(exc, "stage", None)
        if stage is None:
            raise
        diagnostics = list(getattr(exc, "diagnostics", [])) or [str(exc)]
        (out / "REFUSED.json").write_text(json.dumps(
            {"stage": stage, "error": type(exc).__name__,
             "diagnostics": diagnostics}, indent=1), encoding="utf-8")
        print(f"\n  REFUSED at stage '{stage}' ({type(exc).__name__}):")
        for d in diagnostics:
            print(f"    {d}")
        print(f"  details written to {out / 'REFUSED.json'}")
        print("  (stages dumped above completed before the refusal;"
              " later stage files were not produced)")


if __name__ == "__main__":
    main()
