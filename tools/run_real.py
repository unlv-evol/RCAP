"""Run RCAP end-to-end on the minted PR-12535 SAPs with the local LLM backend."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rcap.backend_ollama import OllamaBackend
from rcap.context import build_context
from rcap.generate import generate
from rcap.intake import load_case
from rcap.materialize import NullTransformationFailure, materialize
from rcap.package import build_package
from rcap.reduction_program import reduce_program
from rcap.reduction_semantic import reduce_semantic
from rcap.request import synthesize
from rcap.selection import select

PR_DIR = Path("/home/adam/Documents/work/SALP/data/out/linkedinKafka-apacheKafka/PR-12535")
OUT = Path(__file__).resolve().parent.parent / "out" / "PR-12535"


def main() -> None:
    backend = OllamaBackend()
    print(f"backend: {backend.name}/{backend.model} digest={backend.model_digest}")
    manifest = json.loads((PR_DIR / "pr.json").read_text())

    for sap in sorted(p.name for p in PR_DIR.glob("sap-*")):
        print(f"\n===== {sap} =====")
        case = load_case(PR_DIR / sap, pr_manifest=manifest)
        red = reduce_semantic(case, select(case))
        try:
            mat = materialize(case, red)
        except NullTransformationFailure as exc:
            print(f"  REFUSED (typed failure at {exc.stage}): {exc.diagnostics[0]}")
            continue
        prog = reduce_program(case, red, mat)
        ctx = build_context(case, red, mat, prog)
        req = synthesize(ctx)

        t0 = time.time()
        expected = {f"/* RCAP_PH_{p.ph_id} */" for a in ctx.program_context
                    if a.role == "target" for p in a.placeholders}
        gen = generate(req, backend, expected_placeholders=expected)
        dt = time.time() - t0
        print(f"  outcome={gen.outcome} in {dt:.1f}s")

        case_out = OUT / sap
        case_out.mkdir(parents=True, exist_ok=True)
        (case_out / "request.txt").write_text(req.prompt)
        (case_out / "generation.json").write_text(gen.model_dump_json(indent=1))
        if gen.outcome == "completion":
            pkg = build_package(case, ctx, gen)
            (case_out / "package.json").write_text(pkg.model_dump_json(indent=1))
            (case_out / "candidate.java").write_text(gen.candidate)
            target = ctx.transformation["target"]
            changed = gen.candidate.strip() != target.strip()
            print(f"  candidate: {len(gen.candidate)} chars; "
                  f"{'DIFFERS from' if changed else 'IDENTICAL to'} reduced target")
        print(f"  written to {case_out}")


if __name__ == "__main__":
    main()
