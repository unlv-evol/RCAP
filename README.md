# RCAP — Reusable Change Adaptation Pipeline

Consumes a SAP directory produced by [SALP](https://github.com/unlv-evol/SALP) and
produces one Adaptation Package (a single LLM candidate plus provenance) per the
RCAP Implementation Reference. RCAP sits between SALP (semantic alignment) and
SVRP (downstream validation): it engineers evidence into one request, records one
candidate, and never judges correctness (I11) or applies patches — the package is
a handoff.

## Stages

One module per stage (`src/rcap/`), one conformance surface per module:

| Stage | Module | Spec |
|---|---|---|
| SAP intake, endpoint resolution, repo-state binding | `intake.py`, `model.py` | §3–§5 |
| Evidence selection | `selection.py` | §5 |
| Semantic evidence reduction (relationship-guided / degraded) | `reduction_semantic.py` | §6 |
| Payload materialization (I7: payloads only after reduction) | `materialize.py` | §7 |
| Program-context reduction (AST cut under protected regions) | `reduction_program.py` | §8 |
| Adaptation context (backend-independent) | `context.py` | §9 |
| Request synthesis (template `rcap-request-v1`) | `request.py` | §10 |
| Candidate generation (the only nondeterministic stage) | `generate.py`, `backend_ollama.py` | §11 |
| Adaptation package (handoff to SVRP) | `package.py` | §12 |
| Composite coordination (units, coupling, sibling signatures) | `coupling.py`, `pipeline.py` | §13 |
| Four-configuration harness + aggregation | `eval_harness.py`, `tools/aggregate_rows.py` | §15 |

Every failure is typed with stage attribution (§14); failure rows keep the stable
case id, the dispositions established before the failure, and characterization.

## Setup

```bash
uv venv .venv
uv pip install -p .venv/bin/python -e "../../SALP[structural]" -e ".[dev]"
.venv/bin/pytest
```

SALP is a pinned library dependency (clone at ../../SALP); RCAP reuses its SAP
models, schema validator, tree-sitter layer, and repo cache. Tests build their
SAP fixtures with SALP's own writer so every fixture passes `salp validate`.

## Running

End-to-end on real SAPs (SALP output under `../../SALP/data/out/`):

```bash
# 1. Build GACPD input for a PR and run SALP (see SALP's README):
.venv/bin/python tools/make_gacpd.py --cache ../../SALP/data/repos \
    --out ../../SALP/data/gacpd --mainline apache/kafka --divergent linkedin/kafka \
    --pr <N> --cutoff <ISO> --divergence <ISO>
# 2. One SAP through all nine stages against the live backend:
.venv/bin/python tools/run_real.py
# 3. The full 4-config harness over every PR (writes out/PR-*/eval-rows.jsonl):
.venv/bin/python tools/run_harness.py
# 4. The section-15 report (absolute + % vs Raw Context; degraded separated):
.venv/bin/python tools/aggregate_rows.py
```

`tools/dump_stages.py <sap-dir>` runs stages 1–7 on one SAP and writes each
intermediate artifact to `out/inspect/<sap>/` for inspection.

Reference backend: a local [ollama](https://ollama.com) serving
`qwen3-coder:30b` on `127.0.0.1:11435` (temperature 0, fixed seed, 16K
context). θ — backend, model, params, model digest — is recorded in every
generation so a weight change is distinguishable from an evidence change. Any
backend implementing `generate(request) -> str` plugs in; `StubBackend` keeps
the whole pipeline deterministic for tests.

## Outputs

Per case and configuration under `out/` (git-ignored; reproducible from the SAP
inputs): `context.json` (compact — its sha256 is the package's `context_ref`),
`request.txt`, `candidate.java`, `package.json`, and `eval-rows.jsonl` rows with
sizes, backend-reported token counts, dispositions (never collapsed),
coverage/fidelity/readiness metadata, per-hunk reduction mode, and outcome.
Composite cases add per-unit artifacts (`request-unit2.txt`, …) and the package
carries every unit plus the coupling record.

## Known gaps and upstream dependencies

- Real SALP output emits only `aligned_to` edges, so semantic reduction runs
  degraded everywhere; relationship-guided mode is exercised by enriched
  fixtures until SALP's edge promotion lands (§21).
- SALP does not yet serialize function-pool fidelity flags, per-payload
  repository pins, a dependency-diff element, or cross-file relationships;
  RCAP records each gap and substitutes the spec-sanctioned fallback
  (e.g. the dependency diff is computed locally and labeled as such).
- Standalone artifacts (hunks with no function entity) are detected and
  recorded but refuse adaptation: transferability needs an upstream signal.
- Known upstream data issues: τ-anchoring on context lines (null
  transformations, refused at materialization) and whole-file τ slices
  (oversized requests, refused as `limits_exceeded`).

See `docs/experiments.md` for the current measurement results.
