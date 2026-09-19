# RCAP experiment writeup — four-configuration ablation on real kafka SAPs

Status: 2026-09-17/18 sweeps (`out/harness-run-20260917-ppathf.log`,
`out/repeat-run-20260918.log`). Everything below is regenerable:
`tools/run_harness.py`, then `tools/aggregate_rows.py`,
`tools/compare_ppathf.py`, `tools/repeat_runs.py`.

## Setup

- **Population**: every conforming SAP SALP produced for 27 apache/kafka →
  linkedin/kafka pull requests — 62 SAPs, all readiness HIGH. Eligibility is
  conformance only; no prior integration outcome plays any part (I12).
- **Backend**: qwen3-coder:30b (qwen3moe, 30.5B/3.3B active, Q4_K_M) on a
  local ollama, temperature 0, fixed seed, 16K context, declared request
  limit num_ctx×4 chars. θ (backend, model, params, digest, request limit)
  is recorded in every generation.
- **Configurations** (§15, one code path, two toggles): Raw Context /
  Program-Reduction-Only / Semantic-Reduction-Only / Full RCAP. 4 rows per
  case, 248 rows total.
- **Measurements per row**: AST nodes before/after, context and prompt sizes,
  backend-reported input/output token counts (never estimated — absent for
  refused calls), dispositions (never collapsed), per-hunk reduction mode,
  coverage/fidelity/readiness metadata, outcome, runtime, units and per-unit
  outcomes for composites. No correctness metric appears in any row (I11).

## Outcomes (248 rows, 62 cases)

| outcome | rows | what it is |
|---|---|---|
| completion | 149 | packageable candidate produced |
| failure:materialization | 60 | 15 cases × 4 configs: null transformation (upstream τ-anchoring; refused, never generated) |
| limits_exceeded | 25 | request beyond the declared window — refused before invocation, never truncated |
| placeholder_violation | 11 | the model dropped a required placeholder; candidate refused packaging |
| backend_error | 3 | 600s backend timeout on very large units |

Composite cases (§13): 21 of 62 cases run as multiple processing units in
application order, with sibling-edit signatures and one Adaptation Package
per case; 15 of 21 complete fully under Full RCAP. All real couplings are
*inferred from shared-function co-location* — none asserted by SALP yet.

## Context efficiency, absolute and % vs Raw Context

From `out/aggregate.md` (degraded population, 47 cases — all real cases run
degraded until SALP promotes edges; tokens are backend-reported and excluded
pairwise where a call was refused):

| configuration | completions | nodes_after | prompt_chars | input_tokens | output_tokens |
|---|---|---|---|---|---|
| raw_context | 39 | 352,674 (ref) | 2,415,228 (ref) | 150,867 (ref) | 94,605 (ref) |
| program_reduction_only | 36 | −26.4% | −19.6% | −9.6% | −10.9% |
| semantic_reduction_only | 39 | +0.0% | +0.0% | +0.0% | −0.1% |
| full_rcap | 35 | −26.4% | −19.6% | −9.6% | −15.4% |

The unmeasured population (15 cases) is the null-τ refusals.

## Repeatability (section 11: repeated generations)

PR-12535's completing case, 5 repetitions × {raw_context, full_rcap},
identical requests (temperature 0, fixed seed; `request_hash` identical):

- **input_tokens: zero variance** (1463 raw / 1129 full, every repetition);
- **output_tokens: zero variance** (369 both modes);
- **candidate code: bit-identical across all repetitions** in both modes;
- only wall-clock varies (full: 15.9–19.4s; raw: 15.8–27.1s — first-rep
  cold effects).

So on this backend the recorded token counts are stable run-to-run and the
whole pipeline is deterministic end-to-end at fixed θ; runtime is the only
noisy column. Full RCAP used 334 fewer input tokens than raw (−22.8%) for a
byte-identical answer on this case.

## Reading the numbers honestly

- **Semantic reduction contributes exactly 0** on real data — as designed:
  with only `aligned_to` edges every hunk runs degraded. This column is the
  §21 enrichment tracker; it moves when SALP promotes edges. On enriched
  synthetic fixtures, relationship-guided reduction demonstrably drops
  unrelated evidence (tests: I6, I8).
- **Program reduction bites on method-level τ, not class-level slices.** The
  strongest cases retain 0.48–0.58 of the raw input; whole-file τ slices
  barely reduce. Upstream ask: member-level τ slicing.
- **Reduction recovered 2 capacity cases** (7 → 5 `limits_exceeded`).
- **Completion is not correctness** (I11): 35/62 full-RCAP completions are
  packageable candidates, not verified ports. The systematic answer is
  SVRP's, consuming these packages.
- **Refusals are results**: null-τ, capacity and placeholder violations are
  typed, diagnosed rows carrying stable ids, dispositions and
  characterization.

See `docs/ppathf-comparison.md` for the comparison against PPatHF
(ISSTA'24), including what is deliberately not compared.

## Reproducing

```bash
.venv/bin/python tools/run_harness.py      # 4 configs × 62 SAPs, live backend
.venv/bin/python tools/aggregate_rows.py   # section-15 table -> out/aggregate.md
.venv/bin/python tools/compare_ppathf.py   # PPatHF comparison -> out/ppathf-comparison.md
.venv/bin/python tools/repeat_runs.py <pr-dir> --reps 5   # nondeterminism study
```
