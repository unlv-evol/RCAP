# RCAP experiment writeup — four-configuration ablation on real kafka SAPs

Status: 2026-09-17 sweep (`out/harness-run-20260917b.log`). Everything below is
regenerable: `tools/run_harness.py` then `tools/aggregate_rows.py`.

## Setup

- **Population**: every conforming SAP SALP produced for 5 apache/kafka →
  linkedin/kafka pull requests (12535, 12568, 12660, 13132, 13758) — 16 SAPs.
  Eligibility is conformance only; no prior integration outcome plays any part
  (I12).
- **Backend**: qwen3-coder:30b (qwen3moe, 30.5B/3.3B active, Q4_K_M) on a local
  ollama, temperature 0, fixed seed, 16K context, declared request limit
  num_ctx×4 chars. θ (backend, model, params, digest, request limit) is
  recorded in every generation; runs are deterministic up to the backend.
- **Configurations** (§15, one code path, two toggles): Raw Context /
  Program-Reduction-Only / Semantic-Reduction-Only / Full RCAP. 4 rows per
  case, 64 rows total.
- **Measurements per row**: AST nodes before/after, context and prompt sizes,
  backend-reported input/output token counts (never estimated — absent for
  refused calls), dispositions (never collapsed), per-hunk reduction mode,
  coverage/fidelity/readiness metadata, outcome, runtime. No correctness
  metric appears in any row (I11); correctness is SVRP's, downstream.

## Outcomes (64 rows)

| outcome | rows | what it is |
|---|---|---|
| completion | 38 | packageable candidate produced |
| failure:materialization | 20 | 5 cases × 4 configs: null transformation (upstream τ-anchoring slices an untouched neighbor; refused, never generated) |
| limits_exceeded | 4 | KStream × 4 configs: 310,302-char whole-interface prompt vs 16K-token window — refused before invocation (previously a silently-truncated garbage "completion") |
| backend_error | 2 | StandaloneHerderTest unit 1 (48KB test-class τ) hit the 600s backend timeout in the two program-reduction configs; it completed in the raw configs (~400s) |

Composite cases (§13, first live run): Produced, Repartitioned,
TopologyDescription and StandaloneHerderTest each ran as 2 processing units in
application order — sibling-edit signatures attached, one Adaptation Package
per case with per-unit candidates and the coupling record (all real couplings
are *inferred from shared-function co-location*, none asserted by SALP).
Before §13 these four cases were typed refusals.

## Context efficiency, absolute and % vs Raw Context

From `out/aggregate.md` (populations never mixed; token columns are
backend-reported and excluded pairwise where a call was refused):

**Degraded population (11 cases — all real cases; SALP emits only `aligned_to`
edges today):**

| configuration | completions | other | nodes_after | prompt_chars | input_tokens |
|---|---|---|---|---|---|
| raw_context | 10 | limits:1 | 32,849 (ref) | 468,983 (ref) | 33,748 (ref, n=10) |
| program_reduction_only | 9 | limits:1, backend_error:1 | 31,581 (−3.9%) | 463,597 (−1.1%) | 21,027 (−2.7%, n=9) |
| semantic_reduction_only | 10 | limits:1 | 32,849 (+0.0%) | 468,983 (+0.0%) | 33,748 (+0.0%, n=10) |
| full_rcap | 9 | limits:1, backend_error:1 | 31,581 (−3.9%) | 463,597 (−1.1%) | 21,027 (−2.7%, n=9) |

The unmeasured population (5 cases) is the null-τ refusals: all four rows fail
at materialization, so no size is a measurement there.

## Reading the numbers honestly

- **Semantic reduction contributes exactly 0** on real data — as designed, not
  as a bug: with only `aligned_to` edges every hunk runs degraded
  (category-level retention). This column is the §21 enrichment tracker; it
  moves when SALP promotes edges. On enriched synthetic fixtures,
  relationship-guided reduction demonstrably drops unrelated evidence
  (tests: I6, I8).
- **Program reduction bites on method-level τ and not on class-level τ.** The
  population average (−3.9% nodes) is dominated by whole-file/class-level
  slices where the compound-statement cut finds little to remove. On the
  method-level cases it is large: SubscriptionStore 1431→979 nodes (−31.6%),
  prompt 8797→6521 chars. Upstream ask: member-level τ slicing for class-level
  changes.
- **Full RCAP == Program-Reduction-Only today**, because the semantic term is
  0. The four-config design is what makes that attributable: same code path,
  toggles only.
- **Completion is not correctness** (I11). Informal spot checks against the
  upstream patches (outside the harness, no row records them) previously found
  exact minimal adaptations on several method-level cases — e.g. the
  SubscriptionStore `!=`→`>` version check relocated into the drifted fork
  API — and javadoc misses on others. The systematic answer is SVRP's,
  consuming these packages.
- **Refusals are results.** 5 null-τ cases and the KStream capacity case are
  typed, diagnosed refusals carrying stable ids, dispositions and
  characterization — data for the upstream fixes, not silent gaps.

## Reproducing

```bash
.venv/bin/python tools/run_harness.py     # 4 configs × 16 SAPs, live backend
.venv/bin/python tools/aggregate_rows.py  # section-15 table -> out/aggregate.md
```

Per-case artifacts land in `out/PR-*/sap-*/<config>/` (context.json —
sha256 equals the package's context_ref — request.txt, candidate.java, and
per-unit variants for composites).
