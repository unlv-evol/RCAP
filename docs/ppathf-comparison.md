# RCAP vs PPatHF — 62-case comparison on real kafka fork data

PPatHF (Pan et al., ISSTA'24) is RCAP ref [1] and the blueprint for our
Program-Context Reduction; harness configuration 2 (Program-Reduction-Only)
mirrors it by design (§15: protected set = changed regions only). This
comparison runs 62 conforming SAPs (27 apache/kafka → linkedin/kafka PRs,
2026-09-17 sweep, `out/harness-run-20260917-ppathf.log`) and compares what is
honestly comparable. Regenerate with `tools/run_harness.py` then
`tools/compare_ppathf.py`.

## 1. Retained fraction after reduction

PPatHF reports its reduction module keeps **0.70** of the original function
length on average (Vim→Neovim, C). Ours, paired per case against Raw Context
on the 47 measurable cases (15 null-τ refusals excluded — nothing to reduce):

| metric | mean (all 47) | median | cases where the cut fires | their mean |
|---|---|---|---|---|
| prompt chars | 0.88 | 0.93 | 30 | **0.81** |
| AST nodes | 0.84 | 0.92 | 30 | **0.75** |

Strongest cases reach 0.48–0.58 (ConsumerCoordinator, KafkaAdminClient,
AbstractCoordinator, PartitionGroup, BatchBuilder). The gap to PPatHF's 0.70
average is structural, not algorithmic: their population is whole C functions;
ours includes class-level and whole-file τ slices where a compound-statement
cut finds little to remove — the same cases behind the SALP member-level
slicing ask.

Population-level effect (from `out/aggregate.md`, backend-reported tokens):
program reduction −26.4% AST nodes, −19.6% prompt chars, **−9.6% real input
tokens, −10.9% output tokens** vs Raw Context.

## 2. Fitting the window

PPatHF's motivation: even at 8k tokens StarCoder could not fit all 310
patches. Reproduced here: 7 cases exceed the declared request limit raw;
reduction brings 2 of them under it (7 → 5 `limits_exceeded`). RCAP refuses
oversized requests with a typed outcome instead of truncating.

## 3. Where PPatHF's assumptions break on real fork data

PPatHF assumes a one-to-one function correspondence, a non-null
transformation, and a function that fits the window. On this population:

- 24% (15/62) null transformation — upstream τ-anchoring slices an untouched
  neighbor; typed refusal at materialization;
- 34% (21/62) composite — multiple processing units per change (§13 per-unit
  processing with coupling records; 15/21 complete in full RCAP);
- 8 cases exceed the window at least once across configs.

A majority of real fork cases violate at least one assumption the Vim→Neovim
setting takes for granted — the population taxonomy is itself a result.

## 4. Deliberately not compared

- **Accuracy**: PPatHF's headline 42.3% (131/310) is exact match against the
  developer-ported patch, with a LoRA-fine-tuned StarCoder-15.5B. Our
  population mostly has no developer ports (linkedin did not adopt these
  changes), RCAP is zero-shot by design, and correctness is SVRP's downstream
  concern (I11) — no accuracy number is produced, and none should be read
  into completion counts (35/62 full-RCAP completions ≠ 35 correct ports).
- **Model/tuning**: qwen3-coder:30b zero-shot vs fine-tuned StarCoder; any
  outcome difference confounds model, tuning, language and population. The
  within-population ablation (config 2 vs 4, same code path) is the
  attribution-safe comparison; the semantic term is 0 until SALP promotes
  edges (§21).
- New honest-guard observation at this scale: 11 rows record
  `placeholder_violation` — the model dropped a required placeholder and the
  candidate was refused packaging rather than silently accepted; PPatHF's
  pipeline reinserts placeholders without an equivalent check.
