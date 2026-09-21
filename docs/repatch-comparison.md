# RCAP vs RePatch on the RePatch paper's own kafka sample

Run date: 2026-09-20 → 2026-09-21. Work dir: `/home/adam/Documents/work/rcap-class-project/repatch-comparison/`.

## Framing — read this first

RePatch and RCAP produce **different things** and this comparison never claims
otherwise:

- **RePatch** textually/AST-merges a patch into the divergent fork. Its verdict
  is conflicts/files/loc. *Conflict-free ≠ correct*: the prior validation study
  on this same population showed that of the states RePatch had to resolve,
  only **11/76 even build** (git-clean states build 69/71). In this sample,
  of the 74 RePatch CONFLICT_FREE cases only **4 are VALID** downstream
  (64 BUILD_FAIL, 5 TEST_FAIL, 1 INCONCLUSIVE).
- **RCAP** produces **one LLM candidate Adaptation Package** per case.
  *Completion ≠ correct* (invariant I11): correctness is downstream, in SVRP,
  which has not been run here.
- There is **no developer-ported ground truth** on this population: a prior
  ground-truth sweep found **0 clean cherry-picks** across all 232 paper kafka
  scenarios.

Therefore this is a per-scenario **OUTCOME ALIGNMENT** study on the identical
sample — coverage and failure taxonomy — **never an accuracy claim** for either
tool. For paper context: the paper's claimed 53% RePatch-beats-git rate drops
to **3%** under the honest 2.0 engine (`477-comparison.md` in
`repatch-validation-deliverables/`); cite those numbers from there, not from
here.

## Sample construction

Source of truth: `/home/adam/Documents/work/repatch-validation-deliverables/dataset.jsonl`
(477 records).

| filter | n |
|---|---|
| total records | 477 |
| kafka, apache/kafka → linkedin/kafka (kept) | **393** |
| kafka reverse-direction (linkedin→apache etc.) — excluded | 7 |
| non-kafka pairs — excluded | 77 |

The 393 kept rows have 393 distinct PR numbers → `sample.tsv`.
RePatch-side status: 206 CONFLICTED, 112 GIT_CLEAN, 74 CONFLICT_FREE,
1 SKIPPED_NO_SCENARIO (PR 17374, `pr_metadata_unavailable`).
GIT_CLEAN rows carry null verdicts (git applied the patch cleanly, so RePatch
was not invoked); they are counted as conflict-free for **both** tools in the
matrices below. PR 17374 has no verdict at all ("no_verdict" row).

## Coverage funnel (RCAP side)

| stage | PRs | note |
|---|---|---|
| paper sample | 393 | |
| GACPD minting attempted | 363 new + 30 pre-existing | `make_gacpd.py`, rc=0 for **all 366** mint-log rows (`mint_log.tsv`; 3 rows were re-mints of pre-existing PRs) |
| PRs with ≥1 modified `.java` file | 342 | 51 PRs modify no `.java` file (config/tests/other languages) — recorded exclusion |
| PRs SALP minted SAPs for | **291** | further 51 PRs had `.java` changes but every changed file is absent from linkedin/kafka at the cutoff (GACPD class NA) or otherwise unprocessable — recorded exclusion |
| SAPs minted | 698 dirs (SALP reports 699 minted — one same-named SAP in PR-17049 was minted twice and overwrote itself, see validate note) | all **readiness=HIGH**, no `foundational_unavailable` |
| SAPs run through RCAP full_rcap | **698/698** | one generation per processing unit, qwen3-coder:30b, temp 0, seed 12535, num_ctx 16384 |

`salp validate`: 12 findings, all duplicate-object-id findings in the single
SAP `PR-17049/sap-ProcessorContext` — exactly the SAP that then fails RCAP
intake (the one `failure:intake` below). Upstream-SALP data defect, honestly
recorded, not hidden.

SALP ran in **degraded mode** (RefactoringMiner not configured in this
checkout; only `aligned_to` edges) — semantic-reduction evidence is therefore
limited, as in all prior RCAP runs.

## RCAP outcome taxonomy

Per SAP (698):

| outcome | n | % |
|---|---|---|
| completion (packageable candidate) | 357 | 51.1% |
| failure:materialization (null transformation, "null-τ" — SALP slice yields no τ for the entity) | 198 | 28.4% |
| placeholder_violation | 86 | 12.3% |
| limits_exceeded (typed refusal before invocation: prompt exceeds backend ctx; mean prompt 154k chars vs 14k for completions) | 40 | 5.7% |
| unparseable | 10 | 1.4% |
| backend_error (all 6 are 600–993s timeouts) | 6 | 0.9% |
| failure:intake (the defective PR-17049 SAP) | 1 | 0.1% |

148/698 SAPs (21.2%) are composite (multi-unit); 97/148 completed on **all**
units (package built only then).

Per PR (291 with SAPs; a PR "completes" if ≥1 of its SAPs completes):

| PR-level outcome | n |
|---|---|
| completion (≥1 SAP) | **218** |
| — of which every SAP completed | 86 |
| failure:materialization | 34 |
| placeholder_violation | 27 |
| limits_exceeded | 8 |
| backend_error | 2 |
| unparseable | 1 |
| failure:intake | 1 |

## Outcome alignment matrix — RePatch vs RCAP (393 PRs)

RePatch conflict-free = `repatch_verdict.conflicts==0` or status GIT_CLEAN.
RCAP column is the PR-level aggregate above; `no_sap` = the 102 funnel
exclusions (no SAP existed to run).

| | RCAP completion | RCAP failure/refusal | RCAP no_sap | total |
|---|---|---|---|---|
| RePatch conflict-free (186) | 58 | 27 | 101 | 186 |
| RePatch conflicted (206) | 159 | 46 | 1 | 206 |
| no verdict (1) | 1 | 0 | 0 | 1 |
| total | 218 | 73 | 102 | 393 |

Failure/refusal breakdown: conflict-free row = 14 materialization + 11
placeholder_violation + 1 limits + 1 intake; conflicted row = 20
materialization + 16 placeholder_violation + 7 limits + 2 backend_error +
1 unparseable.

**Interesting cells, named examples:**

- *RePatch-conflicted but RCAP completed* (159): PR 12244 (RePatch 4
  conflicts; RCAP 2/3 SAPs complete, 1 limits_exceeded), PR 12250, PR 12259,
  PR 12288, PR 12568 (RePatch 1 conflict; RCAP StandaloneHerder completion —
  the previously hand-verified 7-line callback port — plus one test-file SAP
  backend timeout).
- *RePatch-conflict-free but RCAP failed/refused* (27): **PR 13887 — the
  single genuine RePatch-beats-git win in the whole 477-scenario study
  (CONFLICT_FREE + VALID where git itself conflicted; three other
  CONFLICT_FREE+VALID cases exist but git also applied those cleanly) — is a
  null-τ materialization failure for RCAP.** Also PR 12893,
  12915, 14281 (null-τ), PR 12468, 12584 (placeholder_violation), PR 15517
  (limits_exceeded — whole-file τ larger than the 16k context).
- *No verdict on either RePatch side*: PR 17374 (SKIPPED_NO_SCENARIO,
  pr_metadata_unavailable) — RCAP still produced candidates (2 of its 4 SAPs
  complete).
- *RePatch-conflicted and RCAP no_sap* (1): PR 16303.

## Outcome alignment matrix — git cherry-pick vs RCAP (393 PRs)

git conflict-free = `git_verdict.conflicts==0` or GIT_CLEAN.

| | RCAP completion | RCAP failure/refusal | RCAP no_sap | total |
|---|---|---|---|---|
| git conflict-free (185) | 57 | 26 | 102 | 185 |
| git conflicted (207) | 160 | 47 | 0 | 207 |
| no verdict (1) | 1 | 0 | 0 | 1 |

Near-identical to the RePatch matrix — expected, since in this honest re-run
RePatch and git verdicts differ on only a handful of scenarios (the "3%"
headline). The one-PR shifts are PR 13887 (RePatch-only clean) and the
composition of the no_sap column.

Read of the two matrices: RCAP's completion coverage is roughly independent of
whether the textual merge conflicts — it completes on **77% (159/206)** of the
PRs where RePatch conflicts (i.e., it emits a candidate where the AST merge
has nothing), and its own failure modes (null-τ, oversized whole-file τ,
placeholder discipline) hit conflict-free and conflicted scenarios alike. The
101 conflict-free × no_sap PRs split 53 GIT_CLEAN / 48 CONFLICT_FREE — PRs
whose changed files don't exist in the fork or aren't Java, populations where
a textual apply already lands (or trivially no-ops) and SALP has nothing to
align.

## Validation context (conflict-free ≠ working)

From `dataset.jsonl` on this sample:

- RePatch CONFLICT_FREE (74): 4 VALID, 5 TEST_FAIL, **64 BUILD_FAIL**, 1
  INCONCLUSIVE → a conflict-free RePatch merge builds+passes in ~5% of cases.
- GIT_CLEAN (112): 47 VALID, 21 TEST_FAIL, 2 BUILD_FAIL, 42 INCONCLUSIVE.

This is the yardstick to keep in mind for RCAP's 218 "completions" too: a
completion is a *candidate*, not a validated adaptation.

## Runtime / tokens

- RCAP sweep wall time: ~13.4 h (sum of per-SAP runtimes 48,380 s; single
  GPU, sequential). Minting: ~18 min for 366 PRs. SALP run+validate: ~13 min.
- Backend tokens (backend-reported, never estimated): **1,710,527 input**,
  **845,877 output** across the 698-SAP sweep.
- Mean completion runtime 75.3 s/SAP; the 6 backend_errors are 600 s-class
  timeouts on large test-file SAPs.

## What this does — and does not — show

**Shows:**
- On the paper's own 393-scenario kafka sample, the RCAP pipeline runs end to
  end: 291/393 PRs yield SAPs, 698 SAPs run, every outcome typed; 218 PRs
  (74.9% of PRs-with-SAPs, 55.5% of the full sample) yield at least one LLM
  candidate, 86 PRs complete on every SAP.
- RCAP produces candidates on 159 of the 206 PRs where the honest RePatch
  engine reports conflicts — outcome coverage in exactly the region where
  textual merging stops.
- A reproducible failure taxonomy: null-τ 28.4% of SAPs (upstream SALP
  slicing, the known class-level-τ ask), placeholder violations 12.3%,
  capacity refusals 5.7% (typed, pre-invocation), backend timeouts 0.9%.

**Does not show:**
- **No correctness claim on either side.** RCAP completions await SVRP
  (build/test validation was not run on any candidate). RePatch conflict-free
  is textual only — on this very sample only 4/74 of its conflict-free merges
  are VALID.
- No "RCAP beats RePatch" claim is possible from outcome alignment: the two
  outputs are not the same artifact type, and there is no ported ground truth
  on this population (0 clean cherry-picks in the prior ground-truth sweep).
- The 102 no_sap PRs are upstream exclusions (51 no modified `.java`, 51 all
  changed files absent from the fork / unprocessable), not RCAP refusals — but
  they honestly cap RCAP's whole-sample coverage at 291/393.
- SALP ran degraded (no RefactoringMiner edges), so semantic reduction
  evidence is thinner than the design intends; null-τ and placeholder rates
  may improve with enriched SALP output.

## Files

- `sample.tsv` — the 393-row sample with both verdicts and validation outcome.
- `prs_to_mint.txt`, `mint_log.tsv`, `mint_saps.sh`, `mint_stdout.log` — minting.
- `salp_run.log` — SALP fetch-repos/run/validate transcript (699 minted, 12
  validate findings on PR-17049/sap-ProcessorContext).
- `run_rcap.py`, `run_rcap.log` — the full_rcap runner (incremental, resumable).
- `results.jsonl` — 698 per-SAP records (outcome, units, tokens, runtime,
  readiness).
- `compare.py`, `comparison.json` — aggregation and the matrices above.

---

**Update 2026-09-21 — independent verification.** An adversarial authenticity
audit (`verification-audit/repatch-authenticity.md`) recomputed every number
in this document from the raw artifacts: the funnel, the full SAP/PR outcome
taxonomies, both alignment matrices cell-for-cell, the 4/74 VALID claim, and
the token/runtime totals all reproduce exactly; for 15 stratified spot-check
rows the deterministic pre-LLM stages were re-executed and reproduced every
failure outcome, with rebuilt prompts matching recorded sizes byte-exactly.
Verdict: authentic; the small errata it found (mint-funnel split, two
timing claims, timeout range, PR 13887 phrasing) are applied above. One
caveat stands: this sweep persisted outcome/token rows only — candidate
files and generation records were not written to disk, so the 357 candidates
cannot be inspected post hoc. A re-run with artifact persistence (planned
together with SVRP validation) closes that.
