# Full-population run: 477 RePatch validation cases

Drivers for running RCAP over every case of the RePatch validation dataset
(`validation-study/validation-data/dataset.jsonl`, 477 cases, 10 repository
pairs), then relabelling and validating the output. Data and results live
outside the repo in `rcap-class-project/run477/`; the paths are absolute there.

## Pipeline

| Step | Script | Output (under `run477/`) |
|---|---|---|
| 1. Mint GACPD-format dirs per case | `mint_477.py` | `gacpd-477/`, `mint_477_log.jsonl` |
| 2. SALP over those dirs | `salp run` (SALP CLI) | `salp-477-out/` |
| 3. RCAP over every SAP (resumable) | `run_477.py` | `results-477-v2.jsonl` |
| 4. MO / ED / SP relabel | `classify_mo.py`, `split_results.py` | `mo-relabel/` |
| 5. Did the fork later port it? | `later_ported.py` | `mo-relabel/later_ported.jsonl` |
| 6. Regenerate MO completions, keeping candidates | `run_persist.py` | `validate/candidates/` |
| 7. Splice → build → test each candidate | `validate.py`, `validate_loop.sh` | `validate/verdicts-*.jsonl` |

`probe_units.py` runs the pipeline up to request synthesis without an LLM.

## Things that matter

- **Diff base.** `mint_477.py` derives each PR's base from git (merge-base with
  the mainline, or the first parent if the head has landed). The dataset's
  `base_revision` is the fork divergence point, not the PR's parent, and is only
  cross-checked, never used: using it fabricated diffs in the first attempt.
- **"MO" from `make_gacpd.py` is not GACPD's MO.** The generator labels a file
  MO whenever it exists in the fork. GACPD's MO means *missed opportunity*: the
  upstream change is absent from the fork. Step 4 relabels by checking whether
  the change's added lines are already in the fork at the target commit
  (≤ 20% → MO, ≥ 80% → ED, otherwise SP). This is an approximation of GACPD.
- **Validation baseline.** `validate.py` runs each test class on the unmodified
  fork first; only new failures count against a candidate. Abstract test classes
  run through their subclasses, and class names come from the source's package
  clause (Kafka's `src/test/scala/unit/` is not part of the package).
- **JDK.** The linkedin/kafka fork uses Gradle 7.1.1; `validate.py` pins JDK 11
  per invocation (`-Dorg.gradle.java.home`) rather than relying on a global
  Gradle setting.
- **Completion is not correctness.** Steps 6–7 exist to test candidates; a
  completion only means the output passed RCAP's structural checks.
