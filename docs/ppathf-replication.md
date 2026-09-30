# PPatHF Replication on a Local Zero-Shot Backend

Replication of the evaluation of Pan et al., "Automating Zero-Shot Patch
Porting for Hard Forks" (ISSTA'24), on **their** Vim->Neovim test set (310
patches), using **our** local LLM backend (ollama, qwen3-coder:30b, zero-shot,
RCAP-style prompt), scored with **their** metric definitions.

Date of run: 2026-09-19/20. Everything in this directory; nothing outside it
was modified.

## 1. Dataset

Obtained from the Google Drive link in the PPatHF README
(`PPatHF/README.md`, "Dataset" section) via `gdown --folder`:

| file | cases | role |
|---|---|---|
| `data/data/vim_neovim_test_all.json` | **310** | full test set (the paper's evaluation set) |
| `data/data/sliced/vim_neovim_test_sliced_all.json` | 310 | same cases after the paper's own reduction module (placeholdered inputs + `removed_pieces_*` recovery maps) |
| `data/data/vim_neovim_test_{2048,4096,8192}.json` | 115/217/274 | the paper's length-filtered subsets (not needed here; we use the full 310) |
| `data/data/finetune.json` | 4829 | their finetuning corpus (unused -- we are zero-shot by design) |

Each case: `func_before_source`/`func_after_source` (Vim f_s, f_s'),
`func_before_target` (Neovim f_f), `func_after_target` (developer-ported
ground truth f_f'). All 310 `commit_id_target` values are unique.

## 2. Metric definitions used (taken from their code)

Implemented in `score.py`, faithfully re-coded from the cloned replication
package (`./PPatHF`):

- **Tokenization** -- `PPatHF/reduction/fcu.py::get_cleaned_tokens` (lines
  75-101): drop empty lines, parse with the tree-sitter C grammar, take leaf
  tokens in order, discard `comment` nodes. So whitespace/formatting and
  comments are ignored by all metrics.
- **Accuracy (exact match)** -- `PPatHF/metrics.py` line 77: token-sequence
  equality of the candidate vs `func_after_target`.
- **AED** -- `metrics.py` line 78: token-level Levenshtein distance
  (`editdistance.eval` over the token lists) candidate vs `func_after_target`,
  averaged over cases. Unit = tree-sitter C tokens.
- **RED** -- `metrics.py` lines 79-82: per case
  `dist(candidate, f_f') / dist(f_f, f_f')` (denominator forced to 1 when the
  target-side change is token-empty), averaged over cases.
- **Denominator / failure handling** -- `PPatHF/test.py::align_test_metrics`
  (lines 119-154) + `generate_dummy` (line 27) + paper Sec. 6.1: over the full
  310, any case the model cannot handle (over the length limit) or whose
  generation is *incomplete* (last non-empty output line does not start with
  `}`, `test.py` line 92) is substituted by the **dummy output = unchanged
  fork function f_f** (=> that case scores exact=false unless the target
  change was comment-only, RED=1, AED=src_distance). Their "Complete" column
  counts non-substituted cases. We apply exactly this protocol.
- **Recovery for the reduced pass** -- `PPatHF/reduction/reducer.py::
  do_recovering` (lines 507-547): each `/* placeholder_N */` comment in the
  generation is replaced by the recorded removed piece. We implement it as
  literal textual replacement of the exact placeholder comment (their
  placeholders always take exactly this form, cf. `test.py` line 199).

**Scorer validation.** Scoring the all-dummy output (every case = f_f)
reproduces the paper's "Origin" row: Accuracy 0/310, RED 1.000, AED **26.66**
vs their **26.60** (0.2% off -- tree-sitter grammar version difference; we use
the current `tree-sitter-c` wheel, they pinned tree-sitter 0.20.1 from 2023).
This bounds the tokenization skew of every AED number below at ~0.2%.

## 3. Setup: theirs vs ours

| | PPatHF (paper) | This replication |
|---|---|---|
| Model | StarCoder-15.5B, fp16 | qwen3-coder:30b (Qwen3-Coder 30B-A3B MoE), **Q4_K_M** GGUF |
| Adaptation | **LoRA fine-tuned** on 4,829 Vim/Neovim commits (+ their reduction module for the PPatHF rows) | **Zero-shot** -- no fine-tuning, no demonstrations |
| Prompt | Their instruction + `### Function Before (vim) / After (vim) / Before (neovim) / After (neovim):` completion format (`PPatHF/porting/data.py` lines 13-25) | RCAP-style sectioned prompt (mirrors `rcap/src/rcap/request.py`): Task instructions, source BEFORE/AFTER and fork target in ```c blocks, Output section demanding one ```c block (`harness.py`) |
| Context limit | 2k / 4k / 8k tokens (prompt+completion), left-truncation | 16,384-token ctx; prompts > **65,536 chars refused up front** (`limits_exceeded`, RCAP's declared request limit -- no silent truncation; runtime truncation additionally detected via `prompt_eval_count`, never triggered) |
| Decoding | greedy (HF `generate`, no sampling) | temperature 0, seed 12535 |
| Failure default | dummy = unchanged f_f | identical protocol |
| Hardware | GPU cluster (theirs) | RTX 2070 SUPER 8 GB, hybrid CPU/GPU offload, local ollama @127.0.0.1:11435 |

## 4. Results

Paper numbers = Table 1 (p. 370) and Table 2 (p. 371) of the paper,
Vim->Neovim, N=310, dummy-substituted ("aligned") protocol.

| Approach | Complete | Accuracy (of 310) | AED | RED (mean) |
|---|---|---|---|---|
| Origin (do nothing) | / | 0 (0.0%) | 26.60 | 1.00 |
| git-apply | / | 0 (0.0%) | / | / |
| patch | / | 0 (0.0%) | 26.57 | 1.00 |
| FixMorph | / | 13 (4.2%) | 31.56 | 1.52 |
| StarCoder @2k (no FT) | 83 (26.8%) | 29 (9.4%) | 23.05 | 0.94 |
| StarCoder @4k (no FT) | 185 (59.7%) | 67 (21.6%) | 18.76 | 0.83 |
| StarCoder @8k (no FT) | 255 (82.3%) | 95 (30.6%) | 14.30 | 0.67 |
| PPatHF @2k | 150 (48.4%) | 72 (23.2%) | 19.80 | 0.70 |
| PPatHF @4k | 242 (78.1%) | 111 (35.8%) | 14.45 | 0.51 |
| **PPatHF @8k (headline)** | 290 (93.5%) | **131 (42.3%)** | **11.98** | **0.43** |
| **Ours: qwen3-coder:30b zero-shot, full inputs** | **300 (96.8%)** | **122 (39.4%)** | **14.31** | **2.11** |
| Ours: same, reduced inputs (their reduction output) | 302 (97.4%) | 94 (30.3%) | 21.98 | 2.73 |

Completions-only denominators (not dummy-substituted, cases the model
actually generated a complete candidate for):

| Run | n | Accuracy | AED | RED (mean) |
|---|---|---|---|---|
| Ours, full inputs | 300 | 122 (40.7%) | 13.95 | 2.15 |
| Ours, reduced inputs | 302 | 94 (31.1%) | 21.97 | 2.78 |

Non-completions, full-input run: 9 `limits_exceeded` (prompt > 65,536 chars;
max prompt in the set is 180,531 chars), 1 `backend_error` (a 55k-char case
that hit the 600 s timeout while generating a very long function). Reduced
run: 8 `limits_exceeded`. All were scored as dummy per the paper's protocol.

### On the RED number

RED is a **mean of per-case ratios** and is destroyed by a handful of cases
whose *developer* target change was tiny (src_distance 1-7 tokens) but where
our zero-shot model rewrote the function toward the Vim implementation:
top-5 per-case REDs are 263, 66, 45, 34, 24. Distribution over the 310
full-input cases: **122 cases RED = 0** (exact), **148 cases 0 < RED <= 1**,
**40 cases RED > 1** (model made the function worse than doing nothing).
Median RED = **0.14**; mean excluding the top-5 outliers = 0.73. The paper's
fine-tuned model learned to stay conservative on the fork side, so its ratio
mean is not outlier-dominated; our zero-shot mean is. We report their exact
formula's value (2.11) and flag it as incomparable-in-spirit rather than
substituting a friendlier statistic for the headline.

### Reduced-input pass (their reduction module's output)

This is the closest analogue to RCAP's Program-Reduction-Only configuration:
same prompts but over `func_*_sliced_*` with an added RCAP-style placeholder
preservation instruction, and their `do_recovering` applied before scoring.
Zero-shot, the reduction **hurts**: 94 vs 122 exact. Two mechanisms observed:
(a) the model must reproduce `/* placeholder_N */` comments byte-exactly and
sometimes drops or rewrites them (11 unrecoverable-placeholder warnings
during scoring => the removed code is not restored => large distances);
(b) with a 16k context, reduction only converts 1 refusal (9->8), so its
paper benefit -- fitting more cases under a tight limit -- barely applies.
This mirrors the paper's own Table 2 finding that reduction's overall gain
comes almost entirely from the *Unique* (newly-fitting) cases, not from
making shared cases easier, and matches our RCAP observation that reduction
is a capacity tool, not an accuracy tool.

## 5. Honest confounds

Differences between the two rows attribute to the **whole stack**, not to
any single component:

1. **Model family and size**: StarCoder-15.5B (2023) vs Qwen3-Coder-30B MoE
   (2025, ~3.3B active params) -- different pretraining data, incl. almost
   certainly *post-2022-07 Neovim code*, which is exactly what the paper's
   date cutoff was designed to exclude for StarCoder. Our model may have
   seen some ground-truth patches during pretraining; we cannot rule out
   test-set leakage, the paper could (for their model).
2. **Quantization**: Q4_K_M 4-bit vs fp16.
3. **Fine-tuned vs zero-shot**: their headline row had LoRA fine-tuning on
   4,829 in-distribution commits; ours is pure zero-shot (that comparison is
   the point, but it cuts both ways with confound 1).
4. **Prompt format**: RCAP sectioned-markdown prompt vs their completion-style
   `###` template.
5. **Context budget**: 16k tokens vs 8k max -- our model attempts 300
   generations vs their 290; on their-denominator terms this inflates our
   "Complete" but also exposes us to the hardest long cases.
6. **Refusal policy**: our 65,536-char refusal is RCAP's, not theirs (they
   filter by model tokenizer length). Both end in dummy substitution, so the
   aligned metrics remain comparable.
7. **Scorer re-implementation**: modern tree-sitter wheel vs their pinned
   0.20.1 build (bounded at ~0.2% AED skew by the Origin-row validation);
   candidate extraction from a ```c fence vs their `###`-split postprocess
   (equivalent role); reduced-pass recovery via literal placeholder
   replacement vs their AST-located replacement (same effect when the
   placeholder comment survives verbatim, which is also what their tolerant
   mode assumes).

## 6. Example cases (full-input run)

**Exact match -- `019_159a0b651f`** (src_distance 75, non-trivial): the Vim
patch removes the whole `file_id` bookkeeping block; the Neovim function is
heavily diverged (different APIs `os_fileid`/`FNAMECMP`, different brace
style). The model reproduced the developer's multi-hunk deletion
token-for-token (diff vs ground truth: empty).

```
dev change (excerpt):        model vs ground truth:
-  FileID file_id;           (no differences)
-  bool file_id_ok = os_fileid((char *)fname, &file_id);
-    if (si->sn_name != NULL) { ... }
+    if (si->sn_name != NULL && FNAMECMP(si->sn_name, fname) == 0) {
```

**Near miss -- `011_d9e5737fdc`** (gen_distance 2): model inserted the correct
new logic at the correct place but wrote it in Vim's unbraced style; ground
truth uses Neovim's mandatory-brace style. 2 tokens off:

```
model:                          ground truth:
          if (*that == NUL)               if (*that == NUL) {
            break;                          break;
                                          }
```

**Failure -- `073_06f9da547c`** (src_distance 1, gen_distance 263, the RED=263
outlier): the developer's entire ported change was `lnum` -> `lnum
FUNC_ATTR_UNUSED` in the signature of `init_chartabsize_arg`. The model
instead back-ported Vim's whole function body (`FEAT_PROP_POPUP` text-prop
code, `CLEAR_POINTER`, Vim macros) into Neovim -- porting the
*implementation* rather than the *patch*. This over-porting mode is what
fine-tuning suppressed in the paper and is the main driver of our RED
blow-up.

**Capacity refusals -- 9 cases** with prompts up to 180k chars (whole-file-
sized functions), refused at the RCAP 65,536-char limit -- the same
oversized-request refusal behavior our RCAP sweep produced, and the cases
the paper's reduction module exists for.

## 7. Runtime and tokens

| Run | cases called | wall (sum of calls) | input tokens | output tokens |
|---|---|---|---|---|
| Full inputs (`results.jsonl`) | 301 | 4.75 h | 925,494 | 306,517 |
| Reduced inputs (`results-reduced.jsonl`) | 302 | 2.82 h | 616,402 | 186,897 |
| **Total** | 603 | **7.57 h** | **1,541,896** | **493,414** |

Mean ~57 s/call (full) and ~34 s/call (reduced) on the RTX 2070 SUPER 8 GB
hybrid setup; temperature 0, seed 12535, num_ctx 16384 throughout.

## 8. Takeaway

A 2025 zero-shot 30B/4-bit local model with an RCAP-style prompt lands
**within 3 accuracy points of the paper's fine-tuned headline system**
(39.4% vs 42.3% exact on the identical 310-case benchmark, identical
metrics and failure protocol) and clearly beats every non-fine-tuned
baseline they report (best: StarCoder@8k, 30.6%). Its failure profile is
qualitatively different, though: when it fails it fails big (over-porting
the source implementation), so the edit-distance metrics that reward
conservative failure (AED 14.31 vs 11.98; mean RED 2.11 vs 0.43, median
0.14) still favor the fine-tuned system. The paper's reduction module,
applied to our setup, reduces accuracy (94/310) because its capacity benefit
is mostly voided by the larger context window while its placeholder
discipline adds a new zero-shot failure mode.

## Files

- `harness.py` -- generation harness (resumable, refusals, truncation guard)
- `score.py` -- paper-faithful metrics (provenance in docstring)
- `results.jsonl`, `results-reduced.jsonl` -- per-case raw generations
- `scored-main.json`, `scored-reduced.json` -- per-case metric rows
- `run-main.log`, `run-reduced.log` -- per-case progress logs
- `PPatHF/` -- their replication package (clone, unmodified)
- `data/` -- their dataset (Google Drive download)
