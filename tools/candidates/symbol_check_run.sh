#!/bin/bash
# Symbol check of one unpacked cand32k run (results-*.jsonl + gen-cache), all configurations, then the table.
#   bash symbol_check_run.sh <run dir, e.g. .../qwen3-coder_30b-cand32k> [<ablation run dir ...-cand32k-abl>]
# Env: PY (python with RCAP's deps), SAP_ROOT (default out-a), REPO_CACHE (bare clones), STUDY (inputs/),
#      UNITS (default shared-units.tsv), OVERRIDES (RCAP replay overrides; default tools/candidates/cand32k-overrides.json).
# Writes <run dir>/symcheck/{direct,ppathf,rcap-full,rcap-noprog[,abl]}.jsonl and summary.md / summary.json.
set -eu
RUN=$(cd "${1:?usage: symbol_check_run.sh RUN_DIR [ABL_RUN_DIR]}" && pwd); ABL=${2:-}
HERE=$(cd "$(dirname "$0")" && pwd)
P=/home/adam/Documents/work/rcap-class-project
PY=${PY:-$P/3-studies/2026-10-pipeline-fixes-yardstick-dialogue/rcap/.venv/bin/python}
SAP_ROOT=${SAP_ROOT:-$P/1-sap-inputs/sap-2026-11/out-a}
REPO_CACHE=${REPO_CACHE:-/home/adam/Documents/work/SALP/data/repos}
STUDY=${STUDY:-$P/3-studies/2026-11-candidates-32k}
UNITS=${UNITS:-$HERE/shared-units.tsv}
OV=(); [ -n "${OVERRIDES:-}" ] && OV=(--overrides "$OVERRIDES")
O=$RUN/symcheck; mkdir -p "$O"
C=(--sap-root "$SAP_ROOT" --repo-cache "$REPO_CACHE" --units "$UNITS")
"$PY" "$HERE/symbol_check.py" --config direct --results "$RUN/results-direct.jsonl" "${C[@]}" --out "$O/direct.jsonl"
"$PY" "$HERE/symbol_check.py" --config ppathf --results "$RUN/results-ppathf.jsonl" --inputs "$STUDY/inputs/ppathf-all.jsonl" \
  "${C[@]}" --out "$O/ppathf.jsonl"
"$PY" "$HERE/symbol_check.py" --config rcap-full --results "$RUN/results-rcap-full_rcap.jsonl" --gen-cache "$RUN/gen-cache" \
  ${OV[@]+"${OV[@]}"} "${C[@]}" --out "$O/rcap-full.jsonl"
"$PY" "$HERE/symbol_check.py" --config rcap-noprog --results "$RUN/results-rcap-semantic_reduction_only.jsonl" \
  --gen-cache "$RUN/gen-cache" ${OV[@]+"${OV[@]}"} "${C[@]}" --out "$O/rcap-noprog.jsonl"
OUTS=("$O/direct.jsonl" "$O/ppathf.jsonl" "$O/rcap-full.jsonl" "$O/rcap-noprog.jsonl")
if [ -n "$ABL" ]; then
  ABL=$(cd "$ABL" && pwd)
  "$PY" "$HERE/symbol_check.py" --config abl --results "$ABL/results-rcap-full_rcap.jsonl" --gen-cache "$ABL/gen-cache" \
    --gen-cache "$RUN/gen-cache" ${OV[@]+"${OV[@]}"} "${C[@]}" --out "$O/abl.jsonl"
  OUTS+=("$O/abl.jsonl")
fi
cd "$HERE" && "$PY" symbol_summary.py "${OUTS[@]}" --units "$UNITS" --md "$O/summary.md" --json "$O/summary.json"
