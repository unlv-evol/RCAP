"""The four-configuration evaluation harness (RCAP ref section 15).

Runs the same eligible case through Raw Context / Program-Reduction-Only /
Semantic-Reduction-Only / Full RCAP — one implementation, two toggles — and
measures the transitions between the intermediate artifacts. Measurements use
one shared definition across modes; dispositions are reported separately,
never collapsed (I4). No correctness metric appears in any row (I11):
correctness is downstream, in SVRP.
"""

from __future__ import annotations

import time
from pathlib import Path

from pydantic import BaseModel, Field

from rcap.config import EvaluationMode, ExecutionConfig
from rcap.generate import Backend
from rcap.pipeline import run_case

TYPED_FAILURES = ("intake", "semantic_reduction", "materialization",
                  "program_reduction", "context_construction", "packaging")


class EvalRow(BaseModel):
    """One (case, configuration) measurement row (section 15 result schema)."""

    case_id: str
    config_id: str
    ref_config_id: str = ""                          # the Raw Context config percentages reference
    mode: str                                        # the harness configuration
    reduction_mode: dict[str, str] = Field(default_factory=dict)  # per hunk
    dispositions: dict[str, int] = Field(default_factory=dict)    # never collapsed
    nodes_before: int = 0
    nodes_after: int = 0
    placeholders: int = 0
    context_chars: int = 0
    prompt_chars: int = 0
    # Backend-reported token counts, recorded only when the runtime reports
    # them (section 15: "measured separately"); never estimated. The context
    # itself is backend-independent and has no honest token count without a
    # tokenizer, so context size stays in characters (context_chars).
    input_tokens: int | None = None
    output_tokens: int | None = None
    # SAP characterization, explanatory metadata only (I10): min-over-hunks
    # Coverage/Fidelity scores and the aggregate Readiness level.
    coverage: float | None = None
    fidelity: float | None = None
    readiness: str | None = None
    request_hash: str | None = None
    # Section 13: processing units in the case (1 for a plain case). Size and
    # token measurements are sums over units; outcome is completion only when
    # every unit completed, else the first non-completing unit's state.
    units: int = 1
    unit_outcomes: dict[str, str] = Field(default_factory=dict)  # entity -> outcome
    outcome: str = ""                                # completion | failure:<stage> | gen state
    runtime_ms: int = 0


def run_configs(
    sap_dir, pr_manifest: dict | None, backend: Backend,
    modes: tuple[EvaluationMode, ...] = tuple(EvaluationMode),
    base_config: ExecutionConfig | None = None,
    out_path: Path | None = None,
    artifacts_dir: Path | None = None,
) -> list[EvalRow]:
    base = base_config or ExecutionConfig()
    ref_config_id = f"{base.config_id}:{EvaluationMode.RAW.value}"
    rows = []
    for mode in modes:
        config = base.model_copy(update={"mode": mode,
                                         "config_id": f"{base.config_id}:{mode.value}"})
        t0 = time.time()
        try:
            result = run_case(sap_dir, pr_manifest, backend, config)
            scores = result.case.characterization_scores()
            gens = [ur.generation for ur in result.units]
            outcome = next((g.outcome for g in gens if g.outcome != "completion"),
                           "completion")
            row = EvalRow(
                case_id=result.case.case_id,
                config_id=config.config_id,
                ref_config_id=ref_config_id,
                mode=mode.value,
                reduction_mode={h: m.value for h, m in result.reduction.mode.items()},
                dispositions=_disposition_counts(result.reduction),
                nodes_before=sum(a.nodes_before for a in result.program.artifacts),
                nodes_after=sum(a.nodes_after for a in result.program.artifacts),
                placeholders=sum(len(a.placeholders) for a in result.program.artifacts),
                context_chars=sum(len(ur.context.model_dump_json())
                                  for ur in result.units),
                prompt_chars=sum(len(ur.request.prompt) for ur in result.units),
                input_tokens=_summed_usage(gens, "input_tokens"),
                output_tokens=_summed_usage(gens, "output_tokens"),
                coverage=scores["coverage"],
                fidelity=scores["fidelity"],
                readiness=scores["readiness"],
                request_hash=result.request.request_hash,
                units=len(result.units),
                unit_outcomes={ur.unit.entity: ur.generation.outcome
                               for ur in result.units if ur.unit is not None},
                outcome=outcome,
            )
            if artifacts_dir is not None:
                mode_dir = artifacts_dir / mode.value
                mode_dir.mkdir(parents=True, exist_ok=True)
                for i, ur in enumerate(result.units):
                    suffix = "" if len(result.units) == 1 else f"-unit{i + 1}"
                    (mode_dir / f"request{suffix}.txt").write_text(
                        ur.request.prompt, encoding="utf-8")
                    if ur.generation.outcome == "completion":
                        (mode_dir / f"candidate{suffix}.java").write_text(
                            ur.generation.candidate, encoding="utf-8")
        except Exception as exc:
            stage = getattr(exc, "stage", None)
            if stage not in TYPED_FAILURES:
                raise
            # Stable case identity (never a filesystem path) and the dispositions
            # established before the failure (section 14 / section 20).
            sap = Path(sap_dir)
            case_id = getattr(exc, "case_id", None) or f"{sap.parent.name}/{sap.name}"
            scores = getattr(exc, "characterization", None) or {}
            row = EvalRow(case_id=case_id, config_id=config.config_id,
                          ref_config_id=ref_config_id,
                          mode=mode.value, outcome=f"failure:{stage}",
                          dispositions=getattr(exc, "dispositions", {}),
                          coverage=scores.get("coverage"),
                          fidelity=scores.get("fidelity"),
                          readiness=scores.get("readiness"))
        row.runtime_ms = int((time.time() - t0) * 1000)
        rows.append(row)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(row.model_dump_json() + "\n")
    return rows


def _summed_usage(gens, key: str) -> int | None:
    """Sum a backend-reported token count over units; None unless every unit
    reported it (a partial sum would understate silently)."""
    values = [g.usage.get(key) for g in gens]
    if any(not isinstance(v, int) for v in values):
        return None
    return sum(values)


def _disposition_counts(reduction) -> dict[str, int]:
    counts: dict[str, int] = {}
    for d in reduction.dispositions.values():
        counts[d.value] = counts.get(d.value, 0) + 1
    return counts
