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
                  "program_reduction", "packaging")


class EvalRow(BaseModel):
    """One (case, configuration) measurement row (section 15 result schema)."""

    case_id: str
    config_id: str
    mode: str                                        # the harness configuration
    reduction_mode: dict[str, str] = Field(default_factory=dict)  # per hunk
    dispositions: dict[str, int] = Field(default_factory=dict)    # never collapsed
    nodes_before: int = 0
    nodes_after: int = 0
    placeholders: int = 0
    context_chars: int = 0
    prompt_chars: int = 0
    request_hash: str | None = None
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
    rows = []
    for mode in modes:
        config = base.model_copy(update={"mode": mode,
                                         "config_id": f"{base.config_id}:{mode.value}"})
        t0 = time.time()
        try:
            result = run_case(sap_dir, pr_manifest, backend, config)
            row = EvalRow(
                case_id=result.case.case_id,
                config_id=config.config_id,
                mode=mode.value,
                reduction_mode={h: m.value for h, m in result.reduction.mode.items()},
                dispositions=_disposition_counts(result.reduction),
                nodes_before=sum(a.nodes_before for a in result.program.artifacts),
                nodes_after=sum(a.nodes_after for a in result.program.artifacts),
                placeholders=sum(len(a.placeholders) for a in result.program.artifacts),
                context_chars=len(result.context.model_dump_json()),
                prompt_chars=len(result.request.prompt),
                request_hash=result.request.request_hash,
                outcome=result.generation.outcome,
            )
            if artifacts_dir is not None:
                mode_dir = artifacts_dir / mode.value
                mode_dir.mkdir(parents=True, exist_ok=True)
                (mode_dir / "request.txt").write_text(
                    result.request.prompt, encoding="utf-8")
                if result.generation.outcome == "completion":
                    (mode_dir / "candidate.java").write_text(
                        result.generation.candidate, encoding="utf-8")
        except Exception as exc:
            stage = getattr(exc, "stage", None)
            if stage not in TYPED_FAILURES:
                raise
            row = EvalRow(case_id=str(sap_dir), config_id=config.config_id,
                          mode=mode.value, outcome=f"failure:{stage}")
        row.runtime_ms = int((time.time() - t0) * 1000)
        rows.append(row)

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a", encoding="utf-8") as fh:
            for row in rows:
                fh.write(row.model_dump_json() + "\n")
    return rows


def _disposition_counts(reduction) -> dict[str, int]:
    counts: dict[str, int] = {}
    for d in reduction.dispositions.values():
        counts[d.value] = counts.get(d.value, 0) + 1
    return counts
