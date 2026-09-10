"""Execution configuration: everything needed to reproduce a run (RCAP ref section 2).

The configuration is recorded in provenance and echoed into every produced
artifact, so a run is reconstructable from its outputs alone.
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field

_CONFIGS = Path(__file__).resolve().parent.parent.parent / "configs"


class EvaluationMode(StrEnum):
    """The four evaluation-harness configurations (RCAP ref section 15)."""

    RAW = "raw_context"
    PROGRAM_ONLY = "program_reduction_only"
    SEMANTIC_ONLY = "semantic_reduction_only"
    FULL = "full_rcap"

    @property
    def semantic_reduction(self) -> bool:
        return self in (EvaluationMode.SEMANTIC_ONLY, EvaluationMode.FULL)

    @property
    def program_reduction(self) -> bool:
        return self in (EvaluationMode.PROGRAM_ONLY, EvaluationMode.FULL)


class AdmissiblePolicy(BaseModel):
    """Which relationship types semantic reduction may traverse; versioned data, not code."""

    policy_id: str
    admissible: frozenset[str]
    inadmissible: frozenset[str] = frozenset()

    @classmethod
    def load(cls, name: str = "admissible-v1") -> AdmissiblePolicy:
        raw = json.loads((_CONFIGS / f"{name}.json").read_text(encoding="utf-8"))
        return cls(
            policy_id=raw["policy_id"],
            admissible=frozenset(raw["admissible"]),
            inadmissible=frozenset(raw.get("inadmissible", [])),
        )


class ExecutionConfig(BaseModel):
    """One run's recorded configuration (mode, policy, template, backend, params)."""

    config_id: str = "dev"
    mode: EvaluationMode = EvaluationMode.FULL
    admissible_policy: str = "admissible-v1"
    template_id: str = "rcap-request-v1"
    template_version: str = "1"
    backend: str = "stub"
    generation_params: dict[str, object] = Field(default_factory=dict)
    # Program-context reduction tuning (recorded, never hard-coded).
    min_placeholder_lines: int = 3
