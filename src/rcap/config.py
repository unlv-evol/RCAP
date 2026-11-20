"""Execution configuration: everything needed to reproduce a run (RCAP ref section 2).

The configuration is recorded in provenance and echoed into every produced
artifact, so a run is reconstructable from its outputs alone.
"""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Literal

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
    # Never placehold a target line an upstream hunk maps to (rcap.reduction_program
    # .target_hunk_lines). Off reproduces the reduction of ede1b40 and earlier.
    protect_target_hunks: bool = True
    # Edit evidence (rcap.evidence): a function unit's prompt also shows the same
    # change's edits outside functions in its file (removed/added lines) and the
    # signatures of the functions the PR edits in its other files. Off reproduces
    # the prompts of c24f060 and earlier byte for byte.
    edit_evidence: bool = False
    # Relevance filter of the edit evidence (rcap.evidence, "Relevance filter"): "none" shows every
    # entry (the prompts of 08e7a59); "names" keeps only the entries whose declared, imported or
    # re-signed names the function uses. No effect unless edit_evidence is on.
    evidence_filter: Literal["none", "names"] = "none"
    # Placeholder scheme of the program cut (rcap.reduction_program, rcap.generate).
    # "rcap-v1": each of source.before, source.after and the target is cut on its own, with
    # per-side ids (source.before.N / source.after.N / target.N); the prompt tells the model to
    # keep the placeholders; an answer that loses, repeats or invents one is rejected
    # (placeholder_violation). "shared-v1" is PPatHF's scheme: a block is cut only when it is in
    # all three sides, under ONE id the three share (copying upstream code then copies a
    # placeholder the fork has too); the prompt says nothing about placeholders; and an answer
    # is recovered tolerantly: a lost placeholder drops its code and is counted
    # (GenerationRecord.placeholders_lost), a repeated or invented one is removed. Default
    # "rcap-v1" reproduces the prompts and outcomes of 11b66e4 and earlier.
    placeholder_scheme: Literal["rcap-v1", "shared-v1"] = "rcap-v1"
