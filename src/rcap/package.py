"""Adaptation Package construction (RCAP ref section 12): the handoff to SVRP.

References rather than copies: the exact context (by hash), the originating
SAP, characterization as metadata, generation config, and provenance down to
the repository states. Carries what SVRP needs and only RCAP can preserve —
the placeholder recovery map, the correspondence type, and the evidence
dispositions (reduced-as-irrelevant stays distinguishable from
unavailable-to-SALP). No downstream validation outcome and no prior
integration result may appear (I11, I12): the model has no field for one.
"""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, Field

from rcap.context import AdaptationContext
from rcap.generate import GenerationRecord
from rcap.model import CaseModel
from rcap.reduction_program import PlaceholderEntry


class PackagingFailure(Exception):
    def __init__(self, case_id: str, diagnostics: list[str]):
        self.stage = "packaging"
        self.diagnostics = diagnostics
        super().__init__(f"packaging failure for {case_id}: {'; '.join(diagnostics)}")


class AdaptationPackage(BaseModel):
    case_id: str
    candidate: dict[str, str]                       # artifact + output_format
    recovery_map: list[PlaceholderEntry] = Field(default_factory=list)
    context_ref: str                                # content hash of the exact context
    sap_ref: str
    characterization: dict[str, object] = Field(default_factory=dict)  # metadata (I10)
    correspondence: dict[str, object] = Field(default_factory=dict)
    request_ref: dict[str, str] = Field(default_factory=dict)
    generation_meta: dict[str, object] = Field(default_factory=dict)
    provenance: dict[str, object] = Field(default_factory=dict)
    outcome: str = "completion"


def build_package(
    case: CaseModel,
    context: AdaptationContext,
    generation: GenerationRecord,
) -> AdaptationPackage:
    problems = []
    if generation.outcome != "completion" or not generation.candidate:
        problems.append(f"no packageable candidate: generation outcome {generation.outcome}")
    if generation.request_hash == "":
        problems.append("request is not identifiable")
    if problems:
        raise PackagingFailure(case.case_id, problems)

    context_ref = hashlib.sha256(context.model_dump_json().encode()).hexdigest()
    target_art = next(a for a in context.program_context if a.role == "target")

    return AdaptationPackage(
        case_id=case.case_id,
        candidate={"artifact": generation.candidate, "output_format": "java_function"},
        recovery_map=target_art.placeholders,
        context_ref=context_ref,
        sap_ref=case.sap_id,
        characterization=dict(context_aggregate(case)),
        correspondence=context.correspondence.model_dump(),
        request_ref={"template_id": "rcap-request-v1", "template_version": "1",
                     "request_hash": generation.request_hash},
        generation_meta={**generation.generation_config, "exec_id": generation.exec_id},
        provenance={
            "repo_state": case.repo_state,
            "source_repo": case.source_repo,
            "target_repo": case.target_repo,
            "evidence_dispositions": context.dispositions,
            "reduction_mode": context.reduction_mode,
            "trace_path": ["candidate", "request", "context", "selection", "sap"],
        },
        outcome="completion",
    )


def context_aggregate(case: CaseModel) -> dict[str, object]:
    agg = case.characterization.get("aggregate")
    return agg if isinstance(agg, dict) else {}
