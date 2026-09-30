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

# I11/I12: keys that would smuggle a downstream validation result or a prior
# integration outcome into the package. Their appearance anywhere in the
# package structure fails packaging validation (section 17 negative tests).
FORBIDDEN_DOWNSTREAM_MARKERS = (
    "build_result", "test_result", "tests_passed", "validation_result",
    "compile_result", "correctness", "prior_integration", "integration_outcome",
)


class PackagingFailure(Exception):
    def __init__(self, case_id: str, diagnostics: list[str]):
        self.stage = "packaging"
        self.diagnostics = diagnostics
        super().__init__(f"packaging failure for {case_id}: {'; '.join(diagnostics)}")


class UnitEntry(BaseModel):
    """One processing unit inside a composite case's single package (section 13)."""

    entity: str
    hunk_ids: list[str]
    coupling_basis: str | None = None
    coupling_note: str | None = None
    candidate: dict[str, str]
    recovery_map: list[PlaceholderEntry] = Field(default_factory=list)
    context_ref: str
    request_ref: dict[str, str] = Field(default_factory=dict)
    generation_meta: dict[str, object] = Field(default_factory=dict)


class AdaptationPackage(BaseModel):
    case_id: str
    candidate: dict[str, str]                       # artifact + output_format
    recovery_map: list[PlaceholderEntry] = Field(default_factory=list)
    context_ref: str                                # content hash of the exact context
    sap_ref: str
    # Section 13: one package per case; a composite case carries every unit
    # here (in application order) and the coupling record — including whether
    # each coupling was asserted by SALP or inferred from co-location — so
    # downstream evaluation can separate incomplete alignment from intentional
    # reduction. Empty for a plain single-unit case.
    units: list[UnitEntry] = Field(default_factory=list)
    coupling: dict[str, object] = Field(default_factory=dict)
    characterization: dict[str, object] = Field(default_factory=dict)  # metadata (I10)
    correspondence: dict[str, object] = Field(default_factory=dict)
    request_ref: dict[str, str] = Field(default_factory=dict)
    generation_meta: dict[str, object] = Field(default_factory=dict)
    provenance: dict[str, object] = Field(default_factory=dict)
    outcome: str = "completion"


def _unit_entries(units) -> list[UnitEntry]:
    entries = []
    for ur in units:
        if ur.unit is None:
            continue
        ctx, gen = ur.context, ur.generation
        target_art = next(a for a in ctx.program_context if a.role == "target")
        entries.append(UnitEntry(
            entity=ur.unit.entity, hunk_ids=ur.unit.hunk_ids,
            coupling_basis=ur.unit.coupling_basis,
            coupling_note=ur.unit.coupling_note,
            candidate={"artifact": gen.candidate, "output_format": "java_function"},
            recovery_map=target_art.placeholders,
            context_ref=hashlib.sha256(ctx.model_dump_json().encode()).hexdigest(),
            request_ref={"template_id": "rcap-request-v1", "template_version": "1",
                         "request_hash": gen.request_hash},
            generation_meta={**gen.generation_config, "exec_id": gen.exec_id,
                             "executed_at": gen.executed_at},
        ))
    return entries


def build_package(
    case: CaseModel,
    context: AdaptationContext,
    generation: GenerationRecord,
    *,
    units=None,
    coupling=None,
) -> AdaptationPackage:
    problems = []
    for gen in [generation, *[u.generation for u in (units or [])]]:
        if gen.outcome != "completion" or not gen.candidate:
            problems.append(f"no packageable candidate: generation outcome {gen.outcome}")
        if gen.request_hash == "":
            problems.append("request is not identifiable")
    if problems:
        raise PackagingFailure(case.case_id, sorted(set(problems)))

    context_ref = hashlib.sha256(context.model_dump_json().encode()).hexdigest()
    target_art = next(a for a in context.program_context if a.role == "target")

    composite = units is not None and len(units) > 1
    package = AdaptationPackage(
        units=_unit_entries(units) if composite else [],
        coupling=(coupling.model_dump(exclude={"case_id"})
                  if composite and coupling is not None else {}),
        case_id=case.case_id,
        candidate={"artifact": generation.candidate, "output_format": "java_function"},
        recovery_map=target_art.placeholders,
        context_ref=context_ref,
        sap_ref=case.sap_id,
        # Section 12: coverage/fidelity/readiness as metadata (I10) — the
        # min-over-hunks scores alongside SALP's aggregate record.
        characterization={**context_aggregate(case),
                          **{k: v for k, v in case.characterization_scores().items()
                             if v is not None}},
        correspondence=context.correspondence.model_dump(),
        request_ref={"template_id": "rcap-request-v1", "template_version": "1",
                     "request_hash": generation.request_hash},
        generation_meta={**generation.generation_config,
                         "exec_id": generation.exec_id,
                         "executed_at": generation.executed_at},
        provenance={
            "repo_state": case.repo_state,
            "source_repo": case.source_repo,
            "target_repo": case.target_repo,
            # Section 13 partial mappings: the mapped/unmapped distinction must
            # stay visible downstream and never read as deliberate reduction.
            "change_type": case.change_type,
            "evidence_dispositions": context.dispositions,
            "reduction_mode": context.reduction_mode,
            "trace_path": ["candidate", "request", "context", "selection", "sap"],
        },
        outcome="completion",
    )
    validate_package(package)
    return package


def context_aggregate(case: CaseModel) -> dict[str, object]:
    agg = case.characterization.get("aggregate")
    return agg if isinstance(agg, dict) else {}


def _keys_recursive(value: object):
    if isinstance(value, dict):
        for k, v in value.items():
            yield str(k)
            yield from _keys_recursive(v)
    elif isinstance(value, list):
        for v in value:
            yield from _keys_recursive(v)


def validate_package(package: AdaptationPackage) -> None:
    """Packaging validation (section 17): a downstream validation result or a
    prior integration outcome appearing anywhere in the package structure is a
    packaging failure (I11, I12). Keys are checked, not values — evidence ids
    like '...verification:covering_tests' are legitimate references."""
    offending = sorted({k for k in _keys_recursive(package.model_dump())
                        if any(m in k.lower() for m in FORBIDDEN_DOWNSTREAM_MARKERS)})
    if offending:
        raise PackagingFailure(package.case_id, [
            ("I11/I12 violation: downstream-result field(s) in package: "
             f"{', '.join(offending)}")])
