"""Adaptation Context construction (RCAP ref section 9).

Organizes retained semantic evidence and reduced program representations into
the canonical, backend-independent context. Fixes WHAT information the
adaptation uses; formatting belongs to Request Synthesis. Every element is
traceable to SAP evidence (I2/I3); characterization rides as a reference, never
as content (I10); dispositions stay distinct (I4); no developer target solution
exists anywhere upstream to leak (I13). A context failing any validity
condition is a context-construction failure.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from rcap.materialize import MaterializationRecord
from rcap.model import CaseModel
from rcap.reduction_program import (
    ContextConstructionFailure,
    ProgramReductionRecord,
    ReducedArtifact,
)
from rcap.reduction_semantic import Disposition, SemanticReductionManifest


class CompositeCaseUnsupported(Exception):
    """Typed refusal for multi-entity cases until section-13 per-unit processing
    exists. Refusing is mandatory here: building a context from only the first
    entity would silently misrepresent a composite case as fully covered."""

    def __init__(self, case_id: str, entities: list[str]):
        self.stage = "context_construction"
        detail = (f"composite case: {len(entities)} materialized entities "
                  f"({', '.join(entities)}); per-unit processing (spec section 13) "
                  f"is not implemented — refusing rather than truncating to the first entity")
        self.diagnostics = [detail]
        super().__init__(f"composite case unsupported for {case_id}: {self.diagnostics[0]}")


class ContextRelationship(BaseModel):
    id: str
    type: str
    src: str
    dst: str


class ContextConstraint(BaseModel):
    id: str
    kind: str          # e.g. rename | dependency_constraint | blocking_conflict
    detail: str
    evidence_ref: str  # retained evidence object id
    blocking: bool = False


class Correspondence(BaseModel):
    type: str = "one_to_one"  # ONE_TO_ONE | SPLIT | MERGE | RELOCATE | NEW
    uncertain: bool = False
    reason: str | None = None


class AdaptationContext(BaseModel):
    case_id: str
    transformation: dict[str, str]              # role -> reduced text
    edit_regions: list[str]
    target_localization: dict[str, str | None]  # file, function
    relationships: list[ContextRelationship] = Field(default_factory=list)
    constraints: list[ContextConstraint] = Field(default_factory=list)
    program_context: list[ReducedArtifact] = Field(default_factory=list)
    correspondence: Correspondence = Field(default_factory=Correspondence)
    dispositions: dict[str, list[str]] = Field(default_factory=dict)
    metadata_refs: dict[str, str] = Field(default_factory=dict)  # sap/characterization refs
    reduction_mode: dict[str, str] = Field(default_factory=dict)


def build_context(
    case: CaseModel,
    reduction: SemanticReductionManifest,
    materialized: MaterializationRecord,
    program: ProgramReductionRecord,
) -> AdaptationContext:
    if len(reduction.materialize) > 1:
        raise CompositeCaseUnsupported(case.case_id, list(reduction.materialize))
    entity = reduction.materialize[0] if reduction.materialize else None
    if entity is None:
        raise ContextConstructionFailure(case.case_id, ["no program entity to adapt"])
    arts = {a.role: a for a in program.artifacts if a.entity == entity}
    for role in ("source.before", "source.after", "target"):
        if role not in arts:
            raise ContextConstructionFailure(case.case_id, [f"missing reduced artifact {role}"])

    retained = reduction.retained_ids()
    relationships = [
        ContextRelationship(
            id=f"{e.src}-{e.rel}->{e.dst}", type=e.rel, src=e.src, dst=e.dst)
        for e in reduction.retained_edges
    ]

    constraints: list[ContextConstraint] = []
    for oid in sorted(retained):
        rec = case.evidence.get(oid)
        if rec is None:
            continue
        if rec.blocking_conflict:
            constraints.append(ContextConstraint(
                id=f"constraint:{oid}", kind="blocking_conflict",
                detail=str(rec.attributes or rec.element), evidence_ref=oid, blocking=True))
        elif rec.category == "refactoring" and rec.state == "PRESENT":
            a = rec.attributes
            constraints.append(ContextConstraint(
                id=f"constraint:{oid}", kind=str(a.get("kind", "refactoring")),
                detail=f"{a.get('from', '?')} -> {a.get('to', '?')}", evidence_ref=oid))

    # Validity: every retained relationship endpoint and constraint traceable to the SAP.
    problems = [f"constraint {c.id} evidence {c.evidence_ref} is not retained"
                for c in constraints if c.evidence_ref not in retained]
    if problems:
        raise ContextConstructionFailure(case.case_id, problems)

    dispositions = {d.value: sorted(oid for oid, dd in reduction.dispositions.items() if dd is d)
                    for d in Disposition}

    return AdaptationContext(
        case_id=case.case_id,
        transformation={role: arts[role].reduced_text for role in arts},
        edit_regions=[er for t in case.transformations for er in t.edit_regions],
        target_localization={"file": case.target_file,
                             "function": entity},
        relationships=relationships,
        constraints=constraints,
        program_context=[arts[r] for r in ("source.before", "source.after", "target")],
        correspondence=Correspondence(
            uncertain=True, reason="fidelity flags not serialized by current SALP output"),
        dispositions=dispositions,
        metadata_refs={"sap": case.sap_id,
                       "characterization": f"{case.sap_id}/characterization.json"},
        reduction_mode={h: m.value for h, m in reduction.mode.items()},
    )
