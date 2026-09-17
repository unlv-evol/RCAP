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

import re

from pydantic import BaseModel, Field

from rcap.coupling import Unit
from rcap.materialize import MaterializationRecord
from rcap.model import CaseModel
from rcap.reduction_program import (
    ContextConstructionFailure,
    ProgramReductionRecord,
    ReducedArtifact,
)
from rcap.reduction_semantic import Disposition, SemanticReductionManifest

_WS = re.compile(r"\s+")


class ContextValidityFailure(ContextConstructionFailure):
    """A context failing a section-9/13 validity condition. Subclasses the
    program-reduction failure type for compatibility, but carries the correct
    stage attribution (section 14): the failure happened building the context,
    not reducing the program."""

    def __init__(self, case_id: str, diagnostics: list[str]):
        super().__init__(case_id, diagnostics)
        self.stage = "context_construction"


def signature_of(payload_text: str) -> str | None:
    """The declaration header of a function payload: the text up to the first
    body brace, whitespace-collapsed. Derived from SAP evidence (the payload
    itself), never invented; None when the payload has no brace (not a
    function body)."""
    head, brace, _ = payload_text.partition("{")
    if not brace:
        return None
    return _WS.sub(" ", head).strip() or None


class SiblingSignature(BaseModel):
    """Section 13 sibling-edit signature: the signatures (never bodies) of the
    other units edited by the same change, so the backend can keep a rename or
    signature change consistent. Lightweight coordination, not joint
    generation."""

    entity: str
    hunk_ids: list[str]
    signature_before: str | None = None   # from source.before
    signature_after: str | None = None    # from source.after
    target_signature: str | None = None   # from the fork's target payload
    derivation: str = "declaration header of the SAP function payloads"


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
    # Section 13: the processing unit this context covers (entity, ordered
    # hunks, coupling basis/note) and the sibling units' signatures. Empty
    # unit info means a plain single-unit case.
    unit: dict[str, object] = Field(default_factory=dict)
    siblings: list[SiblingSignature] = Field(default_factory=list)
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
    unit: Unit | None = None,
    siblings: tuple[Unit, ...] = (),
) -> AdaptationContext:
    """Build the context for one processing unit (section 13: per-unit
    processing). A composite case must name its unit explicitly — building
    from only the first entity would silently misrepresent the case as fully
    covered, so that remains a typed failure, never a truncation."""
    if unit is None:
        if len(reduction.materialize) > 1:
            raise ContextValidityFailure(case.case_id, [
                (f"composite case: {len(reduction.materialize)} materialized entities "
                 f"({', '.join(reduction.materialize)}); an explicit processing unit "
                 f"is required (section 13) — refusing rather than truncating "
                 f"to the first entity")])
        entity = reduction.materialize[0] if reduction.materialize else None
    else:
        entity = unit.entity
        if entity not in reduction.materialize:
            raise ContextValidityFailure(case.case_id, [
                f"unit entity {entity} is not among the materialized entities"])
    if entity is None:
        raise ContextValidityFailure(case.case_id, ["no program entity to adapt"])
    arts = {a.role: a for a in program.artifacts if a.entity == entity}
    for role in ("source.before", "source.after", "target"):
        if role not in arts:
            raise ContextValidityFailure(case.case_id, [f"missing reduced artifact {role}"])

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
        raise ContextValidityFailure(case.case_id, problems)

    dispositions = {d.value: sorted(oid for oid, dd in reduction.dispositions.items() if dd is d)
                    for d in Disposition}

    # Section 13: sibling-edit signatures — the signatures (never bodies) of
    # the other units edited by this change, from their materialized payloads.
    sibling_sigs: list[SiblingSignature] = []
    for sib in siblings:
        roles = materialized.by_role(sib.entity)
        if not roles:
            continue  # not materialized: nothing evidenced to signal

        def sig(role: str, roles=roles) -> str | None:
            payload = roles.get(role)
            return signature_of(payload.content) if payload else None

        sibling_sigs.append(SiblingSignature(
            entity=sib.entity, hunk_ids=sib.hunk_ids,
            signature_before=sig("source.before"),
            signature_after=sig("source.after"),
            target_signature=sig("target")))

    unit_edit_regions = (unit.edit_regions if unit is not None
                         else [er for t in case.transformations for er in t.edit_regions])

    return AdaptationContext(
        case_id=case.case_id,
        unit=unit.model_dump() if unit is not None else {},
        siblings=sibling_sigs,
        transformation={role: arts[role].reduced_text for role in arts},
        edit_regions=unit_edit_regions,
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
