"""Payload materialization: load program payloads for retained entities only (I7).

RCAP ref section 7. Materialization is a consequence of reduction, never an
input to it: exactly the program-entity vertices the semantic-reduction
manifest names are resolved, each against the case's pinned repository states,
with provenance and a content hash carried per payload. A missing payload
whose evidence is PRESENT is an error for foundational payloads — never a
silent removal.

Note: current SALP output does not serialize the function-pool fidelity flags
(source_after_is_region, target_is_whole_file, no_function_reason); until it
does, each payload records `fidelity_flags_available=False` and downstream
stages treat the payload as a clean function body.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from pydantic import BaseModel, Field

from rcap.model import CaseModel
from rcap.reduction_semantic import SemanticReductionManifest

_ROLES = ("source.before", "source.after", "target")


class NullTransformationFailure(Exception):
    """tau carries no change: source.before == source.after (RCAP ref section 14).

    A vacuous transformation gives the backend an empty diff to "apply" —
    observed to produce hallucinated edits — so it is a typed evidence failure
    at materialization, never a silent completion. Root cause today is
    upstream: SALP anchoring the edit region on a context line and slicing an
    untouched neighbor function.
    """

    def __init__(self, case_id: str, entity: str):
        self.stage = "materialization"
        self.diagnostics = [
            f"{entity}: null transformation (source.before == source.after); no change to adapt"
        ]
        super().__init__(f"null transformation for {case_id} ({entity})")


class MaterializationFailure(Exception):
    def __init__(self, case_id: str, diagnostics: list[str]):
        self.stage = "materialization"
        self.diagnostics = diagnostics
        super().__init__(f"materialization failure for {case_id}: {'; '.join(diagnostics)}")


class MaterializedPayload(BaseModel):
    entity: str          # e.g. functions/fn-1
    role: str            # source.before | source.after | target
    ref: str             # SAP-relative path
    content: str
    sha256: str
    fidelity_flags_available: bool = False


class MaterializationRecord(BaseModel):
    case_id: str
    payloads: list[MaterializedPayload] = Field(default_factory=list)
    diagnostics: list[str] = Field(default_factory=list)

    def by_role(self, entity: str) -> dict[str, MaterializedPayload]:
        return {p.role: p for p in self.payloads if p.entity == entity}


def materialize(case: CaseModel, reduction: SemanticReductionManifest) -> MaterializationRecord:
    sap_dir = Path(case.sap_dir)
    record = MaterializationRecord(case_id=case.case_id)
    record.diagnostics.append(
        "fidelity flags not present in SAP output; payloads treated as clean function bodies"
    )
    errors: list[str] = []

    for entity in reduction.materialize:
        fn_id = entity.split("/", 1)[1]
        entry = case.functions.get(fn_id)
        if entry is None:
            errors.append(f"{entity}: retained entity has no function-pool entry")
            continue
        for role in _ROLES:
            ref = next((r for name, r in entry.refs.items() if name.startswith(role)), None)
            if ref is None:
                errors.append(f"{entity}: missing payload for {role}")
                continue
            path = sap_dir / ref
            if not path.is_file():
                errors.append(f"{entity}: payload ref {ref} does not resolve")
                continue
            content = path.read_text(encoding="utf-8")
            record.payloads.append(MaterializedPayload(
                entity=entity, role=role, ref=ref, content=content,
                sha256=hashlib.sha256(content.encode()).hexdigest(),
            ))

    if errors:
        raise MaterializationFailure(case.case_id, errors)

    for entity in reduction.materialize:
        roles = record.by_role(entity)
        before = roles.get("source.before")
        after = roles.get("source.after")
        if before and after and before.content == after.content:
            raise NullTransformationFailure(case.case_id, entity)
    return record
