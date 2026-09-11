"""Evidence selection: inclusive eligibility over the SAP index (RCAP ref section 5).

Operates over the index only; no payload is materialized to decide eligibility
(I7). Foundational evidence is admitted unconditionally (I1); everything
associated with the affected entities is admitted for consideration —
over-admission here is corrected downstream by Semantic Evidence Reduction,
whereas under-admission cannot be. States and provenance are carried forward
unchanged (I4). Pure function of the index: reproducible by construction.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from rcap.model import CaseModel

FOUNDATIONAL_CATEGORIES = frozenset(
    {"source_change", "target_localization", "function_transformation"}
)


class SelectedEvidence(BaseModel):
    evidence_id: str
    category: str
    state: str  # preserved verbatim
    hunk_id: str
    admitted_via: list[str]  # "foundational" | edge descriptions | "associated_with_case"
    blocking_conflict: bool = False


class SelectionManifest(BaseModel):
    """Index-level manifest of every selected evidence object (never payload)."""

    case_id: str
    selected: dict[str, SelectedEvidence] = Field(default_factory=dict)


def select(case: CaseModel) -> SelectionManifest:
    """Admit foundational evidence unconditionally, and all associated support evidence."""
    edges_touching: dict[str, list[str]] = {}
    for e in case.relationships:
        desc = f"{e.src} -{e.rel}-> {e.dst}"
        edges_touching.setdefault(e.src, []).append(desc)
        edges_touching.setdefault(e.dst, []).append(desc)

    manifest = SelectionManifest(case_id=case.case_id)
    for oid, rec in sorted(case.evidence.items()):
        if rec.category in FOUNDATIONAL_CATEGORIES:
            via = ["foundational"]
        else:
            via = edges_touching.get(oid, ["associated_with_case"])
        manifest.selected[oid] = SelectedEvidence(
            evidence_id=oid, category=rec.category, state=rec.state,
            hunk_id=rec.hunk_id, admitted_via=via,
            blocking_conflict=rec.blocking_conflict,
        )
    return manifest
