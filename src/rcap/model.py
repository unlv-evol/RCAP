"""The in-memory case model intake produces (RCAP ref section 3).

Index-level only: evidence states, references, and the relationship graph.
No program payload is loaded here (invariant I7); payload_ref strings point at
files that materialization resolves later. Evidence states are preserved
verbatim from the SAP and never re-derived (I4). Characterization is attached
as opaque metadata and never consulted as a gate (I10). Nothing in this model
records any prior integration outcome (I12) or developer target solution (I13).
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field

_EDIT_REGION = re.compile(r"^(H-\d+):(ER-\d+)$")


class EndpointKind(StrEnum):
    """The three valid edge-endpoint kinds (RCAP ref section 4)."""

    ELEMENT = "element"
    FUNCTION = "function"
    EDIT_REGION = "edit_region"


class Endpoint(BaseModel):
    kind: EndpointKind
    key: str  # object_id | functions/<fn_id> | H-<n>:ER-<m>


class EvidenceRecord(BaseModel):
    """One evidence object in the index, state preserved verbatim (I4)."""

    object_id: str
    element: str
    category: str
    hunk_id: str
    state: str  # PRESENT | VERIFIED_ABSENT | UNAVAILABLE | NOT_APPLICABLE — never collapsed
    representation: float | None = None
    payload_ref: str | None = None
    attributes: dict[str, object] = Field(default_factory=dict)
    blocking_conflict: bool = False
    provenance: dict[str, object] | None = None


class RelationshipEdge(BaseModel):
    """A typed edge from a hunk index; semantic reduction traverses these."""

    src: str
    rel: str
    dst: str
    state: str = "PRESENT"
    evidence: str | None = None
    hunk_id: str = ""


class Transformation(BaseModel):
    """tau = (f_s, f'_s, f_t) as references, plus its edit regions."""

    hunk_id: str
    fn_id: str | None
    edit_regions: list[str]
    f_s_ref: str | None = None
    f_s_prime_ref: str | None = None
    f_t_ref: str | None = None


class FunctionEntry(BaseModel):
    """Function-pool entry: payload refs plus the fidelity flags (load-bearing for section 8)."""

    fn_id: str
    refs: dict[str, str] = Field(default_factory=dict)
    source_before_is_region: bool = False
    source_after_is_region: bool = False
    target_is_whole_file: bool = False
    no_function_reason: str | None = None
    target_match_kind: str | None = None
    signature: str | None = None


class CaseModel(BaseModel):
    """A validated, index-level view of one SAP: the substrate for selection and reduction."""

    case_id: str
    sap_id: str
    change_id: str
    change_type: Literal["mapped", "composite", "standalone", "partial"] | str
    schema_version: str
    sap_dir: str
    source_repo: str | None = None
    target_repo: str | None = None
    source_file: str | None = None
    target_file: str | None = None
    transformations: list[Transformation]
    hunk_order: list[str]
    functions: dict[str, FunctionEntry]
    evidence: dict[str, EvidenceRecord]
    relationships: list[RelationshipEdge]
    characterization: dict[str, object] = Field(default_factory=dict)  # metadata, never a gate (I10)
    # Section 9(7): per-category index confidence ("<hunk>:<category>" -> score),
    # carried for candidate ranking; never a retention gate.
    category_confidence: dict[str, float] = Field(default_factory=dict)
    repo_state: dict[str, object] = Field(default_factory=dict)
    diagnostics: list[str] = Field(default_factory=list)

    def characterization_scores(self) -> dict[str, object]:
        """Section-15 explanatory metadata: min-over-hunks Coverage/Fidelity
        scores (SALP's own aggregation rule) plus the aggregate Readiness
        level. Explanatory only, never a gate (I10); every field is None when
        the SAP carries no characterization."""
        out: dict[str, object] = {"coverage": None, "fidelity": None, "readiness": None}
        agg = self.characterization.get("aggregate")
        if isinstance(agg, dict):
            out["readiness"] = agg.get("readiness")
        hunks = self.characterization.get("hunks")
        if isinstance(hunks, dict):
            cov = [h.get("coverage_score") for h in hunks.values() if isinstance(h, dict)]
            fid = [h.get("fidelity_score") for h in hunks.values() if isinstance(h, dict)]
            cov = [c for c in cov if isinstance(c, (int, float))]
            fid = [f for f in fid if isinstance(f, (int, float))]
            if cov:
                out["coverage"] = min(cov)
            if fid:
                out["fidelity"] = min(fid)
        return out

    def resolve_endpoint(self, raw: str) -> Endpoint | None:
        """Resolve an edge endpoint to one of exactly three kinds, or None.

        A None return is a context-construction diagnostic for the caller to
        record, never a silently dropped edge (RCAP ref section 6).
        """
        if raw in self.evidence:
            return Endpoint(kind=EndpointKind.ELEMENT, key=raw)
        if raw.startswith("functions/"):
            fn_id = raw.split("/", 1)[1]
            if fn_id in self.functions:
                return Endpoint(kind=EndpointKind.FUNCTION, key=raw)
            return None
        if _EDIT_REGION.match(raw):
            for t in self.transformations:
                if raw in t.edit_regions:
                    return Endpoint(kind=EndpointKind.EDIT_REGION, key=raw)
            return None
        return None
