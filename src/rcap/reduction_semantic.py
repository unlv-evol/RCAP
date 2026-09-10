"""Semantic Evidence Reduction: which selected evidence matters for THIS adaptation.

RCAP ref section 6. Reachability from the foundational roots along admissible
typed edges (recorded policy, not code — I2/I6), keeping the four dispositions
distinct throughout (I4). Per-hunk `reduction_mode`: an edge-poor hunk (only
`aligned_to` edges) falls back to category-level retention — degraded mode — a
defined and reportable configuration, never a broken empty context.
NOT_APPLICABLE categories are excluded from reduction entirely: not retained,
not metadata, not reduced-irrelevant, not counted in any ratio.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from rcap.config import AdmissiblePolicy
from rcap.model import CaseModel, RelationshipEdge
from rcap.selection import SelectionManifest


class Disposition(StrEnum):
    RETAINED_MATERIAL = "retained_adaptation_material"
    RETAINED_METADATA = "retained_metadata_only"
    REDUCED_IRRELEVANT = "reduced_irrelevant"
    UNRESOLVED_UNAVAILABLE = "unresolved_unavailable"
    EXCLUDED_NOT_APPLICABLE = "excluded_not_applicable"


class ReductionMode(StrEnum):
    RELATIONSHIP_GUIDED = "relationship_guided"
    DEGRADED = "degraded"
    OFF = "semantic_reduction_off"  # harness Raw/Program-only configs (section 15)


class EvidenceFailure(Exception):
    """A foundational element is UNAVAILABLE or unreachable: no valid basis (section 14)."""

    def __init__(self, case_id: str, diagnostics: list[str]):
        self.stage = "semantic_reduction"
        self.diagnostics = diagnostics
        super().__init__(f"evidence failure for {case_id}: {'; '.join(diagnostics)}")


class SemanticReductionManifest(BaseModel):
    """The retained subgraph plus a disposition for every selected object."""

    case_id: str
    policy_id: str
    mode: dict[str, ReductionMode]  # per hunk
    dispositions: dict[str, Disposition] = Field(default_factory=dict)
    retained_edges: list[RelationshipEdge] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)  # BLOCKING_CONFLICT ids (I5)
    unresolved_dependencies: list[str] = Field(default_factory=list)
    materialize: list[str] = Field(default_factory=list)  # program-entity vertices (for I7)

    def retained_ids(self) -> set[str]:
        return {oid for oid, d in self.dispositions.items()
                if d in (Disposition.RETAINED_MATERIAL, Disposition.RETAINED_METADATA)}


_FOUNDATIONAL = ("source_change", "target_localization", "function_transformation")


def reduce_semantic(
    case: CaseModel,
    selection: SelectionManifest,
    policy: AdmissiblePolicy | None = None,
    *,
    enabled: bool = True,
) -> SemanticReductionManifest:
    """One code path for all four harness configs: `enabled=False` retains every
    selected object (category-level, like degraded) and records mode OFF, so
    differences between configs are attributable to the toggle alone."""
    policy = policy or AdmissiblePolicy.load()

    _foundational_guard(case, selection)

    manifest = SemanticReductionManifest(
        case_id=case.case_id, policy_id=policy.policy_id, mode={},
    )

    hunk_edges = {t.hunk_id: [] for t in case.transformations}
    for e in case.relationships:
        hunk_edges.setdefault(e.hunk_id, []).append(e)

    for t in case.transformations:
        edges = hunk_edges.get(t.hunk_id, [])
        # Only ADMISSIBLE promoted edges count toward guided mode: an
        # inadmissible edge (same_pull_request etc., I8) must not flip the mode
        # and silently starve retention.
        promoted = ({e.rel for e in edges} & policy.admissible) - {"aligned_to"}
        if not enabled:
            mode = ReductionMode.OFF
        elif promoted:
            mode = ReductionMode.RELATIONSHIP_GUIDED
        else:
            mode = ReductionMode.DEGRADED
        manifest.mode[t.hunk_id] = mode
        hunk_evidence = {oid: s for oid, s in selection.selected.items()
                         if s.hunk_id == t.hunk_id}
        if mode in (ReductionMode.DEGRADED, ReductionMode.OFF):
            _reduce_degraded(hunk_evidence, manifest)
        else:
            _reduce_guided(case, t.hunk_id, hunk_evidence, edges, policy, manifest)
        if t.fn_id:
            fn_vertex = f"functions/{t.fn_id}"
            if fn_vertex not in manifest.materialize:
                manifest.materialize.append(fn_vertex)

    return manifest


def _foundational_guard(case: CaseModel, selection: SelectionManifest) -> None:
    problems = [
        f"foundational element {s.evidence_id} is {s.state}"
        for s in selection.selected.values()
        if s.category in _FOUNDATIONAL and s.state == "UNAVAILABLE"
    ]
    if problems:
        raise EvidenceFailure(case.case_id, problems)


def _reduce_degraded(hunk_evidence: dict, manifest: SemanticReductionManifest) -> None:
    """Category-level retention: keep every PRESENT and VERIFIED_ABSENT object, reduce nothing."""
    for oid, s in sorted(hunk_evidence.items()):
        if s.state == "PRESENT":
            manifest.dispositions[oid] = Disposition.RETAINED_MATERIAL
            if s.blocking_conflict:
                manifest.constraints.append(oid)
        elif s.state == "VERIFIED_ABSENT":
            manifest.dispositions[oid] = Disposition.RETAINED_METADATA
        elif s.state == "UNAVAILABLE":
            manifest.dispositions[oid] = Disposition.UNRESOLVED_UNAVAILABLE
            manifest.unresolved_dependencies.append(oid)
        elif s.state == "NOT_APPLICABLE":
            manifest.dispositions[oid] = Disposition.EXCLUDED_NOT_APPLICABLE


def _reduce_guided(
    case: CaseModel,
    hunk_id: str,
    hunk_evidence: dict,
    edges: list[RelationshipEdge],
    policy: AdmissiblePolicy,
    manifest: SemanticReductionManifest,
) -> None:
    """Reachability from the foundational roots along admissible edges (ref algorithm, section 6)."""
    transformation = next(t for t in case.transformations if t.hunk_id == hunk_id)

    out_edges: dict[str, list[RelationshipEdge]] = {}
    for e in edges:
        out_edges.setdefault(e.src, []).append(e)

    # Foundational vertices are always retained (I1): edit regions, tau's
    # function, and the foundational evidence elements.
    retained: set[str] = set(transformation.edit_regions)
    if transformation.fn_id:
        retained.add(f"functions/{transformation.fn_id}")
    for oid, s in hunk_evidence.items():
        if s.category in _FOUNDATIONAL and s.state != "NOT_APPLICABLE":
            retained.add(oid)

    frontier = list(transformation.edit_regions)
    visited: set[str] = set(frontier)
    while frontier:
        u = frontier.pop()
        for edge in out_edges.get(u, []):
            if edge.rel not in policy.admissible:
                continue  # membership and other inadmissible edges are never traversed (I8)
            v = edge.dst
            if case.resolve_endpoint(v) is None:
                continue  # already recorded as an intake diagnostic
            v_state = case.evidence[v].state if v in case.evidence else "PRESENT"
            if edge.state == "UNAVAILABLE" or v_state == "UNAVAILABLE":
                if v not in manifest.unresolved_dependencies:
                    manifest.unresolved_dependencies.append(v)
                continue  # visible, recorded, but does not expand the frontier
            manifest.retained_edges.append(edge)
            if v in visited:
                continue
            visited.add(v)
            retained.add(v)
            if v_state == "VERIFIED_ABSENT":
                continue  # compact assertion: retained, never expanded
            frontier.append(v)

    for oid, s in sorted(hunk_evidence.items()):
        if s.state == "NOT_APPLICABLE":
            manifest.dispositions[oid] = Disposition.EXCLUDED_NOT_APPLICABLE
        elif s.state == "UNAVAILABLE":
            manifest.dispositions[oid] = Disposition.UNRESOLVED_UNAVAILABLE
            if oid not in manifest.unresolved_dependencies:
                manifest.unresolved_dependencies.append(oid)
        elif oid in retained:
            if s.state == "VERIFIED_ABSENT":
                manifest.dispositions[oid] = Disposition.RETAINED_METADATA
            else:
                manifest.dispositions[oid] = Disposition.RETAINED_MATERIAL
                if s.blocking_conflict:
                    manifest.constraints.append(oid)  # always retained, marked (I5)
        else:
            manifest.dispositions[oid] = Disposition.REDUCED_IRRELEVANT
