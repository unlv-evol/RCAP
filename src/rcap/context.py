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

# I13: markers of a developer target solution / evaluation oracle. Nothing
# upstream may carry one; their appearance is a validity failure.
_ORACLE_MARKERS = ("ground_truth", "developer_edit", "developer_solution",
                   "oracle", "expected_target", "reference_solution")


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
    provenance_ref: str | None = None  # resolvable SAP reference (section 9 schema)


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
    # Section 9(2): file, region, containing function, match kind, and — when
    # SALP recorded multiple plausible correspondences — the alternatives and
    # their ambiguity, never silently collapsed.
    target_localization: dict[str, object]
    # Section 9(7): per-category index confidence and the localization
    # alignment confidence/similarity — metadata for candidate ranking, never
    # a retention gate; a low-confidence localization is surfaced, not hidden.
    confidence: dict[str, object] = Field(default_factory=dict)
    relationships: list[ContextRelationship] = Field(default_factory=list)
    constraints: list[ContextConstraint] = Field(default_factory=list)
    program_context: list[ReducedArtifact] = Field(default_factory=list)
    correspondence: Correspondence = Field(default_factory=Correspondence)
    dispositions: dict[str, list[str]] = Field(default_factory=dict)
    metadata_refs: dict[str, str] = Field(default_factory=dict)  # sap/characterization refs
    reduction_mode: dict[str, str] = Field(default_factory=dict)


def _tail(object_id: str) -> str:
    return object_id.rsplit(":", 1)[-1]


def _retained_by_tail(case: CaseModel, retained: set[str], *tails: str):
    for oid in sorted(retained):
        rec = case.evidence.get(oid)
        if rec is not None and _tail(oid) in tails:
            yield rec


def _localization(case: CaseModel, retained: set[str], entity: str) -> dict[str, object]:
    """Section 9(2): target file, region, containing function, match kind, and
    alternatives with their ambiguity preserved."""
    loc: dict[str, object] = {"file": case.target_file, "function": entity,
                              "region": None, "match_kind": None,
                              "alternatives": [], "ambiguous": False}
    for rec in _retained_by_tail(case, retained,
                                 "localized_target_function", "target_function"):
        a = rec.attributes
        loc["match_kind"] = loc["match_kind"] or a.get("match_kind")
        loc["region"] = loc["region"] or a.get("region") or a.get("method")
    for rec in _retained_by_tail(case, retained, "alternative_candidates"):
        if rec.state == "PRESENT":
            candidates = rec.attributes.get("candidates") or []
            loc["alternatives"] = list(candidates)
            loc["ambiguous"] = bool(candidates)
    return loc


def _localization_confidence(case: CaseModel, retained: set[str]) -> dict[str, object]:
    for rec in _retained_by_tail(case, retained, "alignment_confidence"):
        if rec.state == "PRESENT":
            return {k: v for k, v in rec.attributes.items()
                    if k in ("confidence", "similarity")}
    return {}


def _derive_correspondence(case: CaseModel, retained: set[str]) -> Correspondence:
    """Section 9(9): type the correspondence from SAP alignment evidence.
    Insufficient evidence yields one_to_one with an uncertain flag — never a
    guess, and RCAP never invents correspondences for a non-one-to-one case."""
    functions = list(_retained_by_tail(case, retained,
                                       "localized_target_function", "target_function"))
    alternatives = [rec for rec in _retained_by_tail(case, retained,
                                                     "alternative_candidates")
                    if rec.state == "PRESENT" and rec.attributes.get("candidates")]
    if alternatives:
        n = sum(len(rec.attributes["candidates"]) for rec in alternatives)
        return Correspondence(uncertain=True, reason=(
            f"{n} alternative target correspondences recorded by SALP; "
            "not typed beyond one_to_one without stronger alignment evidence"))
    kinds = {rec.attributes.get("match_kind") for rec in functions} - {None}
    if len(functions) == 1 and kinds <= {"exact", "signature"} and kinds:
        return Correspondence(type="one_to_one", uncertain=False,
                              reason=f"single localized target function, "
                                     f"match_kind={next(iter(kinds))}")
    return Correspondence(uncertain=True, reason=(
        "SAP alignment evidence insufficient to type the correspondence; "
        "defaulting to one_to_one rather than guessing"))


def _artifact_of(dep: str) -> str:
    parts = dep.split(":")
    return parts[1] if len(parts) >= 2 else parts[0]


def _touches(dep: str, tau_text: str) -> bool:
    """Does a dependency's artifact name appear in the edit's program text?"""
    return _artifact_of(dep).lower() in tau_text.lower()


def _dependency_constraints(case: CaseModel, retained: set[str],
                            tau_text: str) -> list[ContextConstraint]:
    """Section 9(6): dependency constraints from the DIFF, never the raw
    lists. A SALP-emitted diff element is consumed directly; otherwise the
    diff is computed locally from the two lists, and the constraint records
    that RCAP did so. Only entries touching APIs the edit region uses enter
    the context; the raw lists stay referenced by evidence id only."""
    out: list[ContextConstraint] = []

    for rec in _retained_by_tail(case, retained, "dependency_diff", "api_diff"):
        if rec.state != "PRESENT":
            continue
        a = rec.attributes
        for kind in ("added", "removed", "version_changed"):
            for dep in a.get(kind) or []:
                if _touches(str(dep), tau_text):
                    out.append(ContextConstraint(
                        id=f"constraint:{rec.object_id}:{dep}",
                        kind="dependency_constraint",
                        detail=f"{kind}: {dep}", evidence_ref=rec.object_id))
        return out  # the SALP diff is authoritative when present

    lists: dict[str, tuple[str, list[str]]] = {}
    for side, tail in (("source", "source_dependencies"),
                       ("target", "target_dependencies")):
        for rec in _retained_by_tail(case, retained, tail):
            if rec.state == "PRESENT":
                lists[side] = (rec.object_id,
                               [str(d) for d in rec.attributes.get("dependencies") or []])
    if "source" not in lists or "target" not in lists:
        return out

    def by_key(deps: list[str]) -> dict[str, str]:
        return {dep.rsplit(":", 1)[0]: dep for dep in deps}

    src_ref, src = lists["source"]
    tgt_ref, tgt = lists["target"]
    src_map, tgt_map = by_key(src), by_key(tgt)
    note = "diff computed locally by RCAP from the SAP dependency lists"
    entries = (
        [("removed_in_target", src_map[k], src_ref) for k in src_map.keys() - tgt_map.keys()]
        + [("added_in_target", tgt_map[k], tgt_ref) for k in tgt_map.keys() - src_map.keys()]
        + [("version_changed", f"{src_map[k]} -> {tgt_map[k]}", tgt_ref)
           for k in src_map.keys() & tgt_map.keys() if src_map[k] != tgt_map[k]]
    )
    for kind, dep, ref in entries:
        if _touches(dep, tau_text):
            out.append(ContextConstraint(
                id=f"constraint:{ref}:{dep}", kind="dependency_constraint",
                detail=f"{kind}: {dep} ({note})", evidence_ref=ref))
    return out


def build_context(
    case: CaseModel,
    reduction: SemanticReductionManifest,
    materialized: MaterializationRecord,
    program: ProgramReductionRecord,
    unit: Unit | None = None,
    siblings: tuple[Unit, ...] = (),
    helpers: tuple[str, ...] = (),
) -> AdaptationContext:
    """Build the context for one processing unit (section 13: per-unit
    processing). A composite case must name its unit explicitly — building
    from only the first entity would silently misrepresent the case as fully
    covered, so that remains a typed failure, never a truncation."""
    if unit is None:
        tau_entities = [e for e in reduction.materialize if e not in helpers]
        if len(tau_entities) > 1:
            raise ContextValidityFailure(case.case_id, [
                (f"composite case: {len(tau_entities)} materialized entities "
                 f"({', '.join(tau_entities)}); an explicit processing unit "
                 f"is required (section 13) — refusing rather than truncating "
                 f"to the first entity")])
        entity = tau_entities[0] if tau_entities else None
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
            id=f"{e.src}-{e.rel}->{e.dst}", type=e.rel, src=e.src, dst=e.dst,
            provenance_ref=("pr.json#cross_file_relationships" if e.hunk_id == "PR"
                            else f"hunks/{e.hunk_id}/hunk.json#relationships"))
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

    # Section 9(6): dependency constraints from the diff, filtered to APIs the
    # edit's program text actually uses.
    tau_text = "\n".join(arts[r].reduced_text for r in arts)
    constraints.extend(_dependency_constraints(case, retained, tau_text))

    # Validity: every retained relationship endpoint and constraint traceable to the SAP.
    problems = [f"constraint {c.id} evidence {c.evidence_ref} is not retained"
                for c in constraints if c.evidence_ref not in retained]
    # I13: no evaluation-only or historical target solution may be present —
    # nothing upstream should carry one, so anything that looks like a
    # developer solution or oracle is a validity failure, never content.
    for oid in sorted(retained):
        rec = case.evidence.get(oid)
        if rec is None:
            continue
        haystack = " ".join([oid.lower(), (rec.element or "").lower(),
                             *(str(k).lower() for k in rec.attributes)])
        marker = next((m for m in _ORACLE_MARKERS if m in haystack), None)
        if marker:
            problems.append(f"I13 oracle leakage: retained evidence {oid} carries "
                            f"a developer-solution/oracle marker ({marker})")
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

    # Section 9(5): retained helper entities beyond tau — their reduced
    # target-side representation (what exists in the fork), recovery maps kept.
    helper_arts = [a for h in helpers for a in program.artifacts
                   if a.entity == h and a.role == "target"]

    return AdaptationContext(
        case_id=case.case_id,
        unit=unit.model_dump() if unit is not None else {},
        siblings=sibling_sigs,
        transformation={role: arts[role].reduced_text for role in arts},
        edit_regions=unit_edit_regions,
        target_localization=_localization(case, retained, entity),
        confidence={"category": dict(case.category_confidence),
                    "localization": _localization_confidence(case, retained)},
        relationships=relationships,
        constraints=constraints,
        program_context=[arts[r] for r in ("source.before", "source.after", "target")]
                        + helper_arts,
        correspondence=_derive_correspondence(case, retained),
        dispositions=dispositions,
        metadata_refs={"sap": case.sap_id,
                       "characterization": f"{case.sap_id}/characterization.json"},
        reduction_mode={h: m.value for h, m in reduction.mode.items()},
    )
