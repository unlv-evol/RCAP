"""Composite-case coordination (RCAP ref section 13).

Same pipeline, additional coordination — never a second adaptation
architecture. A composite change T = {tau_1..tau_k} is partitioned into
processing units: hunks joined by an explicit inter-hunk edge
(same_function_as, or a stronger depends_on when SALP emits it) are coupled by
assertion; hunks that merely share a target function (transformation.f_t) with
no explicit link are conservatively coupled, and the record states that the
coupling was inferred from shared-function co-location rather than asserted by
SALP. Independent units are never force-merged (that would inflate context)
and coupled hunks never lose their application order (change.json hunk_order).
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from rcap.model import CaseModel

# Relationship types that assert coupling between hunks (section 13).
COUPLING_RELS = frozenset({"same_function_as", "depends_on"})

_HUNK_PREFIX = re.compile(r"^(H-\d+)\b")

INFERRED_NOTE = ("coupling inferred from shared-function co-location "
                 "(transformation.f_t), not asserted by SALP")


class Unit(BaseModel):
    """One processing unit: a target entity and the ordered hunks that edit it."""

    entity: str                       # functions/<fn_id>
    hunk_ids: list[str]               # in application order (change.json hunk_order)
    edit_regions: list[str] = Field(default_factory=list)
    coupling_basis: str | None = None  # explicit_edge | inferred_shared_target_function
    coupling_note: str | None = None


class CouplingRecord(BaseModel):
    case_id: str
    units: list[Unit] = Field(default_factory=list)   # in application order
    # Explicit SALP-asserted edges joining hunks of DIFFERENT units: recorded
    # so the backend may elect joint generation; RCAP never joins units itself.
    inter_unit_edges: list[dict[str, str]] = Field(default_factory=list)
    standalone_hunks: list[str] = Field(default_factory=list)  # no function entity
    diagnostics: list[str] = Field(default_factory=list)


def _hunk_of(endpoint: str) -> str | None:
    m = _HUNK_PREFIX.match(endpoint)
    return m.group(1) if m else None


def detect_units(case: CaseModel) -> CouplingRecord:
    order = {h: i for i, h in enumerate(case.hunk_order)}
    record = CouplingRecord(case_id=case.case_id)

    by_entity: dict[str, list] = {}
    for t in case.transformations:
        if t.fn_id is None:
            record.standalone_hunks.append(t.hunk_id)
            continue
        by_entity.setdefault(f"functions/{t.fn_id}", []).append(t)

    # Explicit coupling edges, keyed by the (unordered) hunk pair they join.
    explicit_pairs: dict[frozenset[str], list[dict[str, str]]] = {}
    for edge in case.relationships:
        if edge.rel not in COUPLING_RELS:
            continue
        hunks = {h for h in (_hunk_of(edge.src), _hunk_of(edge.dst), edge.hunk_id or None)
                 if h}
        if len(hunks) == 2:
            explicit_pairs.setdefault(frozenset(hunks), []).append(
                {"rel": edge.rel, "src": edge.src, "dst": edge.dst})

    for entity, ts in by_entity.items():
        ts.sort(key=lambda t: order.get(t.hunk_id, len(order)))
        hunk_ids = [t.hunk_id for t in ts]
        basis = note = None
        if len(hunk_ids) > 1:
            pairs_within = [p for p in explicit_pairs if p <= set(hunk_ids)]
            if pairs_within:
                basis = "explicit_edge"
                note = "coupling asserted by SALP inter-hunk relationship"
            else:
                basis = "inferred_shared_target_function"
                note = INFERRED_NOTE
        record.units.append(Unit(
            entity=entity, hunk_ids=hunk_ids,
            edit_regions=[er for t in ts for er in t.edit_regions],
            coupling_basis=basis, coupling_note=note))

    record.units.sort(key=lambda u: order.get(u.hunk_ids[0], len(order)))

    # Explicit edges joining hunks that landed in DIFFERENT units.
    unit_of = {h: u.entity for u in record.units for h in u.hunk_ids}
    for pair, edges in explicit_pairs.items():
        a, b = sorted(pair)
        if unit_of.get(a) and unit_of.get(b) and unit_of[a] != unit_of[b]:
            record.inter_unit_edges.extend(edges)
            record.diagnostics.append(
                f"explicit coupling between units {unit_of[a]} and {unit_of[b]} "
                f"({a}<->{b}): recorded for the backend; units are not joined")

    if record.standalone_hunks:
        record.diagnostics.append(
            f"standalone hunks (no function entity): {', '.join(record.standalone_hunks)}")
    return record
