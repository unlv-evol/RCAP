"""Section-9 context content: dependency-diff constraints, per-category
confidence, correspondence derivation, localization alternatives, helpers."""

import pytest

from rcap.context import (
    _dependency_constraints,
    build_context,
)
from rcap.intake import load_case
from rcap.materialize import materialize
from rcap.model import EvidenceRecord
from rcap.reduction_program import reduce_program
from rcap.reduction_semantic import reduce_semantic
from rcap.request import synthesize
from rcap.selection import select


def _record(oid: str, state: str = "PRESENT", **attributes) -> EvidenceRecord:
    return EvidenceRecord(object_id=oid, element=oid.rsplit(":", 1)[-1],
                          category=oid.split(":")[1], hunk_id=oid.split(":")[0],
                          state=state, attributes=attributes)


@pytest.fixture()
def case(sap_dir, pr_manifest):
    return load_case(sap_dir, pr_manifest=pr_manifest)


def _stages(case):
    red = reduce_semantic(case, select(case))
    mat = materialize(case, red)
    prog = reduce_program(case, red, mat)
    return red, mat, prog


def test_dependency_diff_computed_locally_from_lists(case):
    """No SALP diff element: RCAP computes the diff from the two lists,
    records that it did, includes only entries touching the edit's text, and
    never serializes the raw lists."""
    src = _record("H-1:compatibility:source_dependencies",
                  dependencies=["org.example:config:1.0", "junit:junit:4.13"])
    tgt = _record("H-1:compatibility:target_dependencies",
                  dependencies=["org.example:config:2.0", "junit:junit:4.13"])
    case.evidence[src.object_id] = src
    case.evidence[tgt.object_id] = tgt

    cons = _dependency_constraints(case, {src.object_id, tgt.object_id},
                                   tau_text="int v = c.getInt(\"version\"); Config c;")
    assert len(cons) == 1
    only = cons[0]
    assert only.kind == "dependency_constraint"
    assert "version_changed" in only.detail and "org.example:config" in only.detail
    assert "computed locally by RCAP" in only.detail
    assert "junit" not in " ".join(c.detail for c in cons), \
        "entries not touching the edit region stay out"


def test_salp_diff_element_is_authoritative(case):
    diff = _record("H-1:compatibility:dependency_diff",
                   added=["org.example:config:2.0"], removed=[])
    case.evidence[diff.object_id] = diff
    cons = _dependency_constraints(case, {diff.object_id}, tau_text="Config c;")
    assert len(cons) == 1
    assert cons[0].detail.startswith("added:")
    assert "computed locally" not in cons[0].detail


def test_raw_dependency_lists_never_serialized(case):
    src = _record("H-1:compatibility:source_dependencies",
                  dependencies=["org.example:config:1.0",
                                "org.apache.directory:api-all:1.0.2"])
    tgt = _record("H-1:compatibility:target_dependencies",
                  dependencies=["org.example:config:2.0",
                                "org.apache.directory:api-all:1.0.2"])
    case.evidence[src.object_id] = src
    case.evidence[tgt.object_id] = tgt
    ctx = build_context(case, *_stages(case))
    dumped = ctx.model_dump_json()
    assert "org.example:config:1.0 -> org.example:config:2.0" in dumped
    assert "api-all" not in dumped, "the raw lists stay referenced, not copied"


def test_correspondence_insufficient_evidence_is_uncertain(case):
    red, mat, prog = _stages(case)
    ctx = build_context(case, red, mat, prog)
    assert ctx.correspondence.type == "one_to_one"
    assert ctx.correspondence.uncertain
    assert "insufficient" in ctx.correspondence.reason


def test_correspondence_typed_from_match_kind(case):
    case.evidence["H-1:localization:target_function"].attributes["match_kind"] = "exact"
    ctx = build_context(case, *_stages(case))
    assert ctx.correspondence.type == "one_to_one"
    assert not ctx.correspondence.uncertain
    assert "match_kind=exact" in ctx.correspondence.reason


def test_alternatives_keep_ambiguity_visible(case):
    alt = _record("H-1:localization:alternative_candidates",
                  candidates=[{"method": "validate(Config)"},
                              {"method": "validateAll(Config)"}])
    case.evidence[alt.object_id] = alt
    ctx = build_context(case, *_stages(case))
    assert ctx.target_localization["ambiguous"] is True
    assert len(ctx.target_localization["alternatives"]) == 2
    assert ctx.correspondence.uncertain
    assert "alternative target correspondences" in ctx.correspondence.reason
    prompt = synthesize(ctx).prompt
    assert "ambiguous" in prompt and "2 plausible target functions" in prompt


def test_confidence_carried_as_metadata(case):
    conf = _record("H-1:localization:alignment_confidence",
                   confidence=0.25, similarity={"additions": {}})
    case.evidence[conf.object_id] = conf
    ctx = build_context(case, *_stages(case))
    assert ctx.confidence["category"], "per-category index confidence recorded"
    assert ctx.confidence["localization"]["confidence"] == 0.25, \
        "a low-confidence localization is surfaced, not hidden"
    # Never exposed to the backend as instructions (I10 discipline).
    assert "0.25" not in synthesize(ctx).prompt


def test_helper_entities_are_supporting_context_not_tau(composite_sap_dir,
                                                        composite_pr_manifest):
    """A retained entity beyond tau enters as reduced fork-side context: shown,
    marked do-not-modify, and its placeholders never bind the candidate."""
    case = load_case(composite_sap_dir, pr_manifest=composite_pr_manifest)
    red, mat, prog = _stages(case)
    ctx = build_context(case, red, mat, prog, helpers=("functions/fn-2",))

    assert ctx.target_localization["function"] == "functions/fn-1"
    entities = {a.entity for a in ctx.program_context}
    assert entities == {"functions/fn-1", "functions/fn-2"}

    prompt = synthesize(ctx).prompt
    assert "Related fork context (reduced helpers)" in prompt
    assert "functions/fn-2 (do not modify)" in prompt
