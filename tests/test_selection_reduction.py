"""Selection (section 5) and Semantic Evidence Reduction (section 6) conformance tests."""

import json

import pytest
from salp.packaging import validate_pr_dir

from rcap.intake import load_case
from rcap.reduction_semantic import (
    Disposition,
    EvidenceFailure,
    ReductionMode,
    reduce_semantic,
)
from rcap.selection import select


@pytest.fixture(scope="module")
def case(sap_dir, pr_manifest):
    return load_case(sap_dir, pr_manifest=pr_manifest)


@pytest.fixture(scope="module")
def enriched_case(enriched_pr_dir, enriched_sap_dir):
    manifest = json.loads((enriched_pr_dir / "pr.json").read_text(encoding="utf-8"))
    return load_case(enriched_sap_dir, pr_manifest=manifest)


# --- selection (section 5) ---------------------------------------------------

def test_selection_admits_foundational_unconditionally(case):
    sel = select(case)
    foundational = [s for s in sel.selected.values() if s.admitted_via == ["foundational"]]
    assert {s.category for s in foundational} == {
        "source_change", "target_localization", "function_transformation"}


def test_selection_preserves_states_verbatim(case):
    sel = select(case)
    assert sel.selected["H-1:refactoring:rename"].state == "VERIFIED_ABSENT"
    assert sel.selected["H-1:verification"].state == "UNAVAILABLE"


def test_selection_is_deterministic(case):
    assert select(case) == select(case)


def test_selection_carries_no_payload(case):
    dump = select(case).model_dump_json()
    assert "NullPointerException" not in dump


# --- degraded mode (edge-poor hunk: only aligned_to) -------------------------

def test_degraded_mode_detected_and_recorded(case):
    red = reduce_semantic(case, select(case))
    assert red.mode == {"H-1": ReductionMode.DEGRADED}


def test_degraded_retains_categories_reduces_nothing(case):
    red = reduce_semantic(case, select(case))
    d = red.dispositions
    assert d["H-1:source_change:edit_region"] == Disposition.RETAINED_MATERIAL
    assert d["H-1:refactoring:rename"] == Disposition.RETAINED_METADATA
    assert d["H-1:verification"] == Disposition.UNRESOLVED_UNAVAILABLE
    assert Disposition.REDUCED_IRRELEVANT not in d.values()
    assert "H-1:verification" in red.unresolved_dependencies


def test_degraded_still_names_what_to_materialize(case):
    red = reduce_semantic(case, select(case))
    assert red.materialize == ["functions/fn-1"]


# --- relationship-guided mode (promoted edges) -------------------------------

def test_guided_mode_detected(enriched_case):
    red = reduce_semantic(enriched_case, select(enriched_case))
    assert red.mode == {"H-1": ReductionMode.RELATIONSHIP_GUIDED}


def test_guided_retains_reachable_reduces_unreachable(enriched_case):
    """A rename on the affected entity is retained; an unrelated one is reduced (I6)."""
    red = reduce_semantic(enriched_case, select(enriched_case))
    d = red.dispositions
    assert d["H-1:refactoring:rename_setup"] == Disposition.RETAINED_MATERIAL
    assert d["H-1:compatibility:api_diff"] == Disposition.RETAINED_MATERIAL
    assert d["H-1:refactoring:unrelated_rename"] == Disposition.REDUCED_IRRELEVANT
    assert d["H-1:surrounding:enclosing_context"] == Disposition.REDUCED_IRRELEVANT


def test_guided_same_pr_edge_not_traversed(enriched_case):
    """I8: shared-PR membership alone is never a semantic relationship."""
    red = reduce_semantic(enriched_case, select(enriched_case))
    assert not any(e.rel == "same_pull_request" for e in red.retained_edges)


def test_guided_verified_absent_retained_as_assertion(enriched_case):
    red = reduce_semantic(enriched_case, select(enriched_case))
    assert red.dispositions["H-1:compatibility:objects_absent"] == \
        Disposition.RETAINED_METADATA


def test_guided_unavailable_recorded_not_expanded(enriched_case):
    """An UNAVAILABLE neighbor is an unresolved dependency, never RCAP-removed."""
    red = reduce_semantic(enriched_case, select(enriched_case))
    assert red.dispositions["H-1:verification"] == \
        Disposition.UNRESOLVED_UNAVAILABLE
    assert "H-1:verification" in red.unresolved_dependencies


def test_guided_blocking_conflict_is_constraint(enriched_case):
    """I5: a reachable BLOCKING_CONFLICT is always retained and marked."""
    red = reduce_semantic(enriched_case, select(enriched_case))
    assert red.dispositions["H-1:compatibility:version_conflict"] == \
        Disposition.RETAINED_MATERIAL
    assert red.constraints == ["H-1:compatibility:version_conflict"]


def test_guided_deterministic(enriched_case):
    a = reduce_semantic(enriched_case, select(enriched_case))
    b = reduce_semantic(enriched_case, select(enriched_case))
    assert a == b


def test_enriched_fixture_still_passes_salp_validate(enriched_pr_dir):
    report = validate_pr_dir(enriched_pr_dir)
    assert report.ok, report.errors


# --- foundational guard (section 14) -----------------------------------------

def test_evidence_failure_on_unavailable_foundational(case):
    sel = select(case)
    mutated = sel.model_copy(deep=True)
    mutated.selected["H-1:transformation:triple"].state = "UNAVAILABLE"
    with pytest.raises(EvidenceFailure) as err:
        reduce_semantic(case, mutated)
    assert err.value.stage == "semantic_reduction"


# --- NOT_APPLICABLE exclusion (I4 / section 6) -------------------------------

def test_not_applicable_excluded_from_reduction(case):
    sel = select(case)
    mutated = sel.model_copy(deep=True)
    mutated.selected["H-1:surrounding:enclosing_context"].state = "NOT_APPLICABLE"
    red = reduce_semantic(case, mutated)
    assert red.dispositions["H-1:surrounding:enclosing_context"] == \
        Disposition.EXCLUDED_NOT_APPLICABLE
