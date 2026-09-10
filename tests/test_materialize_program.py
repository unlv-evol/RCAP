"""Materialization (section 7) and Program-Context Reduction (section 8) tests."""

import json
import shutil

import pytest

from rcap.intake import load_case
from rcap.materialize import MaterializationFailure, materialize
from rcap.reduction_program import (
    recover,
    reduce_program,
)
from rcap.reduction_semantic import reduce_semantic
from rcap.selection import select


@pytest.fixture(scope="module")
def case(sap_dir, pr_manifest):
    return load_case(sap_dir, pr_manifest=pr_manifest)


@pytest.fixture(scope="module")
def pipeline(case):
    sel = select(case)
    red = reduce_semantic(case, sel)
    mat = materialize(case, red)
    return case, red, mat


# --- materialization (section 7) ---------------------------------------------

def test_materializes_exactly_the_retained_entities(pipeline):
    """I7: the materialized set equals the retained program-entity set."""
    _, red, mat = pipeline
    assert {p.entity for p in mat.payloads} == set(red.materialize)
    assert {p.role for p in mat.payloads} == {"source.before", "source.after", "target"}


def test_payloads_carry_hash_and_ref(pipeline):
    _, _, mat = pipeline
    for p in mat.payloads:
        assert p.sha256 and p.ref.startswith("functions/")
    assert any("fidelity flags" in d for d in mat.diagnostics)


def test_missing_payload_is_error_not_silent(case, sap_dir, tmp_path, pr_manifest):
    broken = tmp_path / "sap-NoTarget"
    shutil.copytree(sap_dir, broken)
    (broken / "functions" / "fn-1" / "target.java").rename(
        broken / "functions" / "fn-1" / "targe_.java")
    # bypass schema gate on purpose: materialization must catch it itself
    bcase = case.model_copy(deep=True)
    bcase.sap_dir = str(broken)
    bcase.functions["fn-1"].refs.pop("target.java")
    red = reduce_semantic(bcase, select(bcase))
    with pytest.raises(MaterializationFailure) as err:
        materialize(bcase, red)
    assert any("target" in d for d in err.value.diagnostics)


# --- program-context reduction (section 8) -----------------------------------

def test_untouched_for_loop_is_placeheld(pipeline):
    case, red, mat = pipeline
    rec = reduce_program(case, red, mat)
    art = rec.artifact("functions/fn-1", "source.before")
    assert len(art.placeholders) >= 1
    assert any("for (Node n" in p.original_text for p in art.placeholders)
    assert "RCAP_PH_" in art.reduced_text


def test_changed_region_never_placeheld(pipeline):
    """The edited if-statement stays; the change itself is always protected."""
    case, red, mat = pipeline
    rec = reduce_program(case, red, mat)
    for role, marker in (("source.before", "if (v > 2)"), ("source.after", "if (v >= 2)")):
        art = rec.artifact("functions/fn-1", role)
        assert marker in art.reduced_text
        assert all(marker not in p.original_text for p in art.placeholders)


def test_recovery_reconstructs_original_exactly(pipeline):
    case, red, mat = pipeline
    rec = reduce_program(case, red, mat)
    by_role = mat.by_role("functions/fn-1")
    for art in rec.artifacts:
        assert recover(art) == by_role[art.role].content


def test_target_reduction_mirrors_source_segments(pipeline):
    """A target segment is removable only when the source also removes it."""
    case, red, mat = pipeline
    rec = reduce_program(case, red, mat)
    art_t = rec.artifact("functions/fn-1", "target")
    # the fork's for-loop body drifted (registerNode), so it must NOT be placeheld
    assert "registerNode" in art_t.reduced_text
    # but the identical null-check IS removable on both sides
    assert any("c == null" in p.original_text for p in art_t.placeholders)


def test_size_measured_and_reduced(pipeline):
    case, red, mat = pipeline
    rec = reduce_program(case, red, mat)
    for art in rec.artifacts:
        assert art.nodes_before > 0
        assert art.nodes_after <= art.nodes_before


def test_program_reduction_deterministic(pipeline):
    case, red, mat = pipeline
    assert reduce_program(case, red, mat) == reduce_program(case, red, mat)


def test_evidence_identifier_lines_protected(enriched_pr_dir, enriched_sap_dir):
    """I9: a retained rename (setup -> setupInternal) protects the call-site line."""
    manifest = json.loads((enriched_pr_dir / "pr.json").read_text(encoding="utf-8"))
    ecase = load_case(enriched_sap_dir, pr_manifest=manifest)
    red = reduce_semantic(ecase, select(ecase))
    mat = materialize(ecase, red)
    rec = reduce_program(ecase, red, mat)
    for role in ("source.before", "source.after"):
        art = rec.artifact("functions/fn-1", role)
        assert "setup(v);" in art.reduced_text
        assert all("setup(v)" not in p.original_text for p in art.placeholders)
