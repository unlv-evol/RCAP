"""Intake conformance tests, mapped to the invariants they verify (I1-I13)."""

import shutil

import pytest
from salp.packaging import validate_pr_dir

from rcap.intake import IntakeFailure, load_case


def test_fixture_passes_salp_validate(pr_dir):
    """The synthetic fixture is schema-identical to real SALP output."""
    report = validate_pr_dir(pr_dir)
    assert report.ok, report.errors


def test_intake_builds_case_model(sap_dir, pr_manifest):
    case = load_case(sap_dir, pr_manifest=pr_manifest)
    assert case.case_id == "PR-1/sap-Validator"
    assert case.change_type == "mapped"
    assert case.source_repo == "apache/kafka"
    assert [t.hunk_id for t in case.transformations] == ["H-1"]
    assert case.transformations[0].edit_regions == ["H-1:ER-1"]
    assert case.transformations[0].fn_id == "fn-1"


def test_i1_foundational_evidence_preserved(sap_dir, pr_manifest):
    """I1: source change, target localization, and tau are all in the index."""
    case = load_case(sap_dir, pr_manifest=pr_manifest)
    cats = {r.category for r in case.evidence.values()}
    assert {"source_change", "target_localization", "function_transformation"} <= cats
    t = case.transformations[0]
    assert t.f_s_ref and t.f_s_prime_ref and t.f_t_ref


def test_i4_states_preserved_verbatim(sap_dir, pr_manifest):
    """I4: VERIFIED_ABSENT and UNAVAILABLE stay distinguishable, never collapsed."""
    case = load_case(sap_dir, pr_manifest=pr_manifest)
    states = {r.object_id: r.state for r in case.evidence.values()}
    assert states["H-1:refactoring:rename"] == "VERIFIED_ABSENT"
    assert states["H-1:verification"] == "UNAVAILABLE"


def test_i7_no_payload_loaded(sap_dir, pr_manifest):
    """I7: intake is index-level; no program text enters the case model."""
    case = load_case(sap_dir, pr_manifest=pr_manifest)
    dump = case.model_dump_json()
    assert "NullPointerException" not in dump  # a line from the payload bodies
    assert "registerNode" not in dump


def test_i10_characterization_is_metadata_not_gate(sap_dir, pr_manifest):
    """I10: characterization rides along as metadata; intake never gates on it."""
    case = load_case(sap_dir, pr_manifest=pr_manifest)
    assert case.characterization.get("aggregate", {}).get("readiness") in {
        "LOW", "MODERATE", "HIGH"}


def test_endpoint_resolver_three_kinds(sap_dir, pr_manifest):
    case = load_case(sap_dir, pr_manifest=pr_manifest)
    assert case.resolve_endpoint("H-1:ER-1").kind == "edit_region"
    assert case.resolve_endpoint("functions/fn-1").kind == "function"
    assert case.resolve_endpoint("H-1:refactoring:rename").kind == "element"
    assert case.resolve_endpoint("void validate(Config)") is None  # bare signature: not valid


def test_aligned_to_edge_resolves(sap_dir, pr_manifest):
    case = load_case(sap_dir, pr_manifest=pr_manifest)
    assert [e.rel for e in case.relationships] == ["aligned_to"]
    assert not [d for d in case.diagnostics if "resolves to nothing" in d]


def test_intake_determinism(sap_dir, pr_manifest):
    """Acceptance: two runs over the same SAP produce the identical case model."""
    a = load_case(sap_dir, pr_manifest=pr_manifest)
    b = load_case(sap_dir, pr_manifest=pr_manifest)
    assert a == b


def test_intake_failure_on_broken_reference(sap_dir, pr_manifest, tmp_path):
    """A mandatory reference that does not resolve is a typed intake failure."""
    broken = tmp_path / "sap-Broken"
    shutil.copytree(sap_dir, broken)
    (broken / "functions" / "fn-1" / "source.before.java").unlink()
    with pytest.raises(IntakeFailure) as err:
        load_case(broken, pr_manifest=pr_manifest)
    assert err.value.stage == "intake"
    assert any("source.before" in d for d in err.value.diagnostics)


def test_intake_failure_on_missing_index(sap_dir, pr_manifest, tmp_path):
    broken = tmp_path / "sap-NoIndex"
    shutil.copytree(sap_dir, broken)
    (broken / "hunks" / "H-1" / "hunk.json").unlink()
    with pytest.raises(IntakeFailure):
        load_case(broken, pr_manifest=pr_manifest)
