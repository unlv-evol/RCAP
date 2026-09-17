"""Composite, cross-file, and per-unit coordination (RCAP ref section 13)."""

import pytest

from rcap.coupling import detect_units
from rcap.eval_harness import run_configs
from rcap.generate import StubBackend
from rcap.intake import load_case
from rcap.model import RelationshipEdge
from rcap.pipeline import run_case

# Carries the one placeholder the reduced fn-1 target expects; harmless for
# units whose target has none (only MISSING placeholders violate).
CANDIDATE = "```java\nint compute(int a) { /* RCAP_PH_target.1 */ return 0; }\n```"


@pytest.fixture(scope="module")
def composite_case(composite_sap_dir, composite_pr_manifest):
    return load_case(composite_sap_dir, pr_manifest=composite_pr_manifest)


def test_shared_target_function_is_inferred_coupling(composite_case):
    """Hunks that share transformation.f_t with no explicit link are
    conservatively coupled, and the record says the coupling was inferred
    from co-location, not asserted by SALP."""
    record = detect_units(composite_case)
    units = {u.entity: u for u in record.units}
    assert set(units) == {"functions/fn-1", "functions/fn-2"}

    coupled = units["functions/fn-2"]
    assert coupled.hunk_ids == ["H-2", "H-3"], "application order preserved"
    assert coupled.coupling_basis == "inferred_shared_target_function"
    assert "not asserted by SALP" in coupled.coupling_note

    independent = units["functions/fn-1"]
    assert independent.hunk_ids == ["H-1"]
    assert independent.coupling_basis is None, "independent units are never merged"


def test_explicit_edge_is_asserted_coupling(composite_case):
    case = composite_case.model_copy(deep=True)
    case.relationships.append(RelationshipEdge(
        src="H-2:ER-1", rel="same_function_as", dst="H-3:ER-1", hunk_id="H-2"))
    record = detect_units(case)
    coupled = next(u for u in record.units if u.entity == "functions/fn-2")
    assert coupled.coupling_basis == "explicit_edge"


def test_explicit_edge_between_units_is_recorded_not_joined(composite_case):
    case = composite_case.model_copy(deep=True)
    case.relationships.append(RelationshipEdge(
        src="H-1:ER-1", rel="depends_on", dst="H-2:ER-1", hunk_id="H-1"))
    record = detect_units(case)
    assert len(record.units) == 2, "RCAP never joins units itself"
    assert record.inter_unit_edges, "the coupling is recorded for the backend"
    assert any("not joined" in d for d in record.diagnostics)


def test_per_unit_processing_produces_one_package(composite_sap_dir,
                                                  composite_pr_manifest):
    """One execution, one case, one Adaptation Package — with every unit's
    candidate inside it, in application order."""
    result = run_case(composite_sap_dir, composite_pr_manifest,
                      StubBackend(CANDIDATE))
    assert len(result.units) == 2
    assert [ur.unit.entity for ur in result.units] == ["functions/fn-1",
                                                       "functions/fn-2"]
    assert all(ur.generation.outcome == "completion" for ur in result.units)

    pkg = result.package
    assert pkg is not None
    assert [u.entity for u in pkg.units] == ["functions/fn-1", "functions/fn-2"]
    assert pkg.units[1].coupling_basis == "inferred_shared_target_function"
    assert pkg.coupling["units"], "case-level coupling record is preserved"
    # change_type recorded for partial-mapping visibility (section 13)
    assert pkg.provenance["change_type"]


def test_sibling_signatures_not_bodies(composite_sap_dir, composite_pr_manifest):
    result = run_case(composite_sap_dir, composite_pr_manifest,
                      StubBackend(CANDIDATE))
    fn1 = next(ur for ur in result.units if ur.unit.entity == "functions/fn-1")
    sibs = {s.entity: s for s in fn1.context.siblings}
    assert set(sibs) == {"functions/fn-2"}
    assert sibs["functions/fn-2"].signature_after == "int compute(int a)"

    prompt = fn1.request.prompt
    assert "signatures only" in prompt
    assert "int compute(int a)" in prompt
    # The sibling's BODY must never enter another unit's request.
    assert "base + OFFSET" not in prompt
    assert "lookupNode" not in prompt


def test_harness_row_aggregates_units(composite_sap_dir, composite_pr_manifest):
    rows = run_configs(composite_sap_dir, composite_pr_manifest,
                       StubBackend(CANDIDATE))
    for row in rows:
        assert row.units == 2
        assert set(row.unit_outcomes) == {"functions/fn-1", "functions/fn-2"}
        assert row.outcome == "completion"


def test_cross_file_edges_ingested_from_pr_manifest(sap_dir, pr_manifest):
    """An explicit SALP-recorded cross-file relationship is exposed on the
    case; mere same-PR membership stays non-traversable (I8) via the
    admissible policy, which already excludes same_pull_request."""
    manifest = {**pr_manifest, "cross_file_relationships": [
        {"from": "H-1:ER-1", "rel": "depends_on",
         "to": "H-1:compatibility:source_apis", "state": "PRESENT"},
    ]}
    case = load_case(sap_dir, pr_manifest=manifest)
    cross = [e for e in case.relationships if e.hunk_id == "PR"]
    assert len(cross) == 1 and cross[0].rel == "depends_on"

    from rcap.config import AdmissiblePolicy
    policy = AdmissiblePolicy.load()
    assert "depends_on" in policy.admissible
    assert "same_pull_request" not in policy.admissible
