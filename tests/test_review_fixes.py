"""Regression tests for the independent-review findings B1, B2, B4."""

import json
import shutil

import pytest

from rcap.intake import load_case
from rcap.materialize import NullTransformationFailure, materialize
from rcap.reduction_semantic import Disposition, ReductionMode, reduce_semantic
from rcap.selection import select


def test_b1_inadmissible_edge_does_not_flip_mode(sap_dir, pr_manifest, tmp_path):
    """I8: a same_pull_request edge alone must not switch degraded -> guided."""
    poisoned = tmp_path / "sap-SamePR"
    shutil.copytree(sap_dir, poisoned)
    hj = poisoned / "hunks" / "H-1" / "hunk.json"
    index = json.loads(hj.read_text())
    index["relationships"].append({
        "from": "H-1:ER-1", "rel": "same_pull_request",
        "to": "H-1:refactoring:rename", "state": "PRESENT",
        "evidence": "same_pull_request"})
    hj.write_text(json.dumps(index))

    case = load_case(poisoned, pr_manifest=pr_manifest)
    red = reduce_semantic(case, select(case))
    assert red.mode == {"H-1": ReductionMode.DEGRADED}
    # and retention is unchanged: nothing demoted to irrelevant
    assert Disposition.REDUCED_IRRELEVANT not in red.dispositions.values()


def test_b2_null_transformation_is_typed_failure(sap_dir, pr_manifest, tmp_path):
    """An empty before/after diff must never reach the backend as a completion."""
    null_sap = tmp_path / "sap-NullTau"
    shutil.copytree(sap_dir, null_sap)
    fn = null_sap / "functions" / "fn-1"
    (fn / "source.after.java").write_text((fn / "source.before.java").read_text())

    case = load_case(null_sap, pr_manifest=pr_manifest)
    red = reduce_semantic(case, select(case))
    with pytest.raises(NullTransformationFailure) as err:
        materialize(case, red)
    assert err.value.stage == "materialization"
    assert "null transformation" in err.value.diagnostics[0]


def test_b4_refless_category_never_silently_missing(sap_dir, pr_manifest, tmp_path):
    """A category declared in the index spine with ref null still enters the model (I4)."""
    extended = tmp_path / "sap-ExtraCat"
    shutil.copytree(sap_dir, extended)
    hj = extended / "hunks" / "H-1" / "hunk.json"
    index = json.loads(hj.read_text())
    index["evidence"]["artifact_placement"] = {
        "state": "NOT_APPLICABLE", "ref": None, "confidence": None,
        "blocking_conflict": False}
    hj.write_text(json.dumps(index))

    case = load_case(extended, pr_manifest=pr_manifest)
    assert case.evidence["H-1:artifact_placement"].state == "NOT_APPLICABLE"
    red = reduce_semantic(case, select(case))
    assert red.dispositions["H-1:artifact_placement"] == \
        Disposition.EXCLUDED_NOT_APPLICABLE


def test_b4_unavailable_category_carried_with_reason(sap_dir, pr_manifest):
    """Writer nulls the ref for UNAVAILABLE categories; the state must survive intake."""
    case = load_case(sap_dir, pr_manifest=pr_manifest)
    rec = case.evidence["H-1:verification"]
    assert rec.state == "UNAVAILABLE"
    assert rec.category == "verification"


def test_partially_vacuous_composite_processes_the_rest(
        composite_sap_dir, composite_pr_manifest, tmp_path):
    """A composite case with ONE vacuous entity must keep the others.

    Refusing the whole case for a single inert neighbour discarded adaptable
    functions: measured at 13 of 22 entities across 7 refused SAPs, only one of
    which was wholly vacuous. Section 13 processes units independently, so the
    vacuous one is a per-unit disposition, not a case-level failure.
    """
    partial = tmp_path / "sap-PartialNull"
    shutil.copytree(composite_sap_dir, partial)
    fn = partial / "functions" / "fn-1"
    (fn / "source.after.java").write_text((fn / "source.before.java").read_text())

    case = load_case(partial, pr_manifest=composite_pr_manifest)
    red = reduce_semantic(case, select(case))
    record = materialize(case, red)          # must NOT raise

    assert record.null_entities == ["functions/fn-1"]
    assert record.adaptable(red.materialize) == ["functions/fn-2"], \
        "the entity with a real change survives"
    assert any("carry no change and are excluded" in d for d in record.diagnostics), \
        "the exclusion is recorded, not silent"


def test_wholly_vacuous_composite_still_refuses(
        composite_sap_dir, composite_pr_manifest, tmp_path):
    """When NO entity has a change there is nothing to adapt: still a failure,
    and it names every vacuous entity rather than just the first."""
    allnull = tmp_path / "sap-AllNull"
    shutil.copytree(composite_sap_dir, allnull)
    for name in ("fn-1", "fn-2"):
        fn = allnull / "functions" / name
        (fn / "source.after.java").write_text((fn / "source.before.java").read_text())

    case = load_case(allnull, pr_manifest=composite_pr_manifest)
    red = reduce_semantic(case, select(case))
    with pytest.raises(NullTransformationFailure) as err:
        materialize(case, red)
    assert len(err.value.diagnostics) == 2, "every vacuous entity is reported"
