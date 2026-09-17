"""Repository-state binding: recorded, checked for internal consistency, and
reconciled (RCAP ref sections 3(3), 4, 7(2); I2)."""

import json
import shutil

import pytest

from rcap.intake import IntakeFailure, load_case
from rcap.materialize import MaterializationFailure, materialize
from rcap.reduction_semantic import reduce_semantic
from rcap.selection import select

APACHE = "a" * 40
LINKEDIN = "b" * 40


def test_bindings_recorded_per_repo(sap_dir, pr_manifest):
    case = load_case(sap_dir, pr_manifest=pr_manifest)
    bindings = case.repo_state["bindings"]
    assert bindings["apache/kafka"]["commit"] == APACHE
    assert bindings["linkedin/kafka"]["commit"] == LINKEDIN
    # provenance.json and the edit-region evidence both pin apache/kafka:
    assert bindings["apache/kafka"]["pinned_by"] == 2


def _retarget_pin(sap_copy, filename: str, commit: str) -> None:
    path = sap_copy / "hunks" / "H-1" / filename
    doc = json.loads(path.read_text())
    for element in doc["elements"]:
        pin = (element.get("provenance") or {}).get("repository_pin")
        if pin:
            pin["commit"] = commit
    path.write_text(json.dumps(doc))


def test_incompatible_pins_are_an_intake_failure(sap_dir, pr_manifest, tmp_path):
    """Two different commits for the same repository: evidence bound to
    incompatible repository states must not be combined (section 3(3), I2)."""
    broken = tmp_path / "sap-SplitPin"
    shutil.copytree(sap_dir, broken)
    _retarget_pin(broken, "edit_region.json", "c" * 40)  # provenance.json still pins a*40

    with pytest.raises(IntakeFailure) as err:
        load_case(broken, pr_manifest=pr_manifest)
    assert any("incompatible repository states" in d for d in err.value.diagnostics)
    assert any("apache/kafka" in d for d in err.value.diagnostics)


def test_lagging_pr_manifest_is_reconciled_not_failed(sap_dir, pr_manifest):
    """Section 4: the evidence-level resolved pin is authoritative; a PR
    manifest whose binding lags (or reads UNAVAILABLE) is recorded, not a
    failure."""
    lagging = {**pr_manifest, "commit_binding": {"apache/kafka": "UNAVAILABLE"}}
    case = load_case(sap_dir, pr_manifest=lagging)
    notes = case.repo_state["reconciliation"]
    assert any("authoritative over PR-manifest binding 'UNAVAILABLE'" in n for n in notes)
    assert any(n in case.diagnostics for n in notes), "reconciliation is recorded"

    # No manifest binding at all also reconciles rather than failing.
    case = load_case(sap_dir, pr_manifest=pr_manifest)
    assert any("carries no commit binding" in n
               for n in case.repo_state["reconciliation"])

    # An agreeing manifest produces no reconciliation note.
    agreeing = {**pr_manifest, "commit_binding": {"apache/kafka": APACHE,
                                                  "linkedin/kafka": LINKEDIN}}
    assert load_case(sap_dir, pr_manifest=agreeing).repo_state["reconciliation"] == []


def test_pin_disagreement_is_a_materialization_error(sap_dir, pr_manifest):
    """Section 7(2): materialization re-verifies pins against the case binding
    (it is independently invocable), and a disagreement is a typed error,
    never a silent substitution."""
    case = load_case(sap_dir, pr_manifest=pr_manifest)
    case.repo_state["bindings"]["apache/kafka"]["commit"] = "f" * 40

    with pytest.raises(MaterializationFailure) as err:
        materialize(case, reduce_semantic(case, select(case)))
    assert any("disagrees with case binding" in d for d in err.value.diagnostics)


def test_payloads_carry_their_resolving_binding(sap_dir, pr_manifest):
    case = load_case(sap_dir, pr_manifest=pr_manifest)
    record = materialize(case, reduce_semantic(case, select(case)))
    for payload in record.payloads:
        pin = payload.repository_pin
        if payload.role == "target":
            assert pin["repo"] == "linkedin/kafka" and pin["commit"] == LINKEDIN
        else:
            assert pin["repo"] == "apache/kafka" and pin["commit"] == APACHE
