"""Four-configuration harness conformance tests (section 15)."""

import json

import pytest

from rcap.config import EvaluationMode
from rcap.eval_harness import run_configs
from rcap.generate import StubBackend

CANDIDATE = "```java\nvoid validate(Config c) { int v = 0; }\n```"


@pytest.fixture(scope="module")
def rows(enriched_pr_dir, enriched_sap_dir):
    manifest = json.loads((enriched_pr_dir / "pr.json").read_text())
    return {r.mode: r for r in run_configs(
        enriched_sap_dir, manifest, StubBackend(CANDIDATE))}


def test_four_rows_one_shared_schema(rows):
    assert set(rows) == {m.value for m in EvaluationMode}
    assert len({r.case_id for r in rows.values()}) == 1


def test_raw_retains_everything_reduces_nothing(rows):
    raw = rows["raw_context"]
    assert raw.dispositions.get("reduced_irrelevant", 0) == 0
    assert raw.placeholders == 0
    assert raw.nodes_after == raw.nodes_before
    assert set(raw.reduction_mode.values()) == {"semantic_reduction_off"}


def test_program_only_cuts_ast_without_semantic_reduction(rows):
    po = rows["program_reduction_only"]
    assert po.dispositions.get("reduced_irrelevant", 0) == 0
    assert po.placeholders > 0
    assert po.nodes_after < po.nodes_before


def test_semantic_only_reduces_evidence_keeps_full_payloads(rows):
    so = rows["semantic_reduction_only"]
    assert so.dispositions.get("reduced_irrelevant", 0) > 0
    assert so.placeholders == 0
    assert so.nodes_after == so.nodes_before
    assert set(so.reduction_mode.values()) == {"relationship_guided"}


def test_full_reduces_both(rows):
    full = rows["full_rcap"]
    assert full.dispositions.get("reduced_irrelevant", 0) > 0
    assert full.placeholders > 0
    assert full.nodes_after < full.nodes_before


def test_dispositions_internally_consistent(rows):
    """selected = retained + metadata + irrelevant + unavailable + excluded, per config."""
    totals = {m: sum(r.dispositions.values()) for m, r in rows.items()}
    assert len(set(totals.values())) == 1  # same selected population in every config


def test_no_correctness_field_in_rows(rows):
    """I11: the harness measures efficiency and completion, never correctness."""
    fields = set(next(iter(rows.values())).model_dump())
    assert not fields & {"correct", "accuracy", "tests_passed", "build", "valid"}


def test_typed_failure_becomes_row_not_crash(sap_dir, pr_manifest, tmp_path):
    import shutil
    null_sap = tmp_path / "sap-NullTau"
    shutil.copytree(sap_dir, null_sap)
    fn = null_sap / "functions" / "fn-1"
    (fn / "source.after.java").write_text((fn / "source.before.java").read_text())
    rows = run_configs(null_sap, pr_manifest, StubBackend(CANDIDATE))
    assert all(r.outcome == "failure:materialization" for r in rows)
