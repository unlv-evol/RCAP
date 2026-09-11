"""Regression tests for the findings of the 2026-09-11 spec audit."""

import json

import pytest

from rcap.context import CompositeCaseUnsupported, build_context
from rcap.intake import load_case
from rcap.materialize import materialize
from rcap.reduction_program import reduce_program
from rcap.reduction_semantic import reduce_semantic
from rcap.selection import select


@pytest.fixture(scope="module")
def stages(sap_dir, pr_manifest):
    case = load_case(sap_dir, pr_manifest=pr_manifest)
    red = reduce_semantic(case, select(case))
    mat = materialize(case, red)
    prog = reduce_program(case, red, mat)
    return case, red, mat, prog


def test_multi_entity_case_is_refused_not_truncated(stages):
    """Audit bug 1: build_context used materialize[0], silently packaging only
    the first entity of a composite case. It must refuse with a typed failure
    until section-13 per-unit processing exists."""
    case, red, mat, prog = stages
    multi = red.model_copy(deep=True)
    multi.materialize = [*multi.materialize, "functions/fn-2"]
    with pytest.raises(CompositeCaseUnsupported) as err:
        build_context(case, multi, mat, prog)
    assert err.value.stage == "context_construction"
    assert "fn-2" in err.value.diagnostics[0]


def test_single_entity_case_still_builds(stages):
    case, red, mat, prog = stages
    ctx = build_context(case, red, mat, prog)
    assert set(ctx.transformation) == {"source.before", "source.after", "target"}


def test_composite_refusal_becomes_typed_harness_row(sap_dir, pr_manifest, tmp_path,
                                                     monkeypatch):
    """The harness must record the refusal as failure:context_construction."""
    from rcap import eval_harness, pipeline
    from rcap.generate import StubBackend

    def refuse(case, reduction, materialized, program):
        raise CompositeCaseUnsupported(case.case_id, list(reduction.materialize))

    monkeypatch.setattr(pipeline, "build_context", refuse)
    rows = eval_harness.run_configs(sap_dir, pr_manifest, StubBackend("x"),
                                    out_path=tmp_path / "rows.jsonl")
    assert all(r.outcome == "failure:context_construction" for r in rows)
    written = [json.loads(line) for line in
               (tmp_path / "rows.jsonl").read_text().splitlines()]
    assert len(written) == len(rows)


def test_failure_rows_carry_stable_id_and_dispositions(sap_dir, pr_manifest, tmp_path):
    """Audit bug 2: failure rows recorded an absolute filesystem path as case_id
    and dropped the dispositions established before the failure."""
    import shutil

    from rcap.eval_harness import run_configs
    from rcap.generate import StubBackend

    null_sap = tmp_path / "sap-NullTau"
    shutil.copytree(sap_dir, null_sap)
    fn = null_sap / "functions" / "fn-1"
    (fn / "source.after.java").write_text((fn / "source.before.java").read_text())

    rows = run_configs(null_sap, pr_manifest, StubBackend("x"))
    for row in rows:
        assert row.outcome == "failure:materialization"
        assert not row.case_id.startswith("/"), "case_id must not be a path"
        # The case model's own stable identity, taken from SAP metadata:
        assert row.case_id == "PR-1/sap-Validator"
        assert row.dispositions, "pre-failure dispositions must be kept"
        assert "retained_adaptation_material" in row.dispositions
