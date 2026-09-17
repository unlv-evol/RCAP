"""Regression tests for the findings of the 2026-09-11 spec audit."""

import json
import typing

import pytest

from rcap.context import ContextValidityFailure, build_context
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


def test_multi_entity_case_without_unit_is_refused_not_truncated(stages):
    """Audit bug 1: build_context used materialize[0], silently packaging only
    the first entity of a composite case. Section-13 per-unit processing now
    exists (the pipeline names each unit explicitly); a composite build
    WITHOUT a named unit must still refuse, never truncate."""
    case, red, mat, prog = stages
    multi = red.model_copy(deep=True)
    multi.materialize = [*multi.materialize, "functions/fn-2"]
    with pytest.raises(ContextValidityFailure) as err:
        build_context(case, multi, mat, prog)
    assert err.value.stage == "context_construction"
    assert "fn-2" in err.value.diagnostics[0]
    assert "explicit processing unit" in err.value.diagnostics[0]


def test_single_entity_case_still_builds(stages):
    case, red, mat, prog = stages
    ctx = build_context(case, red, mat, prog)
    assert set(ctx.transformation) == {"source.before", "source.after", "target"}


def test_composite_refusal_becomes_typed_harness_row(sap_dir, pr_manifest, tmp_path,
                                                     monkeypatch):
    """The harness must record the refusal as failure:context_construction."""
    from rcap import eval_harness, pipeline
    from rcap.generate import StubBackend

    def refuse(case, reduction, materialized, program, **kwargs):
        raise ContextValidityFailure(case.case_id,
                                     ["composite case: explicit processing unit required"])

    monkeypatch.setattr(pipeline, "build_context", refuse)
    rows = eval_harness.run_configs(sap_dir, pr_manifest, StubBackend("x"),
                                    out_path=tmp_path / "rows.jsonl")
    assert all(r.outcome == "failure:context_construction" for r in rows)
    written = [json.loads(line) for line in
               (tmp_path / "rows.jsonl").read_text().splitlines()]
    assert len(written) == len(rows)


def test_theta_records_params_and_model_version(stages):
    """Audit bug 3: generation params and model digest were computed by the
    backend but never recorded, so a weight change was indistinguishable from
    an evidence change in the stored artifacts (spec sections 10-12)."""
    from rcap.context import build_context
    from rcap.generate import StubBackend, generate
    from rcap.package import build_package
    from rcap.request import synthesize

    case, red, mat, prog = stages
    ctx = build_context(case, red, mat, prog)
    req = synthesize(ctx)

    class InstrumentedBackend(StubBackend):
        params: typing.ClassVar[dict] = {"temperature": 0, "seed": 7, "num_ctx": 1024}
        model_digest = "family/size/quant"

    gen = generate(req, InstrumentedBackend("```java\nvoid x() {}\n```"))
    assert gen.generation_config["params"] == {"temperature": 0, "seed": 7,
                                               "num_ctx": 1024}
    assert gen.generation_config["version"] == "family/size/quant"
    pkg = build_package(case, ctx, gen)
    assert pkg.generation_meta["params"]["seed"] == 7
    assert pkg.generation_meta["version"] == "family/size/quant"

    # A backend without the attributes keeps the minimal theta.
    plain = generate(req, StubBackend("```java\nvoid x() {}\n```"))
    assert "params" not in plain.generation_config
    assert "version" not in plain.generation_config


def test_corrupt_payload_is_typed_failure_not_crash(sap_dir, pr_manifest, tmp_path):
    """Audit bug 5: a non-UTF-8 payload raised an uncaught UnicodeDecodeError
    instead of the typed evidence failure section 7 requires."""
    import shutil

    from rcap.materialize import MaterializationFailure

    broken = tmp_path / "sap-Corrupt"
    shutil.copytree(sap_dir, broken)
    (broken / "functions" / "fn-1" / "target.java").write_bytes(b"\xff\xfe garbage \x80")

    case = load_case(broken, pr_manifest=pr_manifest)
    red = reduce_semantic(case, select(case))
    with pytest.raises(MaterializationFailure) as err:
        materialize(case, red)
    assert err.value.stage == "materialization"
    assert any("not valid UTF-8" in d for d in err.value.diagnostics)


def test_non_ascii_payload_reduces_and_recovers_byte_exact():
    """Audit bug 4: _reduce_one spliced the Python str with tree-sitter BYTE
    offsets, silently corrupting reduction and recovery whenever a non-ASCII
    character preceded a placeheld node."""
    from salp.structural import grammar_for

    from rcap.reduction_program import _reduce_one, recover

    text = (
        "void check(int n) {\n"
        "    // café naïve übermensch — non-ASCII before the removable block\n"
        "    if (n < 0) {\n"
        "        log(n);\n"
        "        throw new IllegalArgumentException();\n"
        "    }\n"
        "    use(n);\n"
        "}\n"
    )
    art = _reduce_one("functions/fn-x", "target", text, set(),
                      grammar_for("java"), 3, None)
    assert len(art.placeholders) == 1, "the if-block must be placeheld"
    assert "IllegalArgumentException" not in art.reduced_text
    assert "café naïve übermensch" in art.reduced_text, "comment must be intact"
    assert "use(n);" in art.reduced_text, "tail must not be duplicated/truncated"
    assert recover(art).encode() == text.encode()


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
