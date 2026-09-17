"""Section-17 negative tests: oracle leakage, evidence invention, foundational
gaps, and downstream results in packages are all rejected."""

import shutil

import pytest

from rcap.context import ContextValidityFailure, build_context
from rcap.generate import StubBackend, generate
from rcap.intake import load_case
from rcap.materialize import materialize
from rcap.model import EvidenceRecord
from rcap.package import PackagingFailure, build_package, validate_package
from rcap.reduction_program import reduce_program
from rcap.reduction_semantic import reduce_semantic
from rcap.request import synthesize, untraceable_lines
from rcap.selection import select


def _stages(case):
    red = reduce_semantic(case, select(case))
    mat = materialize(case, red)
    prog = reduce_program(case, red, mat)
    return red, mat, prog


def test_oracle_leakage_is_rejected(sap_dir, pr_manifest):
    """I13: retained evidence that looks like a developer target solution or
    evaluation oracle fails context validity — it can never become content."""
    case = load_case(sap_dir, pr_manifest=pr_manifest)
    case.evidence["H-1:compatibility:ground_truth_patch"] = EvidenceRecord(
        object_id="H-1:compatibility:ground_truth_patch",
        element="developer_solution", category="compatibility", hunk_id="H-1",
        state="PRESENT", attributes={"patch": "the developer's actual edit"})
    with pytest.raises(ContextValidityFailure) as err:
        build_context(case, *_stages(case))
    assert any("I13 oracle leakage" in d for d in err.value.diagnostics)


def test_request_fact_with_no_context_origin_fails_mapping(sap_dir, pr_manifest):
    """Evidence invention is impossible: every request fact maps to the
    context; a smuggled fact is untraceable."""
    case = load_case(sap_dir, pr_manifest=pr_manifest)
    ctx = build_context(case, *_stages(case))
    request = synthesize(ctx)
    assert untraceable_lines(request.prompt, ctx) == []

    tampered = request.prompt + "\nThe developer also renamed register to enroll.\n"
    leaked = untraceable_lines(tampered, ctx)
    assert leaked == ["The developer also renamed register to enroll."]


def test_foundational_gap_is_evidence_failure_not_empty_context(sap_dir,
                                                                pr_manifest,
                                                                tmp_path):
    """A missing foundational payload is a typed evidence failure with stage
    attribution — never an empty context that generation would hallucinate
    around."""
    broken = tmp_path / "sap-NoTarget"
    shutil.copytree(sap_dir, broken)
    (broken / "functions" / "fn-1" / "target.java").unlink()

    with pytest.raises(Exception) as err:
        case = load_case(broken, pr_manifest=pr_manifest)
        red = reduce_semantic(case, select(case))
        materialize(case, red)
    exc = err.value
    assert getattr(exc, "stage", None) in ("intake", "materialization")
    assert exc.diagnostics, "diagnostics must name what is missing"


def test_downstream_result_in_package_fails_validation(sap_dir, pr_manifest):
    """I11/I12: a downstream validation result or prior integration outcome
    anywhere in the package structure fails packaging validation."""
    case = load_case(sap_dir, pr_manifest=pr_manifest)
    red, mat, prog = _stages(case)
    ctx = build_context(case, red, mat, prog)
    gen = generate(synthesize(ctx), StubBackend("```java\nvoid x() {}\n```"))
    package = build_package(case, ctx, gen)  # clean package validates

    smuggled = package.model_copy(deep=True)
    smuggled.generation_meta["test_result"] = "passed"
    with pytest.raises(PackagingFailure) as err:
        validate_package(smuggled)
    assert "I11/I12" in err.value.diagnostics[0]
    assert "test_result" in err.value.diagnostics[0]

    # Nested keys are caught too (e.g. a backend stuffing theta.params).
    nested = package.model_copy(deep=True)
    nested.provenance["upstream"] = {"prior_integration": {"merged": True}}
    with pytest.raises(PackagingFailure):
        validate_package(nested)


def test_build_package_rejects_contaminated_generation(sap_dir, pr_manifest):
    case = load_case(sap_dir, pr_manifest=pr_manifest)
    red, mat, prog = _stages(case)
    ctx = build_context(case, red, mat, prog)
    gen = generate(synthesize(ctx), StubBackend("```java\nvoid x() {}\n```"))
    gen.generation_config["integration_outcome"] = "MERGED"
    with pytest.raises(PackagingFailure):
        build_package(case, ctx, gen)


def test_prior_integration_outcome_never_consulted_at_intake(sap_dir, pr_manifest):
    """I12: a prior integration outcome in the PR manifest plays no part —
    the case model neither gates on it nor carries it."""
    manifest = {**pr_manifest, "prior_integration_outcome": "FAILED",
                "repatch_verdict": {"exact": 0}}
    case = load_case(sap_dir, pr_manifest=manifest)
    dumped = case.model_dump_json()
    assert "prior_integration_outcome" not in dumped
    assert "repatch_verdict" not in dumped
