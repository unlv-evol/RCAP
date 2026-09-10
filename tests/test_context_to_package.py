"""Context (9), Request (10), Generation (11), and Package (12) conformance tests."""

import pytest

from rcap.generate import StubBackend, generate
from rcap.package import PackagingFailure, build_package
from rcap.pipeline import run_case
from rcap.request import synthesize

GOOD_CANDIDATE = """Sure, here is the adapted function:

```java
void validate(Config c) {
    /* RCAP_PH_target.1 */
    int v = c.getInt("version");
    if (v >= 2)
        setup(v);
    for (Node n : c.nodes()) {
        registerNode(n);
    }
}
```
"""


@pytest.fixture(scope="module")
def result(sap_dir, pr_manifest):
    return run_case(sap_dir, pr_manifest, StubBackend(GOOD_CANDIDATE))


# --- adaptation context (section 9) ------------------------------------------

def test_context_carries_reduced_triple_and_localization(result):
    ctx = result.context
    assert set(ctx.transformation) == {"source.before", "source.after", "target"}
    assert "RCAP_PH_" in ctx.transformation["source.before"]
    assert ctx.target_localization["file"] == "src/main/java/kafka/Validator.java"
    assert ctx.edit_regions == ["H-1:ER-1"]


def test_context_dispositions_distinct(result):
    """I4 end-to-end: the four dispositions arrive at the context still separate."""
    d = result.context.dispositions
    assert "H-1:verification" in d["unresolved_unavailable"]
    assert "H-1:refactoring:rename" in d["retained_metadata_only"]
    assert d.keys() >= {"retained_adaptation_material", "reduced_irrelevant",
                        "excluded_not_applicable"}


def test_context_records_reduction_mode(result):
    assert result.context.reduction_mode == {"H-1": "degraded"}


def test_context_correspondence_uncertainty_recorded(result):
    assert result.context.correspondence.type == "one_to_one"
    assert result.context.correspondence.uncertain is True


# --- request synthesis (section 10) ------------------------------------------

def test_request_carries_context_and_only_context(result):
    """No orphan facts: every code section in the prompt comes from the context."""
    prompt = result.request.prompt
    for role in ("source.before", "source.after", "target"):
        assert result.context.transformation[role] in prompt
    assert "if (v > 2)" in prompt and "if (v >= 2)" in prompt  # the change itself
    assert 'c.getInt("version")' in prompt  # non-removable statement survives


def test_request_hash_stable_and_template_recorded(result):
    again = synthesize(result.context)
    assert again.request_hash == result.request.request_hash
    assert (again.template_id, again.template_version) == ("rcap-request-v1", "1")


def test_prompt_pattern_change_alters_request_not_context(result):
    """Section 10: a template change must never reach back into the context."""
    import rcap.request as rq
    ctx_json_before = result.context.model_dump_json()
    original = rq.INSTRUCTIONS
    try:
        rq.INSTRUCTIONS = original + "\nThink step by step."
        changed = synthesize(result.context)
    finally:
        rq.INSTRUCTIONS = original
    assert changed.request_hash != result.request.request_hash
    assert result.context.model_dump_json() == ctx_json_before


# --- candidate generation (section 11) ---------------------------------------

def test_stub_completion_normalized(result):
    gen = result.generation
    assert gen.outcome == "completion"
    assert gen.candidate.startswith("void validate")  # fenced block extracted
    assert "Sure, here" not in gen.candidate
    assert gen.generation_config["backend"] == "stub"


def test_failure_modes_recorded_distinctly(result):
    req = result.request
    assert generate(req, StubBackend("")).outcome == "no_output"
    assert generate(req, StubBackend("```java\nvoid broken( {{{\n```")).outcome \
        == "unparseable"

    class Boom(StubBackend):
        def generate(self, request):
            raise RuntimeError("backend down")
    assert generate(req, Boom("")).outcome == "backend_error"


def test_compilation_unit_candidate_is_parseable(result):
    """Class-level τ makes the candidate a whole file; the parse probe must
    accept a compilation unit, not only a probe-wrapped bare method."""
    unit = ("```java\npackage p;\n\nimport java.util.List;\n\n"
            "/* header */\npublic class Whole {\n    int f() { return 1; }\n}\n```")
    assert generate(result.request, StubBackend(unit)).outcome == "completion"


def test_placeholder_violation_detected(sap_dir, pr_manifest, result):
    """A candidate that drops a required placeholder is a violation, not a completion."""
    expected = {"/* RCAP_PH_target.1 */", "/* RCAP_PH_target.99 */"}
    gen = generate(result.request, StubBackend(GOOD_CANDIDATE),
                   expected_placeholders=expected)
    assert gen.outcome == "placeholder_violation"


def test_completion_is_not_correctness(result):
    """I11: the record carries no correctness/validation field at all."""
    fields = set(result.generation.model_dump())
    assert not fields & {"correct", "valid", "tests_passed", "build"}


# --- adaptation package (section 12) -----------------------------------------

def test_package_traceability_chain(result):
    pkg = result.package
    assert pkg is not None
    assert pkg.sap_ref == "sap-Validator"
    assert pkg.context_ref and pkg.request_ref["request_hash"] == result.request.request_hash
    assert pkg.generation_meta["backend"] == "stub"
    assert pkg.provenance["source_repo"] == "apache/kafka"
    assert pkg.provenance["reduction_mode"] == {"H-1": "degraded"}


def test_package_preserves_dispositions(result):
    d = result.package.provenance["evidence_dispositions"]
    assert "H-1:verification" in d["unresolved_unavailable"]


def test_package_has_no_downstream_result(result):
    """I11/I12: no field for compilation, tests, or prior integration outcomes."""
    fields = set(result.package.model_dump())
    assert not fields & {"build", "tests", "validation", "repatch_verdict", "git_verdict"}


def test_packaging_failure_without_candidate(result):
    bad = result.generation.model_copy(update={"outcome": "no_output", "candidate": None})
    with pytest.raises(PackagingFailure):
        build_package(result.case, result.context, bad)


# --- end-to-end determinism up to generation ---------------------------------

def test_full_trace_deterministic(sap_dir, pr_manifest):
    a = run_case(sap_dir, pr_manifest, StubBackend(GOOD_CANDIDATE))
    b = run_case(sap_dir, pr_manifest, StubBackend(GOOD_CANDIDATE))
    assert a.context == b.context
    assert a.request.request_hash == b.request.request_hash
    assert a.package.context_ref == b.package.context_ref
