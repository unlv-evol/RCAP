"""Placeholder handling: hard rejection of leaks, and the single opt-in retry."""

import hashlib

from rcap.fidelity import PLACEHOLDER_LEAK, judge
from rcap.generate import (
    StubBackend,
    generate,
    generate_with_placeholder_retry,
    has_placeholder_leak,
    placeholder_report,
)
from rcap.pipeline import run_case
from rcap.request import PLACEHOLDER_REPAIR_TEMPLATE_ID, AdaptationRequest, synthesize_placeholder_repair

PH1, PH2 = "/* RCAP_PH_target.1 */", "/* RCAP_PH_target.2 */"


def req(prompt="## Target\n```java\nvoid f() { /* RCAP_PH_target.2 */ a(); /* RCAP_PH_target.1 */ }\n```\n"):
    return AdaptationRequest(case_id="c", prompt=prompt, request_hash=hashlib.sha256(prompt.encode()).hexdigest())


class Scripted:
    name, model = "scripted", "s-0"

    def __init__(self, answers):
        self.answers, self.prompts = list(answers), []

    def generate(self, request):
        self.prompts.append(request.prompt)
        return self.answers.pop(0)


def code(body):
    return f"```java\nvoid f() {{ {body} }}\n```"


def test_report_names_missing_duplicated_and_unknown():
    r = placeholder_report(f"void f() {{ {PH1} {PH1} /* RCAP_PH_9 */ // RCAP_PH_target.2\n }}", {PH1, PH2})
    assert r == {"missing": [PH2], "duplicated": [PH1],
                 "unknown": ["/* RCAP_PH_9 */", "1 malformed mention(s) of RCAP_PH_"]}
    assert placeholder_report(f"{PH2} {PH1}", {PH1, PH2}) == {"missing": [], "duplicated": [], "unknown": []}


def test_missing_placeholder_diagnostic_is_worded_as_before():
    g = generate(req(), StubBackend(code(PH1)), expected_placeholders={PH1, PH2})
    assert g.outcome == "placeholder_violation"
    assert g.diagnostics == [f"missing placeholders: ['{PH2}']"]


def test_unknown_or_duplicated_placeholder_is_a_hard_rejection():
    # previously a completion: the invented comment went into the fork as code
    g = generate(req(), StubBackend(code("/* RCAP_PH_1 */ a();")), expected_placeholders=set())
    assert g.outcome == "placeholder_violation" and g.candidate is None
    assert g.diagnostics == ["unknown placeholders: ['/* RCAP_PH_1 */']"]
    g = generate(req(), StubBackend(code(f"{PH1} a(); {PH1}")), expected_placeholders={PH1})
    assert g.outcome == "placeholder_violation" and "duplicated" in g.diagnostics[0]
    # no expectation given (callers that do not check): unchanged behaviour
    assert generate(req(), StubBackend(code("/* RCAP_PH_1 */"))).outcome == "completion"
    assert has_placeholder_leak("x /* RCAP_PH_1 */") and not has_placeholder_leak("x") and not has_placeholder_leak(None)


def test_retry_request_states_exactly_what_was_wrong_and_the_rule():
    r = synthesize_placeholder_repair(req(), [PH1, PH2], {"missing": [PH2], "duplicated": [PH1], "unknown": ["/* RCAP_PH_9 */"]})
    assert r.template_id == PLACEHOLDER_REPAIR_TEMPLATE_ID and r.prompt.startswith(req().prompt)
    tail = r.prompt[len(req().prompt):]
    assert f"missing (must appear, but was absent): {PH2}" in tail
    assert f"appeared more than once (must appear exactly once): {PH1}" in tail
    assert "not a placeholder of the target function (you must not write it): /* RCAP_PH_9 */" in tail
    # listed in the order they stand in the target, not sorted
    assert tail.index(f"  {PH2}") < tail.index(f"  {PH1}")
    assert "exactly once" in tail and "no other /* RCAP_PH_... */ comment" in tail
    assert r.request_hash != req().request_hash
    assert synthesize_placeholder_repair(req(), [PH1, PH2], {"missing": [PH2]}).prompt == \
        synthesize_placeholder_repair(req(), [PH2, PH1], {"missing": [PH2]}).prompt     # deterministic
    none = synthesize_placeholder_repair(req("x\n"), [], {"unknown": ["/* RCAP_PH_1 */"]})
    assert "contains no placeholder comment" in none.prompt


def test_one_retry_fixes_a_violation():
    b = Scripted([code(PH1), code(f"{PH2} a(); {PH1}")])
    g, info = generate_with_placeholder_retry(req(), b, expected_placeholders={PH1, PH2})
    assert g.outcome == "completion" and PH2 in g.candidate
    assert info["fixed"] and info["report"]["missing"] == [PH2] and len(b.prompts) == 2
    assert "Placeholder check FAILED" in b.prompts[1] and "Placeholder check FAILED" not in b.prompts[0]
    assert g.request_hash == info["request_hash"] != req().request_hash


def test_only_one_retry_and_the_original_refusal_is_kept():
    b = Scripted([code(PH1), code(PH2), code(f"{PH1} {PH2}")])
    g, info = generate_with_placeholder_retry(req(), b, expected_placeholders={PH1, PH2})
    assert len(b.prompts) == 2 and len(b.answers) == 1          # never a third request
    assert g.outcome == "placeholder_violation" and g.diagnostics == [f"missing placeholders: ['{PH2}']"]
    assert g.request_hash == req().request_hash                 # the first record, unchanged
    assert info["fixed"] is False and info["retry_outcome"] == "placeholder_violation"
    assert info["retry_diagnostics"] == [f"missing placeholders: ['{PH1}']"]


def test_no_retry_for_a_completion_or_another_failure():
    b = Scripted([code(f"{PH1} {PH2}")])
    g, info = generate_with_placeholder_retry(req(), b, expected_placeholders={PH1, PH2})
    assert g.outcome == "completion" and info is None and len(b.prompts) == 1
    b = Scripted(["```java\nvoid f() { ((( \n```"])
    g, info = generate_with_placeholder_retry(req(), b, expected_placeholders={PH1, PH2})
    assert g.outcome == "unparseable" and info is None and len(b.prompts) == 1


def test_pipeline_flag_is_off_by_default_and_requests_are_unchanged(sap_dir, pr_manifest):
    bad = "```java\nvoid validate(Config c) { }\n```"
    good = "```java\nvoid validate(Config c) { /* RCAP_PH_target.1 */ }\n```"
    off = Scripted([bad, good])
    r_off = run_case(sap_dir, pr_manifest, off)
    assert len(off.prompts) == 1 and r_off.units[0].placeholder_retry is None
    on = Scripted([bad, good])
    r_on = run_case(sap_dir, pr_manifest, on, placeholder_retry=True)
    assert on.prompts[0] == off.prompts[0]                      # attempt 1 is byte-identical
    assert r_on.units[0].request.request_hash == r_off.units[0].request.request_hash
    if r_off.generation.outcome == "placeholder_violation":     # the fixture's target has a placeholder
        assert len(on.prompts) == 2 and r_on.generation.outcome == "completion"
        assert r_on.units[0].placeholder_retry["fixed"] and r_on.package is not None
        assert r_off.package is None
    else:
        assert len(on.prompts) == 1 and r_on.units[0].placeholder_retry is None


def test_fidelity_labels_a_leak_as_a_leak_not_as_no_change():
    fork = "void f() {\n  a();\n}\n"
    f = judge(fork, "void f() {\n  /* RCAP_PH_1 */\n  a();\n}\n", "void f() {\n  a();\n}\n", "void f() {\n  b();\n}\n")
    assert f.label == PLACEHOLDER_LEAK and "RCAP_PH_1" in f.evidence[0]
