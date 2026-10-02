"""Re-prompt loop (Algorithm 2 of the test-adaptation paper, adapted): stop on
first success, the k cap with every attempt validated, feedback present in the
request and traceable to the context, cycle detection, oversize handling, and
the anti-weakening guards."""

import pytest

from rcap.feedback import (
    extract_build_feedback,
    extract_test_feedback,
    parse_compile_errors,
    parse_junit_failures,
)
from rcap.generate import StubBackend, generate
from rcap.pipeline import run_case
from rcap.reduction_program import PLACEHOLDER, recover
from rcap.reprompt import (
    RepairContext,
    RepairFeedback,
    UnitState,
    Validation,
    fit_feedback,
    run_reprompt_loop,
)
from rcap.request import REPAIR_TEMPLATE_ID, synthesize, untraceable_lines

FIRST_BODY = "void validate(Config c) { /* RCAP_PH_target.1 */ check(c); missing(c); }"
FIRST = "```java\n" + FIRST_BODY + "\n```"


class ScriptedBackend(StubBackend):
    """Returns the next scripted response per call; records every request."""

    def __init__(self, responses, *, limit=None):
        super().__init__(responses[0])
        self._responses = list(responses)
        self.requests = []
        if limit is not None:
            self.request_limit_chars = limit

    def generate(self, request):
        self.requests.append(request)
        return self._responses[min(len(self.requests) - 1, len(self._responses) - 1)]


def _fn(body: str) -> str:
    return f"```java\nvoid validate(Config c) {{ /* RCAP_PH_target.1 */ {body} }}\n```"


def _units(sap_dir, pr_manifest, backend):
    res = run_case(sap_dir, pr_manifest, backend)
    out = []
    for i, u in enumerate(res.units):
        ent = u.context.target_localization.get("function")
        art = next(a for a in u.context.program_context
                   if a.role == "target" and a.entity == ent)

        def expand(c, art=art):
            for e in art.placeholders:
                c = c.replace(PLACEHOLDER.format(n=e.ph_id), e.original_text, 1)
            return c

        out.append(UnitState(index=i, entity=ent, context=u.context, request=u.request,
                             generation=u.generation, target_original=recover(art),
                             expand=expand))
    return out


def _failing(calls, *, kind="build", text="- F.java:3: cannot find symbol\n    symbol: method missing(Config)"):
    def validate(attempt, expanded):
        calls.append((attempt, list(expanded)))
        return Validation(verdict="BUILD_FAIL" if kind == "build" else "TEST_FAIL",
                          kind=kind, feedback=text, summary="1 compile error(s)")
    return validate


def test_stops_on_first_success(sap_dir, pr_manifest):
    backend = ScriptedBackend([FIRST, _fn("check(c); a(c);"), _fn("check(c); b(c);"),
                               _fn("check(c); never(c);")])
    units = _units(sap_dir, pr_manifest, backend)
    calls = []

    def validate(attempt, expanded):
        calls.append(attempt)
        ok = "b(c)" in expanded[0]
        return Validation(verdict="VALID" if ok else "BUILD_FAIL", kind="build",
                          feedback="- F.java:3: cannot find symbol")

    rec = run_reprompt_loop(units, backend, validate, k=10)
    assert rec["stop_reason"] == "success"
    assert rec["first_success_attempt"] == 3 and rec["final_verdict"] == "VALID"
    assert rec["success_class"] == "clean"
    assert calls == [1, 2, 3]
    assert len(backend.requests) == 3, "no generation after the first success"
    assert [a["attempt"] for a in rec["attempts"]] == [1, 2, 3]


def test_no_reprompt_when_attempt_one_is_valid(sap_dir, pr_manifest):
    backend = ScriptedBackend([FIRST])
    units = _units(sap_dir, pr_manifest, backend)
    rec = run_reprompt_loop(units, backend,
                            lambda a, e: Validation(verdict="VALID"), k=10)
    assert rec["first_success_attempt"] == 1 and rec["attempts_run"] == 1
    assert len(backend.requests) == 1


def test_k_cap_validates_every_attempt_including_the_last(sap_dir, pr_manifest):
    backend = ScriptedBackend([FIRST] + [_fn(f"check(c); v{i}(c);") for i in range(20)])
    units = _units(sap_dir, pr_manifest, backend)
    calls = []
    rec = run_reprompt_loop(units, backend, _failing(calls), k=4)
    assert rec["stop_reason"] == "k_exhausted" and rec["first_success_attempt"] is None
    assert [c[0] for c in calls] == [1, 2, 3, 4], "the k-th candidate is validated too"
    assert len(backend.requests) == 4, "k total generations: 1 normal + k-1 re-prompts"
    assert rec["final_verdict"] == "BUILD_FAIL"


def test_feedback_is_in_the_prompt_and_traceable(sap_dir, pr_manifest):
    backend = ScriptedBackend([FIRST, _fn("check(c); other(c);")])
    units = _units(sap_dir, pr_manifest, backend)
    rec = run_reprompt_loop(units, backend, _failing([]), k=2)
    first, second = backend.requests
    assert "Previous attempt" not in first.prompt
    assert second.template_id == REPAIR_TEMPLATE_ID
    assert "cannot find symbol" in second.prompt
    assert "symbol: method missing(Config)" in second.prompt
    assert FIRST_BODY in second.prompt
    assert "The previous attempt is the code to fix" in second.prompt
    # single turn: the source pair and the fork's original are all restated
    for heading in ("## Source function BEFORE", "## Source function AFTER",
                    "## ORIGINAL function in the fork, before the port"):
        assert heading in second.prompt
    # traceability: the feedback lives in the context, so nothing is untraceable
    fb = RepairFeedback(attempt=2, kind="build",
                        previous_candidate=units[0].generation.candidate,
                        error_text=_failing([])(1, []).feedback)
    ctx, req = fit_feedback(units[0].context, fb, None)
    assert isinstance(ctx, RepairContext) and req.prompt == second.prompt
    assert untraceable_lines(req.prompt, ctx) == []
    assert untraceable_lines(req.prompt + "- F.java:9: injected error\n", ctx) == \
        ["- F.java:9: injected error"]
    # ... and against the plain context the feedback lines ARE flagged
    assert untraceable_lines(req.prompt, units[0].context)
    # distinct identity per attempt
    hashes = [u["request_hash"] for a in rec["attempts"] for u in a["units"]]
    execs = [u["exec_id"] for a in rec["attempts"] for u in a["units"]]
    assert len(set(hashes)) == len(hashes) == 2 and len(set(execs)) == 2


def test_attempt_index_separates_identical_prompts(sap_dir, pr_manifest):
    backend = ScriptedBackend([FIRST])
    ctx = _units(sap_dir, pr_manifest, backend)[0].context
    fb = RepairFeedback(attempt=2, kind="build", previous_candidate="x\n", error_text="e")
    a = fit_feedback(ctx, fb, None)[1]
    b = fit_feedback(ctx, fb.model_copy(update={"attempt": 3}), None)[1]
    assert a.request_hash != b.request_hash


def test_default_request_is_unchanged_by_the_feature(sap_dir, pr_manifest):
    backend = ScriptedBackend([FIRST])
    res = run_case(sap_dir, pr_manifest, backend)
    assert res.request.template_id == "rcap-request-v1"
    assert "Previous attempt" not in res.request.prompt
    assert res.request.prompt.startswith("## Task\nYou are adapting a source-side code change")
    assert synthesize(res.context).request_hash == res.request.request_hash


def test_cycle_detection_stops_without_revalidating(sap_dir, pr_manifest):
    backend = ScriptedBackend([FIRST, _fn("check(c); other(c);"), FIRST])
    units = _units(sap_dir, pr_manifest, backend)
    calls = []
    rec = run_reprompt_loop(units, backend, _failing(calls), k=10)
    assert rec["stop_reason"] == "cycle"
    assert rec["attempts"][-1]["verdict"] == "CYCLE"
    assert rec["attempts"][-1]["cycle_of_attempt"] == 1
    assert [c[0] for c in calls] == [1, 2], "the repeated candidate is not rebuilt"
    assert rec["final_verdict"] == "BUILD_FAIL" and len(backend.requests) == 3


def test_oversize_truncates_feedback_never_code(sap_dir, pr_manifest):
    backend = ScriptedBackend([FIRST, _fn("check(c); other(c);")])
    units = _units(sap_dir, pr_manifest, backend)
    huge = "\n".join(f"- F.java:{i}: cannot find symbol number {i}" for i in range(400))
    base = fit_feedback(units[0].context, RepairFeedback(
        attempt=2, kind="build", previous_candidate=units[0].generation.candidate,
        error_text=""), None)[1]
    limit = len(base.prompt) + 600
    backend.request_limit_chars = limit
    rec = run_reprompt_loop(units, backend, _failing([], text=huge), k=2)
    second = backend.requests[1]
    assert len(second.prompt) <= limit
    assert FIRST_BODY in second.prompt
    assert units[0].context.transformation["target"] in second.prompt
    assert "diagnostics truncated" in second.prompt and "number 399" not in second.prompt
    u = rec["attempts"][1]["units"][0]
    assert u["feedback_truncated"] is True and rec["attempts"][1]["feedback_truncated"]
    assert u["feedback_chars"][0] < u["feedback_chars"][1] == len(huge)


def test_code_alone_over_the_limit_is_refused_as_limits(sap_dir, pr_manifest):
    backend = ScriptedBackend([FIRST])
    units = _units(sap_dir, pr_manifest, backend)
    # the first request fits; the repair request (code restated) cannot
    backend.request_limit_chars = len(units[0].request.prompt) + 20
    rec = run_reprompt_loop(units, backend, _failing([]), k=5)
    assert rec["stop_reason"] == "limits"
    assert len(backend.requests) == 1, "an oversized repair request is never sent"
    assert rec["final_verdict"] == "BUILD_FAIL"


def test_unparseable_reprompt_stops_the_loop(sap_dir, pr_manifest):
    backend = ScriptedBackend([FIRST, "```java\nvoid validate(Config c) { /* RCAP_PH_target.1 */ ((( \n```"])
    units = _units(sap_dir, pr_manifest, backend)
    rec = run_reprompt_loop(units, backend, _failing([]), k=5)
    assert rec["stop_reason"] == "unparseable" and rec["stop_detail"] == "unparseable"
    assert rec["attempts_run"] == 2 and rec["final_verdict"] == "BUILD_FAIL"


def test_echo_is_never_a_success(sap_dir, pr_manifest):
    backend = ScriptedBackend([FIRST])
    units = _units(sap_dir, pr_manifest, backend)
    echo = "```java\n" + units[0].context.transformation["target"] + "```"
    backend = ScriptedBackend([FIRST, echo, _fn("check(c); fixed(c);")])
    units = _units(sap_dir, pr_manifest, backend)
    calls = []

    def validate(attempt, expanded):
        calls.append(attempt)
        return Validation(verdict="VALID" if "fixed" in expanded[0] else "BUILD_FAIL",
                          kind="build", feedback="- F.java:3: cannot find symbol")

    rec = run_reprompt_loop(units, backend, validate, k=5, echo_policy="feedback")
    a2 = rec["attempts"][1]
    assert a2["echo"] is True and a2["verdict"] == "ECHO"
    assert calls == [1, 3], "an echo is not built: it would trivially pass"
    assert "identical to the ORIGINAL function" in backend.requests[2].prompt
    assert rec["first_success_attempt"] == 3


def test_echo_retry_policy_restates_the_failing_candidate(sap_dir, pr_manifest):
    backend = ScriptedBackend([FIRST])
    units = _units(sap_dir, pr_manifest, backend)
    echo = "```java\n" + units[0].context.transformation["target"] + "```"
    backend = ScriptedBackend([FIRST, echo, _fn("check(c); fixed(c);")])
    units = _units(sap_dir, pr_manifest, backend)
    rec = run_reprompt_loop(units, backend, _failing([]), k=3)   # retry is the default
    third = backend.requests[2].prompt
    assert rec["attempts"][1]["verdict"] == "ECHO"
    assert FIRST_BODY in third, "the failing candidate, not the echo, is restated"
    assert "## PREVIOUS ATTEMPT (attempt 1): the code to fix" in third
    assert "symbol: method missing(Config)" in third, "with its diagnostics"
    assert "returned the ORIGINAL function unchanged" in third
    assert third != backend.requests[1].prompt
    assert rec["attempts"][1]["reprompt_basis"].startswith("last failing candidate")


def test_suspicious_shrink_is_flagged_not_a_clean_win(sap_dir, pr_manifest):
    big = "```java\nvoid validate(Config c) {\n/* RCAP_PH_target.1 */\n" + "".join(
        f"    assertEquals({i}, c.get({i}));\n" for i in range(10)) + "}\n```"
    backend = ScriptedBackend([big, _fn("check(c);")])
    units = _units(sap_dir, pr_manifest, backend)
    rec = run_reprompt_loop(
        units, backend,
        lambda a, e: Validation(verdict="VALID" if a == 2 else "TEST_FAIL", kind="test",
                                feedback="- FAILED T.x\n    expected 1 but was 2"), k=5)
    win = rec["attempts"][1]
    assert rec["stop_reason"] == "success" and rec["success_class"] == "suspicious_shrink"
    assert win["suspicious_shrink"] and win["metrics"]["vs_attempt1"]["assertions_delta"] == -10
    assert any("assertion" in r for r in win["shrink_reasons"])
    assert "## Test failures with the previous attempt applied" in backend.requests[1].prompt


JAVAC_LOG = """\
> Task :clients:compileJava
/repo/clients/src/main/java/a/Foo.java:40: warning: [deprecation] old() in Bar has been deprecated
        bar.old();
           ^
/repo/clients/src/test/java/a/FooTest.java:12: error: incompatible types: int cannot be converted to String
        String s = foo.run();
                          ^
/repo/clients/src/main/java/a/Foo.java:105: error: cannot find symbol
        return helper.compute(x);
                     ^
  symbol:   method compute(int)
  location: variable helper of type Helper
/repo/clients/src/main/java/a/Foo.java:230: error: cannot find symbol
        return helper.compute(x);
                     ^
  symbol:   method compute(int)
  location: variable helper of type Helper
/repo/clients/src/main/java/a/Foo.java:300: error: ';' expected
        int y = 3
                 ^
4 errors

FAILURE: Build failed with an exception.
"""


def test_build_feedback_keeps_symbol_lines_orders_and_maps_units():
    errors = parse_compile_errors(JAVAC_LOG)
    assert len(errors) == 4, "the warning is not an error"
    fb = extract_build_feedback(JAVAC_LOG, "clients/src/main/java/a/Foo.java",
                                [(100, 110), (295, 305)], root="/repo")
    assert fb.n_total == 3, "the repeated error is collapsed"
    lines = fb.text.splitlines()
    assert lines[0].startswith("- clients/src/main/java/a/Foo.java:105 [line 6 of your function]")
    assert "symbol:   method compute(int)" in fb.text
    assert "location: variable helper of type Helper" in fb.text
    assert "return helper.compute(x);" in fb.text
    assert "(same error also at lines 230)" in fb.text
    assert fb.text.index("Foo.java:300") < fb.text.index("FooTest.java:12"), \
        "edited-file errors come before other files"
    assert "[in another file that uses the edited one]" in fb.text
    assert "Those files cannot be changed" in fb.text
    assert "deprecat" not in fb.text
    assert fb.units_with_errors == [0, 1] and fb.mappable
    assert "Foo.java:300" not in fb.per_unit[0] and "Foo.java:105" in fb.per_unit[0]
    assert "FooTest.java:12" in fb.per_unit[1], "unattributed errors go to every unit"


def test_build_feedback_caps_and_falls_back():
    log = "\n".join(f"/r/A.java:{i}: error: problem {i}\n    code{i}();\n    ^"
                    for i in range(1, 60))
    fb = extract_build_feedback(log, "A.java", [])
    assert fb.n_shown == 15 and "(44 more errors omitted)" in fb.text
    assert not fb.mappable and fb.units_with_errors == []
    none = extract_build_feedback(
        "FAILURE: Build failed\n* What went wrong:\nExecution failed for task ':x'.\n> boom\n",
        "A.java", [])
    assert none.n_total == 0 and "boom" in none.text


def test_bare_error_line_is_completed_from_the_spliced_source():
    log = "/repo/core/src/test/java/k/T.java:2:15:  error: cannot find symbol\n\nFAILURE: x\n"
    fb = extract_build_feedback(log, "core/src/test/java/k/T.java", [(1, 3)], root="/repo",
                                edited_source="int id() {\n    return a.nodeId();\n}\n")
    assert "    return a.nodeId();" in fb.text and fb.text.splitlines()[2].endswith("^")
    assert fb.text.splitlines()[2].index("^") - 4 == 14


SCALAC_LOG = """\
[Error] /repo/core/src/main/scala/kafka/A.scala:12:5: not found: value foo
[Warn] /repo/core/src/main/scala/kafka/A.scala:3: unused import
[Error] /repo/core/src/main/scala/kafka/A.scala:20: type mismatch;
 found   : Int
 required: String
two errors found
"""


def test_scalac_errors_are_parsed():
    fb = extract_build_feedback(SCALAC_LOG, "core/src/main/scala/kafka/A.scala", [(10, 15)],
                                root="/repo")
    assert fb.n_total == 2 and "unused import" not in fb.text
    assert "A.scala:12 [line 3 of your function]: not found: value foo" in fb.text
    assert "required: String" in fb.text and "errors found" not in fb.text


JUNIT_XML = """<?xml version="1.0" encoding="UTF-8"?>
<testsuite name="a.FooTest" tests="4" failures="3">
  <testcase name="ok()" classname="a.FooTest" time="0.1"/>
  <testcase name="old()" classname="a.FooTest"><failure message="pre-existing" type="E">E: pre-existing
\tat a.FooTest.old(FooTest.java:9)</failure></testcase>
  <testcase name="[1] magic=0" classname="a.FooTest"><failure message="expected: &lt;1&gt; but was: &lt;2&gt;" type="org.opentest4j.AssertionFailedError">org.opentest4j.AssertionFailedError: expected: &lt;1&gt; but was: &lt;2&gt;
\tat org.junit.jupiter.api.AssertionUtils.fail(AssertionUtils.java:55)
\tat a.FooTest.check(FooTest.java:40)
\tat a.Foo.run(Foo.java:12)
\tat java.base/java.lang.reflect.Method.invoke(Method.java:566)</failure></testcase>
  <testcase name="[2] magic=1" classname="a.FooTest"><failure message="expected: &lt;1&gt; but was: &lt;2&gt;" type="org.opentest4j.AssertionFailedError">org.opentest4j.AssertionFailedError: expected: &lt;1&gt; but was: &lt;2&gt;
\tat org.junit.jupiter.api.AssertionUtils.fail(AssertionUtils.java:55)
\tat a.FooTest.check(FooTest.java:40)
\tat a.Foo.run(Foo.java:12)</failure></testcase>
</testsuite>
"""


def test_test_feedback_has_messages_frames_and_only_new_failures(tmp_path):
    xml = tmp_path / "TEST-a.FooTest.xml"
    xml.write_text(JUNIT_XML)
    ran, failures = parse_junit_failures([xml])
    assert ran == 4 and len(failures) == 3
    fb = extract_test_feedback(failures, baseline_failed=["a.FooTest.old()"], ran=ran)
    assert fb.kind == "test" and fb.n_total == 2
    assert "pre-existing" not in fb.text, "baseline failures are not the candidate's"
    assert "expected: <1> but was: <2>" in fb.text
    assert "at a.FooTest.check(FooTest.java:40)" in fb.text
    assert "at a.Foo.run(Foo.java:12)" in fb.text
    assert "org.junit" not in fb.text.replace("org.opentest4j", "") and "java.base" not in fb.text
    assert "and 1 more with the same failure" in fb.text
    assert fb.text.count("FooTest.check") == 1, "identical failures are grouped"


@pytest.mark.parametrize("kind", ["build", "test", "echo"])
def test_generate_accepts_repair_requests(sap_dir, pr_manifest, kind):
    backend = ScriptedBackend([FIRST])
    unit = _units(sap_dir, pr_manifest, backend)[0]
    fb = RepairFeedback(attempt=2, kind=kind, previous_candidate="void f() { /* RCAP_PH_target.1 */ }\n",
                        error_text="- x")
    _, req = fit_feedback(unit.context, fb, None)
    assert generate(req, backend).outcome == "completion"
