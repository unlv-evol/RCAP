"""Round 2 of the re-prompt experiment: PR-level validation in the loop, the
typed refusal, the deterministic co-change pre-check, and the repair prompt
fixes. Stub backends only."""

import hashlib

import pytest

from rcap import cochange as cc
from rcap.feedback import (
    Span,
    TestFailure,
    attribute_errors,
    attribute_test_failures,
    extract_test_feedback,
    failing_tests_intro,
    member_signatures,
    missing_symbols,
    parse_compile_errors,
    rank_members,
    render_errors_for,
)
from rcap.generate import StubBackend, generate
from rcap.pipeline import _expected_placeholders, run_case
from rcap.prloop import VALID_PR, PRUnit, PRValidation, UnitDiagnostic, run_pr_loop
from rcap.reduction_program import PLACEHOLDER, recover
from rcap.reprompt import (
    ECHO_RETRY_NOTE,
    RepairFeedback,
    UnitState,
    Validation,
    fit_feedback,
    parse_decline,
    run_reprompt_loop,
)
from rcap.request import INSTRUCTIONS, REPAIR_TEMPLATE_ID, synthesize, untraceable_lines

FIRST = "```java\nvoid validate(Config c) { /* RCAP_PH_target.1 */ check(c); missing(c); }\n```"


def _fn(body: str) -> str:
    return f"```java\nvoid validate(Config c) {{ /* RCAP_PH_target.1 */ {body} }}\n```"


class FnBackend(StubBackend):
    """Answers with `respond(request, n)`; records every request."""

    def __init__(self, respond):
        super().__init__(FIRST)
        self._respond = respond
        self.requests = []

    def generate(self, request):
        self.requests.append(request)
        return self._respond(request, len(self.requests))


def _pr_units(sap_dir, pr_manifest, names=("A", "B"), first=FIRST):
    """PR units built from the one fixture case: same context, distinct keys
    and files, as if two SAPs of one PR."""
    out = []
    for name in names:
        res = run_case(sap_dir, pr_manifest, StubBackend(first))
        u = res.units[0]
        ent = u.context.target_localization.get("function")
        art = next(a for a in u.context.program_context
                   if a.role == "target" and a.entity == ent)

        def expand(c, art=art):
            for e in art.placeholders:
                c = c.replace(PLACEHOLDER.format(n=e.ph_id), e.original_text, 1)
            return c

        out.append(PRUnit(
            key=f"sap-{name}::fn", sap=f"sap-{name}", file=f"m/src/main/java/{name}.java",
            index=0, entity=ent, context=u.context, request=u.request,
            generation=u.generation, target_original=recover(art), expand=expand,
            expected_placeholders=_expected_placeholders(u.context)))
    return out


def _diag(text="- A.java:3 [line 1 of your function]: cannot find symbol\n"
               "      symbol:   method missing(Config)", kind="build", missing=None):
    return UnitDiagnostic(kind=kind, text=text, summary="1 compile error(s)",
                          missing=missing or [], how=["span"])


# --------------------------------------------------------------- PR-level loop

def test_pr_loop_reprompts_only_the_charged_unit_and_revalidates_together(sap_dir, pr_manifest):
    units = _pr_units(sap_dir, pr_manifest)
    backend = FnBackend(lambda r, n: _fn("check(c); fixed(c);"))
    states = []

    def validate(rnd, applied, mech):
        states.append(dict(applied))
        if "fixed" in applied["sap-A::fn"]:
            return PRValidation(verdict="VALID", unit_verdicts={k: "VALID" for k in applied})
        return PRValidation(
            verdict="BUILD_FAIL", unit_diags={"sap-A::fn": _diag()},
            unit_verdicts={"sap-A::fn": "BUILD_FAIL", "sap-B::fn": "BLOCKED"})

    seen = []
    rec = run_pr_loop(units, backend, validate, k=10,
                      on_attempt=lambda u, attempt, art: seen.append((u.key, attempt)))
    assert rec["stop_reason"] == "settled" and rec["rounds_run"] == 2
    assert rec["attempt1_verdict"] == "BUILD_FAIL" and rec["final_verdict"] == "VALID"
    assert len(backend.requests) == 1, "only the charged unit is re-prompted"
    assert backend.requests[0].template_id == REPAIR_TEMPLATE_ID
    a, b = rec["units"]["sap-A::fn"], rec["units"]["sap-B::fn"]
    assert a["final_verdict"] == VALID_PR and a["first_success_attempt"] == 2
    assert a["attempt1_verdict"] == "BUILD_FAIL" and a["attempts_run"] == 2
    # B's attempt-1 candidate was judged only once A stopped breaking the build:
    assert b["attempt1_verdict"] == "BLOCKED" and b["final_verdict"] == VALID_PR
    assert b["first_success_attempt"] == 1 and b["attempts_run"] == 1
    # every round validates the COMBINED state, sibling's latest candidate included
    assert all(set(s) == {"sap-A::fn", "sap-B::fn"} for s in states)
    assert states[1]["sap-B::fn"] == states[0]["sap-B::fn"]
    assert seen == [("sap-A::fn", 1), ("sap-B::fn", 1), ("sap-A::fn", 2)], "persisted per attempt"
    assert [v["verdict"] for v in b["attempts"][0]["verdicts"]] == ["BLOCKED", "VALID"]


def test_pr_loop_precheck_skips_the_model_and_drops_the_unit(sap_dir, pr_manifest):
    units = _pr_units(sap_dir, pr_manifest)
    backend = FnBackend(lambda r, n: pytest.fail("the model must not be called"))
    evidence = {"symbol": "missing", "file": "m/src/main/java/Other.java", "line": 7,
                "available": False, "why": "no SAP unit covers this line"}

    def validate(rnd, applied, mech):
        if "sap-A::fn" in applied:
            return PRValidation(verdict="BUILD_FAIL", unit_diags={"sap-A::fn": _diag()},
                                unit_verdicts={"sap-A::fn": "BUILD_FAIL", "sap-B::fn": "BLOCKED"})
        return PRValidation(verdict="VALID", unit_verdicts={k: "VALID" for k in applied})

    rec = run_pr_loop(units, backend, validate, k=10,
                      precheck=lambda u, d, applied, mech: [evidence])
    a, b = rec["units"]["sap-A::fn"], rec["units"]["sap-B::fn"]
    assert a["stop_reason"] == "needs_cochange" and a["in_final_state"] is False
    assert a["stop_detail"]["source"] == "precheck" and a["stop_detail"]["evidence"] == [evidence]
    assert a["final_verdict"] == "BUILD_FAIL" and a["attempts_run"] == 1
    assert rec["final_applied_units"] == ["sap-B::fn"], "the dropped unit is reverted"
    assert b["final_verdict"] == VALID_PR
    assert rec["rounds"][0]["actions"][0]["source"] == "precheck"


def test_pr_loop_model_refusal_is_needs_cochange_not_unparseable(sap_dir, pr_manifest):
    units = _pr_units(sap_dir, pr_manifest, names=("A",))
    backend = FnBackend(lambda r, n: "CANNOT_ADAPT: needs logConfig() from BufferConfigInternal.java")
    rec = run_pr_loop(
        units, backend,
        lambda rnd, applied, mech: (PRValidation(
            verdict="BUILD_FAIL", unit_diags={"sap-A::fn": _diag()},
            unit_verdicts={"sap-A::fn": "BUILD_FAIL"}) if applied
            else PRValidation(verdict="NOTHING_APPLIED")), k=10)
    a = rec["units"]["sap-A::fn"]
    assert a["stop_reason"] == "needs_cochange" and a["stop_detail"]["source"] == "model"
    assert a["stop_detail"]["evidence"][0]["symbol"] == "logConfig()"
    assert a["stop_detail"]["evidence"][0]["file"] == "BufferConfigInternal.java"
    assert a["attempts"][1]["outcome"] == "needs_cochange"
    assert a["attempts"][1]["verdicts"][0]["verdict"] == "DECLINED"
    assert a["final_verdict"] == "BUILD_FAIL" and a["first_success_attempt"] is None


def test_pr_loop_echo_retry_keeps_diagnostics_then_drops(sap_dir, pr_manifest):
    units = _pr_units(sap_dir, pr_manifest, names=("A",))
    echo = "```java\n" + units[0].context.transformation["target"] + "```"
    backend = FnBackend(lambda r, n: echo)
    builds = []

    def validate(rnd, applied, mech):
        builds.append(sorted(applied))
        if not applied:
            return PRValidation(verdict="NOTHING_APPLIED")
        return PRValidation(verdict="BUILD_FAIL", unit_diags={"sap-A::fn": _diag()},
                            unit_verdicts={"sap-A::fn": "BUILD_FAIL"})

    rec = run_pr_loop(units, backend, validate, k=10)
    a = rec["units"]["sap-A::fn"]
    assert a["stop_reason"] == "echo" and a["echo"] and a["echoes"] == 2
    assert builds == [["sap-A::fn"], []], "an echo is never built"
    first, second = (r.prompt for r in backend.requests)
    assert ECHO_RETRY_NOTE not in first and ECHO_RETRY_NOTE in second
    # (f) the follow-up restates the FAILING candidate and its diagnostics
    assert "symbol:   method missing(Config)" in second
    assert "check(c); missing(c);" in second.split("## PREVIOUS ATTEMPT")[1]
    assert [x["verdicts"][0]["verdict"] for x in a["attempts"][1:]] == ["ECHO", "ECHO"]


def test_pr_loop_cycle_and_k_cap(sap_dir, pr_manifest):
    # the re-prompt keeps answering with attempt 1's candidate -> cycle, not rebuilt
    units = _pr_units(sap_dir, pr_manifest, names=("A",))
    builds = []

    def failing(rnd, applied, mech):
        builds.append(rnd)
        if not applied:
            return PRValidation(verdict="NOTHING_APPLIED")
        return PRValidation(
            verdict="BUILD_FAIL", unit_verdicts={"sap-A::fn": "BUILD_FAIL"},
            unit_diags={"sap-A::fn": _diag(text=f"- error of round {rnd}")})

    rec = run_pr_loop(units, FnBackend(lambda r, n: FIRST), failing, k=10)
    a = rec["units"]["sap-A::fn"]
    assert a["stop_reason"] == "cycle" and a["stop_detail"]["cycle_of_attempt"] == 1
    assert a["attempts"][-1]["verdicts"][-1]["verdict"] == "CYCLE"
    assert builds == [1, 2], "the repeated candidate is not rebuilt (round 2 is the empty state)"

    # distinct candidates every time -> stops at k total attempts, all validated
    units = _pr_units(sap_dir, pr_manifest, names=("A",))
    backend = FnBackend(lambda r, n: _fn(f"check(c); v{n}(c);"))
    rec = run_pr_loop(units, backend, failing, k=4)
    a = rec["units"]["sap-A::fn"]
    assert a["stop_reason"] == "k_exhausted" and a["attempts_run"] == 4
    assert len(backend.requests) == 3, "k total attempts: 1 normal + k-1 re-prompts"
    assert all(x["verdicts"][0]["verdict"] == "BUILD_FAIL" for x in a["attempts"])


def test_pr_loop_same_candidate_is_retried_when_the_surrounding_state_changed(
        sap_dir, pr_manifest):
    """A unit may return to an earlier candidate; that is a cycle only if the
    siblings around it are also what they were when it failed."""
    units = _pr_units(sap_dir, pr_manifest)

    def respond(request, n):
        # A: v2 then back to its attempt-1 text; B: fixed at its first re-prompt
        return [_fn("check(c); a2(c);"), _fn("check(c); bfix(c);"), FIRST][n - 1]

    def validate(rnd, applied, mech):
        b_ok = "bfix" in applied.get("sap-B::fn", "")
        if b_ok and "missing(c)" in applied.get("sap-A::fn", ""):
            return PRValidation(verdict="VALID", unit_verdicts={k: "VALID" for k in applied})
        diags = {"sap-A::fn": _diag(text=f"- A fails in round {rnd}")}
        if not b_ok:
            diags["sap-B::fn"] = _diag(text="- B fails")
        return PRValidation(verdict="BUILD_FAIL", unit_diags=diags,
                            unit_verdicts={k: "BUILD_FAIL" for k in applied})

    rec = run_pr_loop(units, FnBackend(respond), validate, k=10)
    a = rec["units"]["sap-A::fn"]
    assert a["final_verdict"] == VALID_PR and a["stop_reason"] == "success"
    assert a["first_success_attempt"] == 3, "attempt 3 repeats attempt 1's text, next to a fixed B"


def test_pr_loop_mechanical_hunks_are_dropped_not_reprompted(sap_dir, pr_manifest):
    units = _pr_units(sap_dir, pr_manifest, names=("A",))
    backend = FnBackend(lambda r, n: pytest.fail("no model call expected"))
    seen = []

    def validate(rnd, applied, mech):
        seen.append(list(mech))
        if "f@@1" in mech:
            return PRValidation(verdict="BUILD_FAIL", mech_diags={"f@@1": "compile error in the hunk"},
                                unit_verdicts={k: "BLOCKED" for k in applied})
        if mech:   # fails, nothing chargeable: every remaining mechanical hunk goes
            return PRValidation(verdict="TEST_FAIL", unattributed=["T.x: boom"],
                                unit_verdicts={k: "VALID" for k in applied})
        return PRValidation(verdict="VALID", unit_verdicts={k: "VALID" for k in applied})

    rec = run_pr_loop(units, backend, validate, k=10, mech=["f@@1", "g@@1", "h@@1"])
    assert seen == [["f@@1", "g@@1", "h@@1"], ["g@@1", "h@@1"], []]
    assert rec["mech"]["f@@1"]["status"] == "dropped" and rec["final_mech"] == []
    assert rec["mech"]["g@@1"]["reason"] == "unattributed failure"
    assert rec["units"]["sap-A::fn"]["final_verdict"] == VALID_PR
    assert rec["units"]["sap-A::fn"]["attempt1_verdict"] == "BLOCKED"


def test_pr_loop_unattributed_failure_stops_and_leaves_units_blocked(sap_dir, pr_manifest):
    units = _pr_units(sap_dir, pr_manifest)
    rec = run_pr_loop(
        units, FnBackend(lambda r, n: pytest.fail("no model call expected")),
        lambda rnd, applied, mech: PRValidation(
            verdict="BUILD_FAIL", unattributed=["Other.java:9: something else"],
            unit_verdicts={k: "BLOCKED" for k in applied}), k=10)
    assert rec["stop_reason"] == "unattributed" and rec["final_verdict"] == "BUILD_FAIL"
    assert {u["final_verdict"] for u in rec["units"].values()} == {"BLOCKED"}
    assert {u["stop_reason"] for u in rec["units"].values()} == {"blocked"}


def test_pr_loop_attempt_one_echo_and_bad_splice_are_not_applied(sap_dir, pr_manifest):
    target = _pr_units(sap_dir, pr_manifest, names=("X",))[0].context.transformation["target"]
    units = _pr_units(sap_dir, pr_manifest, names=("A",), first="```java\n" + target + "```") \
        + _pr_units(sap_dir, pr_manifest, names=("B", "C"))
    states = []

    def validate(rnd, applied, mech):
        states.append(sorted(applied))
        if "sap-B::fn" in applied:   # validated WITHOUT B, which is reported back
            return PRValidation(verdict="VALID", not_spliced={"sap-B::fn": "found 0 times"},
                                unit_verdicts={"sap-C::fn": "VALID"})
        return PRValidation(verdict="VALID", unit_verdicts={k: "VALID" for k in applied})

    rec = run_pr_loop(units, FnBackend(lambda r, n: FIRST), validate, k=10)
    assert states == [["sap-B::fn", "sap-C::fn"]], "the echo is never applied; no extra round"
    assert rec["units"]["sap-A::fn"]["stop_reason"] == "echo_attempt1"
    assert rec["units"]["sap-A::fn"]["status"] == "not_applied"
    assert rec["units"]["sap-B::fn"]["stop_reason"] == "integration_fail"
    assert rec["units"]["sap-C::fn"]["final_verdict"] == VALID_PR


def test_pr_loop_passes_the_cochange_view_into_the_request(sap_dir, pr_manifest):
    units = _pr_units(sap_dir, pr_manifest, names=("A",))
    backend = FnBackend(lambda r, n: _fn("check(c); fixed(c);"))
    notes = [cc.CoChangeNote(status="APPLIED", path="m/Other.java",
                             summary="adds `void helper(Config c) {`", symbols=["helper"]),
             cc.CoChangeNote(status="NOT APPLIED", path="m/Gone.java", summary="adds `int gone;`")]

    def validate(rnd, applied, mech):
        if "fixed" in applied["sap-A::fn"]:
            return PRValidation(verdict="VALID", unit_verdicts={"sap-A::fn": "VALID"})
        d = _diag()
        d.fork_context = ["Members of Config (m/Config.java) with a name close to `missing`:",
                          "  public void missed(Config c)"]
        return PRValidation(verdict="BUILD_FAIL", unit_diags={"sap-A::fn": d},
                            unit_verdicts={"sap-A::fn": "BUILD_FAIL"})

    rec = run_pr_loop(units, backend, validate, k=10,
                      repair_extras=lambda u, d, applied, mech: {
                          "cochanges": notes, "allowed_symbols": cc.allowed_symbols(notes)})
    prompt = backend.requests[0].prompt
    assert "- [APPLIED] m/Other.java: adds `void helper(Config c) {`" in prompt
    assert "- [NOT APPLIED] m/Gone.java: adds `int gone;`" in prompt
    assert "Names introduced by APPLIED co-changes (you may use these): helper" in prompt
    assert "  public void missed(Config c)" in prompt
    assert rec["units"]["sap-A::fn"]["attempts"][1]["cochanges"] == [
        "APPLIED|", "NOT APPLIED|"]


# --------------------------------------------------------------- typed refusal

def test_generate_parses_the_typed_refusal_only_when_allowed(sap_dir, pr_manifest):
    res = run_case(sap_dir, pr_manifest, StubBackend(FIRST))
    line = "CANNOT_ADAPT: needs logConfig() from streams/BufferConfigInternal.java"
    declined = generate(res.request, StubBackend(line), allow_decline=True)
    assert declined.outcome == "needs_cochange" and declined.candidate is None
    assert declined.diagnostics == [line]
    assert parse_decline(declined.diagnostics) == {
        "line": line, "symbol": "logConfig()", "file": "streams/BufferConfigInternal.java"}
    # markdown noise around it, and a refusal without the requested shape
    assert generate(res.request, StubBackend("`" + line + "`\n"), allow_decline=True) \
        .outcome == "needs_cochange"
    loose = generate(res.request, StubBackend("CANNOT_ADAPT: the fork has no such API"),
                     allow_decline=True)
    assert loose.outcome == "needs_cochange"
    assert parse_decline(loose.diagnostics)["symbol"] is None
    # the default pipeline does not know the token: nothing changes there
    assert generate(res.request, StubBackend(line)).outcome == "unparseable"
    # code that merely mentions the token is still code
    code = "```java\nvoid validate(Config c) { // CANNOT_ADAPT: see above\n }\n```"
    assert generate(res.request, StubBackend(code), allow_decline=True).outcome == "completion"


def _units(sap_dir, pr_manifest, backend):
    res = run_case(sap_dir, pr_manifest, backend)
    u = res.units[0]
    ent = u.context.target_localization.get("function")
    art = next(a for a in u.context.program_context if a.role == "target" and a.entity == ent)

    def expand(c):
        for e in art.placeholders:
            c = c.replace(PLACEHOLDER.format(n=e.ph_id), e.original_text, 1)
        return c

    return [UnitState(index=0, entity=ent, context=u.context, request=u.request,
                      generation=u.generation, target_original=recover(art), expand=expand)]


def test_single_sap_loop_refusal_and_precheck_stop_with_needs_cochange(sap_dir, pr_manifest):
    fail = lambda a, e: Validation(verdict="BUILD_FAIL", kind="build",  # noqa: E731
                                   feedback="- F.java:3: cannot find symbol")
    backend = FnBackend(lambda r, n: FIRST if n == 1 else "CANNOT_ADAPT: needs x from Y.java")
    rec = run_reprompt_loop(_units(sap_dir, pr_manifest, backend), backend, fail, k=10)
    assert rec["stop_reason"] == "needs_cochange" and rec["needs_cochange_source"] == "model"
    assert rec["attempts"][-1]["verdict"] == "DECLINED" and rec["final_verdict"] == "BUILD_FAIL"
    assert rec["first_success_attempt"] is None

    backend = FnBackend(lambda r, n: FIRST)
    rec = run_reprompt_loop(_units(sap_dir, pr_manifest, backend), backend, fail, k=10,
                            precheck=lambda i, v: [{"symbol": "x", "file": "Y.java"}])
    assert rec["stop_reason"] == "needs_cochange" and rec["needs_cochange_source"] == "precheck"
    assert len(backend.requests) == 1, "no repair request is sent"
    assert rec["needs_cochange"] == [{"symbol": "x", "file": "Y.java"}]


# --------------------------------------------------------------- prompt fixes

def _repair(sap_dir, pr_manifest, **fb):
    ctx = run_case(sap_dir, pr_manifest, StubBackend(FIRST)).context
    base = {"attempt": 2, "kind": "build", "error_text": "- F.java:3: cannot find symbol",
            "previous_candidate": "void validate(Config c) { /* RCAP_PH_target.1 */ bad(c); }\n"}
    return ctx, fit_feedback(ctx, RepairFeedback(**{**base, **fb}), fb.pop("limit", None))


def test_repair_prompt_structure(sap_dir, pr_manifest):
    ctx, (rctx, req) = _repair(sap_dir, pr_manifest)
    p = req.prompt
    # (a) the contradiction is gone
    assert "Use only classes, methods and fields the fork demonstrably has" not in p
    assert "names\nintroduced by co-changes marked APPLIED" in p
    # (d) the previous attempt is THE code to fix; the original is a labelled reference
    order = [p.index(h) for h in (
        "## Source function BEFORE", "## Source function AFTER", "## The change to port",
        "## ORIGINAL function in the fork, before the port (reference only; returning it "
        "unchanged is NOT an acceptable answer)",
        "## Co-changes of the same upstream change",
        "## PREVIOUS ATTEMPT (attempt 1): the code to fix",
        "## Compiler diagnostics for the previous attempt", "## Output")]
    assert order == sorted(order)
    assert "(adapt this)" not in p
    # the anti-echo rule and the way out are repeated where the answer is specified
    out = p.split("## Output")[1]
    assert "Returning the ORIGINAL function unchanged is rejected." in out
    assert "CANNOT_ADAPT: needs <symbol or change> from <file>" in out
    # the change is restated as a diff every round
    assert "-    if (v > 2)" in p and "+    if (v >= 2)" in p
    assert untraceable_lines(p, rctx) == []
    # no PR context: said so, rather than an empty section
    assert "None applied: only this function is changed in the fork" in p


def test_default_request_is_byte_identical(sap_dir, pr_manifest):
    res = run_case(sap_dir, pr_manifest, StubBackend(FIRST))
    assert res.request.template_id == "rcap-request-v1"
    p = res.request.prompt
    # golden values computed with the code as it was BEFORE round 2 (092dfd0)
    assert hashlib.sha256(p.encode()).hexdigest() == \
        "4f69a08ef61176bc86789620a7f301a6284e2a85f61576764fc77a6f2c2001c1"
    assert res.request.request_hash == \
        "8927510de0a4a309e7cb607f6e35e954a00bdd3d40b93b82b2918897e4590efb"
    assert INSTRUCTIONS.startswith("You are adapting a source-side code change")
    assert "## Target function in the fork (adapt this)" in p
    assert p.rstrip().endswith("One ```java block containing the full adapted target function. "
                               "Nothing else.")
    for round2 in ("CANNOT_ADAPT", "Co-changes", "ORIGINAL function", "PREVIOUS ATTEMPT",
                   "The change to port"):
        assert round2 not in p
    assert synthesize(res.context).request_hash == res.request.request_hash


def test_cochanges_and_fork_context_are_traceable_and_cut_before_diagnostics(
        sap_dir, pr_manifest):
    notes = [cc.CoChangeNote(status="APPLIED", path=f"m/F{i}.java",
                             summary=f"adds `void helper{i}() {{`", symbols=[f"helper{i}"])
             for i in range(8)]
    fork = [f"  public void member{i}()" for i in range(10)]
    errors = "\n".join(f"- F.java:{i}: cannot find symbol number {i}" for i in range(40))
    extra = {"cochanges": notes, "allowed_symbols": ["helper0"], "fork_context": fork,
             "error_text": errors, "pr_context": True}
    ctx, (rctx, req) = _repair(sap_dir, pr_manifest, **extra)
    assert untraceable_lines(req.prompt, rctx) == [], "context lines all come from the context"
    assert untraceable_lines(req.prompt + "- [APPLIED] m/Fake.java: adds `x`\n", rctx) != []
    full = len(req.prompt)
    code = [rctx.transformation["target"], rctx.feedback.previous_candidate]

    def fitted(limit):
        rc, rq = fit_feedback(ctx, RepairFeedback(
            attempt=2, kind="build", previous_candidate=code[1], **extra), limit)
        assert len(rq.prompt) <= limit and all(c in rq.prompt for c in code), "never the code"
        return rc.feedback, rq.prompt

    fb, p = fitted(full - 100)             # a little too long: fork context goes first
    assert fb.context_truncated == ["fork_context"] and len(fb.cochanges) == 8
    assert len(fb.fork_context) < 10 and not fb.truncated and "number 39" in p
    fb, p = fitted(full - 500)             # more: then the co-change list, from its end
    assert fb.context_truncated == ["fork_context", "cochanges"] and not fb.fork_context
    assert 0 < len(fb.cochanges) < 8 and fb.cochanges_omitted == 8 - len(fb.cochanges)
    assert "more co-change hunks not listed" in p and not fb.truncated
    fb, p = fitted(full - 1500)            # only then the diagnostics
    assert fb.truncated and not fb.cochanges and "diagnostics truncated" in p


def test_test_feedback_wording_does_not_cue_a_revert():
    f = [TestFailure(name="a.FooTest.t()", kind="AssertionFailedError",
                     message="expected: <4> but was: <3>", frames=["at a.FooTest.t(FooTest.java:9)"])]
    plain = extract_test_feedback(f, ()).text
    assert "unmodified fork" not in plain and "not a reason to undo the change" in plain
    own = extract_test_feedback(f, (), unit_is_test=True, pr_context=True).text
    assert "The function being ported is TEST code" in own
    assert "Restoring the old assertions is not a fix" in own and "CANNOT_ADAPT" in own
    assert "together with the applied co-changes" in own
    assert failing_tests_intro(1) != failing_tests_intro(1, unit_is_test=True)
    assert "expected: <4> but was: <3>" in own and "FooTest.java:9" in own


# --------------------------------------------------------------- co-changes

DIFF = """\
diff --git a/s/src/main/java/a/Buffer.java b/s/src/main/java/a/Buffer.java
index 1..2 100644
--- a/s/src/main/java/a/Buffer.java
+++ b/s/src/main/java/a/Buffer.java
@@ -10 +10 @@ public abstract class Buffer {
-        return new Strict(getLogConfig());
+        return new Strict(logConfig());
@@ -12 +12 @@ public abstract class Buffer {
-    // old
+    // new
@@ -20 +20 @@ public abstract class Buffer {
-    public abstract Map<String, String> getLogConfig();
+    public abstract Map<String, String> logConfig();
diff --git a/s/src/main/java/a/Strict.java b/s/src/main/java/a/Strict.java
index 3..4 100644
--- a/s/src/main/java/a/Strict.java
+++ b/s/src/main/java/a/Strict.java
@@ -5 +5 @@ class Strict {
-        return other(getLogConfig());
+        return other(logConfig());
diff --git a/c/src/main/resources/message/Meta.json b/c/src/main/resources/message/Meta.json
index 5..6 100644
--- a/c/src/main/resources/message/Meta.json
+++ b/c/src/main/resources/message/Meta.json
@@ -8,0 +9,2 @@
+    { "name": "ErrorCode", "type": "int16", "versions": "13+",
+      "about": "The top-level error code." },
diff --git a/s/src/main/java/a/New.java b/s/src/main/java/a/New.java
new file mode 100644
index 0000000..7
--- /dev/null
+++ b/s/src/main/java/a/New.java
@@ -0,0 +1,2 @@
+class New {
+}
"""
OLD_BUFFER = [f"line {i}" for i in range(1, 9)] + [
    "    Strict make() {", "        return new Strict(getLogConfig());", "    }", "    // old"] \
    + [f"line {i}" for i in range(13, 20)] + [
    "    public abstract Map<String, String> getLogConfig();", "}"]


def _regions():
    return [cc.UnitRegion(key="sap-Buffer::make", source_file="s/src/main/java/a/Buffer.java",
                          after_span=(9, 11), before_span=(9, 11)),
            cc.UnitRegion(key="sap-Strict::f", source_file="s/src/main/java/a/Strict.java",
                          after_text="        return other(logConfig());\n",
                          before_text="        return other(getLogConfig());\n")]


def test_diff_parsing_and_coverage_by_change_group():
    files = cc.parse_unified_diff(DIFF)
    assert [f.status for f in files] == ["M", "M", "M", "A"]
    assert files[3].old_path is None and len(files[0].hunks) == 3
    assert files[0].hunks[2].added == [(20, "    public abstract Map<String, String> logConfig();")]
    assert files[0].hunks[2].removed[0][0] == 20
    assert files[2].hunks[0].added[0][0] == 9
    cc.classify_hunks(files, _regions())
    cov = {h.hunk_id.split("/")[-1]: (h.coverage, h.covered_by) for f in files for h in f.hunks}
    assert cov["Buffer.java@@1"] == ("full", ["sap-Buffer::make"]), "by line span"
    assert cov["Buffer.java@@3"] == ("none", []), "the declaration is outside every unit"
    assert cov["Strict.java@@1"] == ("full", ["sap-Strict::f"]), "by text, when no span is known"
    assert cov["Meta.json@@1"] == ("none", []) and cov["New.java@@1"] == ("none", [])


def test_precheck_evidence_declaration_outside_the_unit():
    files = cc.parse_unified_diff(DIFF)
    regions = _regions()
    cc.classify_hunks(files, regions)
    ev = cc.find_evidence(["logConfig", "setErrorCode", "nowhere"], files, regions[1],
                          all_regions=regions)
    assert "nowhere" not in ev, "no evidence: not a co-change, the model gets its turn"
    decl = ev["logConfig"]
    assert [e.file for e in decl] == ["s/src/main/java/a/Buffer.java"] and decl[0].line == 20
    assert decl[0].declaration and decl[0].owner is None and decl[0].covered_by == []
    # a generated accessor is found through the schema field that produces it
    schema = ev["setErrorCode"][0]
    assert schema.file.endswith("Meta.json") and schema.matched_as == "ErrorCode"
    assert schema.declaration
    # the unit's own lines are never evidence against it
    own = cc.find_evidence(["logConfig"], [files[1]], regions[1])
    assert own == {}
    # usages count only when no declaration is found, and carry the unit that holds them
    use = cc.find_evidence(["logConfig"], files[:2], None, all_regions=regions)
    assert use["logConfig"][0].declaration
    only_use = cc.find_evidence(["logConfig"], [files[1]], regions[0], all_regions=regions)
    assert not only_use["logConfig"][0].declaration
    assert only_use["logConfig"][0].owner == "sap-Strict::f"


@pytest.mark.parametrize("line,name,expected", [
    ("    public abstract Map<String, String> logConfig();", "logConfig", True),
    ("    private final Time time;", "time", True),
    ("    static final int MAX = 3;", "MAX", True),
    ("    public KafkaEventQueue(Time time, LogContext ctx) {", "KafkaEventQueue", True),
    ("  def logConfig: LogConfig = config", "logConfig", True),
    ("public class Foo extends Bar {", "Foo", True),
    ('    { "name": "ErrorCode", "type": "int16" },', "ErrorCode", True),
    ("        return new Strict(a, logConfig());", "logConfig", False),
    ("        x = this.logConfig();", "logConfig", False),
    ("        logConfig();", "logConfig", False),
    ("        if (time == null) {", "time", False),
    ("import java.util.Optional;", "Optional", False),
    ("public class Foo extends Bar {", "Bar", False),
])
def test_is_declaration(line, name, expected):
    assert cc.is_declaration(line, name) is expected


def test_mechanical_hunks_get_context_and_close_groups_are_merged():
    files = cc.parse_unified_diff(DIFF)
    g1, g2, g3 = files[0].hunks
    separate = cc.with_context([g1, g3], OLD_BUFFER)
    assert len(separate) == 2 and separate[1].members == [g3.hunk_id]
    patch = separate[1].patch("s/src/main/java/a/Buffer.java")
    assert "@@ -17,5 +17,5 @@" in patch, "3 lines before, 1 after: the file ends there"
    assert " line 19\n-    public abstract Map<String, String> getLogConfig();\n" \
           "+    public abstract Map<String, String> logConfig();\n }\n" in patch
    # groups 1 line apart: one hunk, the line between them as context
    merged = cc.with_context([g1, g2], OLD_BUFFER)
    assert len(merged) == 1 and merged[0].members == [g1.hunk_id, g2.hunk_id]
    body = merged[0].patch("x").split("@@\n", 1)[1]
    assert body.index("+        return new Strict(logConfig());") \
        < body.index("     }") < body.index("-    // old") < body.index("+    // new")
    assert "@@ -7,9 +7,9 @@" in merged[0].patch("x")
    # a new file needs no context and keeps /dev/null
    new = cc.with_context(files[3].hunks, [])[0].patch("s/src/main/java/a/New.java")
    assert "new file mode 100644\n--- /dev/null\n+++ b/s/src/main/java/a/New.java\n@@ -0,0 +1,2 @@" in new


def test_cochange_notes_are_ranked_bounded_and_marked():
    files = cc.parse_unified_diff(DIFF)
    regions = _regions()
    cc.classify_hunks(files, regions)
    applied = lambda h: (h.path.endswith("Buffer.java"), "test")  # noqa: E731
    notes, omitted = cc.cochange_notes(
        files, regions[1], applied, priority_hunks=["s/src/main/java/a/Buffer.java@@3"],
        in_fork=lambda f: not f.path.endswith("Meta.json"))
    assert notes[0].summary == "adds `public abstract Map<String, String> logConfig();`"
    assert notes[0].status == "APPLIED" and notes[0].symbols == ["logConfig"]
    assert all(n.path != "s/src/main/java/a/Strict.java" for n in notes), "own lines are not co-changes"
    assert all(not n.path.endswith("Meta.json") for n in notes) and omitted == 1, \
        "a file the fork lacks is counted, not listed"
    grouped = next(n for n in notes[1:] if n.path.endswith("Buffer.java"))
    assert grouped.summary.startswith("2 hunks"), "other files: one line per file and status"
    assert cc.allowed_symbols(notes) == ["logConfig"]
    assert [n.status for n in notes if n.path.endswith("New.java")] == ["NOT APPLIED"]
    few, left = cc.cochange_notes(files, regions[1], applied, max_notes=1)
    assert len(few) == 1 and left == 2


# --------------------------------------------------------------- attribution

JAVAC = """\
/r/s/src/main/java/a/Strict.java:54: error: cannot find symbol
            return new Strict(recordLimit, logConfig());
                                           ^
  symbol:   method logConfig()
  location: class Strict
/r/s/src/main/java/a/Caller.java:9: error: method make in class Buffer cannot be applied to given types;
        buffer.make(1, 2);
              ^
  required: int
  found: int,int
/r/s/src/main/java/a/Strict.java:90: error: incompatible types: int cannot be converted to String
        String s = 1;
                   ^
/r/s/src/test/java/a/Unrelated.java:3: error: ';' expected
        int y = 3
                 ^
/r/k/src/main/java/b/Use.java:7:15:  error: cannot find symbol
"""
SCALAC = """\
[Error] /r/core/src/main/scala/k/A.scala:12:5: value logConfig is not a member of kafka.log.UnifiedLog
[Error] /r/core/src/main/scala/k/A.scala:20: not enough arguments for method apply: (a: Int)
"""


def test_missing_symbols_from_javac_and_scalac():
    names = {(m.name, m.kind, m.owner) for m in missing_symbols(parse_compile_errors(JAVAC))}
    assert ("logConfig", "method", "Strict") in names
    assert ("make", "signature", "Buffer") in names
    errors = parse_compile_errors(JAVAC)
    errors[-1].body, errors[-1].col = ["        return a.nodeId();", " " * 16 + "^"], 17
    assert ("nodeId", "symbol", None) in {
        (m.name, m.kind, m.owner) for m in missing_symbols(errors)}, "bare zinc line: under the caret"
    sc = missing_symbols(parse_compile_errors(SCALAC))
    assert [(m.name, m.owner) for m in sc] == [("logConfig", "UnifiedLog"), ("apply", None)]
    nested = parse_compile_errors(
        "/r/A.java:3: error: cannot find symbol\n  x\n  symbol:   variable down\n"
        "  location: class KafkaEventQueue.EventHandler\n")
    assert missing_symbols(nested)[0].owner == "KafkaEventQueue.EventHandler"


def test_errors_are_charged_to_units_by_span_name_file_and_owner():
    spans = {
        "s/src/main/java/a/Strict.java": [Span("sap-Strict::f", 50, 56, "unit", ("withMax",))],
        "s/src/main/java/a/Buffer.java": [Span("sap-Buffer::make", 9, 11, "unit", ("make",)),
                                          Span("Buffer.java@@3", 20, 20, "mech", ("logConfig",))],
    }
    errors = attribute_errors(parse_compile_errors(JAVAC), spans, root="/r")
    by = {(e.rel.rsplit("/", 1)[-1], e.line): (e.keys, e.how) for e in errors}
    assert by[("Strict.java", 54)] == (["sap-Strict::f"], "span")
    assert by[("Caller.java", 9)] == (["sap-Buffer::make"], "name"), \
        "a caller that no longer matches is charged to the unit that declares the method"
    assert by[("Strict.java", 90)] == (["sap-Strict::f"], "file")
    assert by[("Unrelated.java", 3)] == ([], None), "never guessed"
    text = render_errors_for("sap-Strict::f", errors, "s/src/main/java/a/Strict.java")
    assert "Strict.java:54 [line 5 of your function]: cannot find symbol" in text
    assert "Strict.java:90 [in the edited file, outside your function]" in text
    assert "Caller.java" not in text and "answer CANNOT_ADAPT" in text
    other = render_errors_for("sap-Buffer::make", errors, "s/src/main/java/a/Buffer.java")
    assert "Caller.java:9 [in another file that uses the edited one]" in other
    # an error in a file with both: the mechanical hunk is charged before the unit
    log = "/r/s/src/main/java/a/Buffer.java:15: error: incompatible types: bad\n"
    e = attribute_errors(parse_compile_errors(log), spans, root="/r")[0]
    assert (e.keys, e.how) == (["Buffer.java@@3"], "file")
    # owner: the error names a type whose file has an applied change
    log = ("/r/s/src/main/java/a/Zed.java:4: error: cannot find symbol\n    b.gone();\n"
           "  symbol:   method gone()\n  location: variable b of type Buffer\n")
    e = attribute_errors(parse_compile_errors(log), spans, root="/r")[0]
    assert e.how == "owner" and set(e.keys) == {"sap-Buffer::make", "Buffer.java@@3"}


def test_test_failures_are_charged_by_frame_own_test_then_mapping():
    spans = {"m/src/main/java/a/Foo.java": [Span("sap-Foo::run", 10, 20), Span("sap-Foo::other", 30, 40)],
             "m/src/test/java/a/FooTest.java": [Span("sap-FooTest::t", 2, 4)],
             "m/src/main/java/a/Bar.java": [Span("sap-Bar::b", 1, 9)]}
    sources = {"m/src/test/java/a/FooTest.java": "class FooTest {\n  @Test\n  void ported() {\n  }\n"
                                                 "  void untouched() {\n  }\n}\n"}
    kw = {"test_file_of_class": {"a.FooTest": "m/src/test/java/a/FooTest.java"},
          "mapped_files": {"a.FooTest": ["m/src/main/java/a/Foo.java"],
                           "a.BarTest": ["m/src/main/java/a/Bar.java"]}, "sources": sources}
    frame = TestFailure("a.FooTest.untouched()", "E", "boom",
                        ["at a.Foo.run(Foo.java:12)", "at a.FooTest.untouched(FooTest.java:6)"])
    own = TestFailure("a.FooTest.ported()", "E", "expected 1", [])
    mapped = TestFailure("a.BarTest.x()", "E", "nope", ["at a.BarTest.x(BarTest.java:5)"])
    whole = TestFailure("a.FooTest.untouched()", "E", "later", [])
    got = attribute_test_failures([frame, own, mapped, whole], spans, **kw)
    how = {k: [(f.message, h) for f, h in v] for k, v in got.items()}
    assert how["sap-Foo::run"] == [("boom", "frame"), ("later", "mapped")]
    assert how["sap-FooTest::t"] == [("expected 1", "own_test")], \
        "a failing ported test is charged to the test unit, not to the production file"
    assert how["sap-Bar::b"] == [("nope", "mapped")]
    assert how["sap-Foo::other"] == [("later", "mapped")], "unlocated: every change of the file"


def test_fork_context_member_signatures():
    src = """public class Buffer extends Base {
    private final Time time;
    public Map<String, String> getLogConfig() {
        return compute(x);
    }
    abstract void run();
    final Foo local = new Foo(a);
    if (x) {
}"""
    members = member_signatures(src)
    assert [n for n, _ in members] == ["time", "getLogConfig", "run"]
    assert rank_members("logConfig", members) == ["public Map<String, String> getLogConfig()"]
    assert rank_members("zzz", members) == []
    scala = member_signatures("  def logConfig: LogConfig = x\n  private[log] val foo = 3\n", "scala")
    assert [n for n, _ in scala] == ["logConfig", "foo"]
