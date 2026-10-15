"""Re-prompt (build-feedback repair) loop.

After "Towards Automated Test Adaptation in Fork Ecosystems via LLMs"
(Mukelabai et al., FSE Companion '26), Algorithm 2: when a candidate fails
validation, build a FRESH single-turn request (no chat history) from an
adapt-don't-reimplement instruction, the failing candidate, the target-side
context and a truncated log of the most relevant diagnostics; query; apply;
re-validate; stop at the first success or after k attempts. Sampling
parameters are fixed throughout.

Differences from the paper, all deliberate:

* attempt 1 is the ordinary RCAP generation; attempts 2..k are re-prompts;
* EVERY attempt is validated, including the last (the paper's loop compiles at
  the top of each iteration, so its N-th candidate is never checked);
* OUR EXTENSION: test failures trigger a re-prompt too (the paper is
  compile-only);
* the request keeps RCAP's source before/after pair, because the adaptation
  being repaired is a change port, not a free-standing test;
* generation is deterministic, so a candidate identical to an earlier attempt
  ends the loop with `cycle` instead of repeating it (temperature is NOT raised);
* anti-weakening guards: a candidate equal to the unmodified target is an ECHO
  and never a success, and a success that shrank the attempt-1 candidate or
  lost tests/assertions is flagged `suspicious_shrink` and not a clean win.

The feedback rides in the context (`RepairContext.feedback`), so request
synthesis renders it and `untraceable_lines` accounts for every line of it.
This module knows nothing about build systems: validation is a callback.
"""

from __future__ import annotations

import difflib
import hashlib
import re
import time
from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel, Field

from rcap.accept import has_placeholder_leak
from rcap.cochange import CoChangeNote
from rcap.context import AdaptationContext
from rcap.generate import Backend, GenerationRecord, effective_limit_chars, generate
from rcap.request import (AdaptationRequest, conversation_hash, render_transcript,
                          request_chars, synthesize)

DEFAULT_K = 10
SHRINK_LINE_RATIO = 0.70   # flag when < 70% of attempt 1's lines remain (> 30% dropped)
ECHO_MESSAGE = (
    "The previous attempt is identical to the ORIGINAL function: the source "
    "change was not applied at all. Apply the change shown in the before/after "
    "pair to the function.")
ECHO_RETRY_NOTE = (
    "Note: a repair attempt for these diagnostics returned the ORIGINAL function "
    "unchanged. That answer is rejected. Fix the previous attempt shown above. If "
    "the original really is the only version of this function that can work in the "
    "fork, then the change cannot be ported here: answer with the CANNOT_ADAPT line.")
# "retry" (default since round 2): the echo is discarded and the next request
# restates the last FAILING candidate WITH its diagnostics. "feedback" is the
# paper's memoryless variant, which loses the diagnostics after an echo.
ECHO_POLICIES = ("retry", "feedback")

# ---- dialogue mode (`run_reprompt_loop(dialogue=True)`) -------------------
# Instead of a fresh single-turn repair request per attempt, each unit keeps
# ONE conversation: the original request, the model's answer, then one short
# user turn per failed attempt. Nothing is restated: the source change, the
# fork's function and every earlier answer are already in the history.
DIALOGUE_TEMPLATE_ID = "rcap-dialogue-v1"
DIALOGUE_TEMPLATE_VERSION = "1"
_DIALOGUE_RULES = (
    "Keep the source change applied and make the smallest edit to your previous "
    "answer that resolves this. Use only names the fork already has; where the "
    "fork names something differently, use the fork's name; do not invent APIs. "
    "Do not undo the change: an answer equal to the fork's original function is "
    "rejected.\n"
    "If no edit to this one function can work, because it depends on something "
    "the fork does not have, answer with this single line instead:\n"
    "CANNOT_ADAPT: needs <symbol or change> from <file>\n"
    "Otherwise return ONLY the complete fixed function in one ```{lang} code "
    "block, with no commentary.")
_DIALOGUE_HEAD = {
    "build": "Your answer does not compile inside the fork. The diagnostics "
             "(line numbers are relative to the function you returned):",
    "test": "Your answer compiles, but tests fail inside the fork:",
}
DIALOGUE_ECHO = ("Your last answer is identical to the fork's ORIGINAL function: the "
                 "source change is not applied at all. That answer is rejected.")
DIALOGUE_ECHO_BACK = ("Go back to your answer before that one and fix it instead. "
                      "Its diagnostics were:")
DIALOGUE_REPEAT = ("Your last answer is identical to the one you gave in attempt {n} of "
                   "this conversation, which already failed with the diagnostics shown "
                   "then. Repeating it cannot work: give a different fix.")


def dialogue_turn(kind: str, error_text: str, language: str | None, *,
                  earlier_errors: str | None = None, repeat_of: int | None = None) -> str:
    """The user turn that follows a failed attempt in dialogue mode."""
    if kind == "repeat":
        head = DIALOGUE_REPEAT.format(n=repeat_of)
    elif kind == "echo":
        head = DIALOGUE_ECHO if not earlier_errors else (
            DIALOGUE_ECHO + " " + DIALOGUE_ECHO_BACK + "\n\n" + earlier_errors.rstrip())
    else:
        head = _DIALOGUE_HEAD.get(kind, _DIALOGUE_HEAD["build"]) + "\n\n" + error_text.rstrip()
    return head + "\n\n" + _DIALOGUE_RULES.format(lang=language or "java") + "\n"


def assistant_turn(candidate: str, language: str | None) -> str:
    """A code answer as it stands in the history: one fenced block."""
    return f"```{language or 'java'}\n{candidate if candidate.endswith(chr(10)) else candidate + chr(10)}```"
_DECLINE_DETAIL = re.compile(r"needs\s+(?P<symbol>.+?)\s+from\s+(?P<file>\S.*)$")

_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"[A-Za-z_$][\w$]*|\d+|\S")
_COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*", re.DOTALL)
_TEST_METHOD = re.compile(r"@(?:Test|ParameterizedTest|RepeatedTest|TestTemplate|TestFactory)\b")
_ASSERTION = re.compile(r"\b(?:assert\w*|verify\w*|fail|intercept|expectThrows)\s*[(\[{]")


class SymbolContext(BaseModel):
    """What ONE name the build could not resolve is, for the context-aware
    repair request (rcap.adaptctx builds it; rcap.request renders it).

    `origin`: added_by_change (declared on an added line of the same upstream
    change, outside the unit), renamed_by_change (the change renames `old_name`
    to it and the fork still declares `old_name`), predates_change (declared
    in the source project before the change; absent from the fork in this
    form), unknown (neither)."""

    symbol: str
    owner: str | None = None
    origin: str = "unknown"
    usage: str = ""                 # the offending source line of the previous attempt
    upstream_file: str | None = None
    upstream_line: int | None = None
    upstream_text: str = ""         # definition: signature + body, or javadoc + signature
    upstream_signature: str = ""    # declaration header only (the cut-down form)
    upstream_complete: bool = False  # upstream_text holds the whole body
    upstream_lines: int = 0         # length of the full definition
    show_body: bool = True          # False once cut to the signature to fit the limit
    old_name: str | None = None     # renamed_by_change
    # added_by_change, body shown: what inlining it into the unit would need
    names_checked: bool = False     # the two lists below were computed
    missing_in_fork: list[str] = Field(default_factory=list)   # names it uses, nowhere in the fork
    qualify: list[str] = Field(default_factory=list)  # types it uses that the unit's file does not import
    fork_heading: str = ""          # e.g. "The fork's ApiKeys (path) declares"
    fork_members: list[str] = Field(default_factory=list)   # fork-side signatures
    hints: list[str] = Field(default_factory=list)          # e.g. the screen's equivalent
    why: dict[str, object] = Field(default_factory=dict)    # recorded, not rendered


class RepairFeedback(BaseModel):
    """What one failed attempt feeds forward into the next request."""

    attempt: int                 # the attempt this feedback is FOR (2..k)
    kind: str                    # build | test | echo
    previous_candidate: str      # the failing candidate, as the model wrote it
    previous_attempt: int | None = None  # attempt that produced it (default: attempt - 1)
    error_text: str              # rendered diagnostics, possibly truncated
    truncated: bool = False
    full_chars: int = 0          # length of the diagnostics before truncation
    # Round 2. All of it is context, so all of it is traceable and all of it
    # counts against the request limit.
    cochanges: list[CoChangeNote] = Field(default_factory=list)
    cochanges_omitted: int = 0
    allowed_symbols: list[str] = Field(default_factory=list)
    fork_context: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)   # e.g. the echo-retry note
    pr_context: bool = False     # validated with the PR's other changes applied
    context_truncated: list[str] = Field(default_factory=list)  # what was cut to fit
    # Context-aware repair (v3 template). Empty = the v2 request, unchanged.
    symbol_context: list[SymbolContext] = Field(default_factory=list)
    symbol_context_omitted: int = 0
    # True (default): each described name ends with the ready-made decline line
    # for it. False: the decline is offered once, in the task and the output
    # section only (variant added after the pre-filled line was seen to be
    # copied back for an inlinable helper).
    decline_prefill: bool = True


class RepairContext(AdaptationContext):
    """An Adaptation Context plus the repair feedback of the previous attempt."""

    feedback: RepairFeedback


class Validation(BaseModel):
    """Outcome of validating one attempt, returned by the driver's callback."""

    verdict: str                     # VALID | BUILD_FAIL | TEST_FAIL | INCONCLUSIVE | INTEGRATION_FAIL
    kind: str | None = None          # build | test: which diagnostics `feedback` holds
    feedback: str = ""               # rendered diagnostics for all units
    unit_feedback: dict[int, str] = Field(default_factory=dict)  # per-unit rendering
    units_with_errors: list[int] = Field(default_factory=list)   # empty = unmappable
    summary: str = ""
    seconds: float = 0.0
    detail: dict[str, object] = Field(default_factory=dict)


@dataclass
class UnitState:
    """One processing unit entering the loop, with its attempt-1 artifacts."""

    index: int
    entity: str | None
    context: AdaptationContext
    request: AdaptationRequest
    generation: GenerationRecord
    target_original: str                  # placeholder-recovered fork text
    expand: Callable[[str], str]          # candidate (placeholder form) -> full text
    expected_placeholders: set[str] | None = None


def _norm(text: str | None) -> str:
    return _WS.sub(" ", text or "").strip()


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text)


def _code_lines(text: str) -> int:
    return sum(1 for ln in text.splitlines() if ln.strip())


def changed_tokens(a: str, b: str) -> int:
    """Tokens that differ between two texts (replace/delete/insert extents)."""
    ta, tb = _tokens(a), _tokens(b)
    if ta == tb:
        return 0
    if len(ta) + len(tb) > 40000:  # whole-file units: fall back to a line diff
        ta, tb = a.splitlines(), b.splitlines()
    sm = difflib.SequenceMatcher(a=ta, b=tb, autojunk=False)
    return sum(max(i2 - i1, j2 - j1) for op, i1, i2, j1, j2 in sm.get_opcodes()
               if op != "equal")


def measure(candidate: str, reference: str) -> dict[str, object]:
    """Size/similarity of `candidate` against a reference text."""
    c_lines, r_lines = _code_lines(candidate), _code_lines(reference)
    return {
        "changed_tokens": changed_tokens(reference, candidate),
        "lines": c_lines,
        "line_delta": c_lines - r_lines,
        "line_ratio": round(c_lines / r_lines, 3) if r_lines else None,
        "test_methods": len(_TEST_METHOD.findall(candidate)),
        "test_methods_delta": len(_TEST_METHOD.findall(candidate))
        - len(_TEST_METHOD.findall(reference)),
        "assertions": len(_ASSERTION.findall(candidate)),
        "assertions_delta": len(_ASSERTION.findall(candidate))
        - len(_ASSERTION.findall(reference)),
    }


def is_echo(candidate: str, target_original: str) -> bool:
    return _norm(candidate) == _norm(target_original)


def code_equal(candidate: str, target_original: str) -> bool:
    """Equal once comments are stripped: only comments were edited."""
    return _norm(_COMMENT.sub("", candidate)) == _norm(_COMMENT.sub("", target_original))


UNIT_CHANGED = "CHANGED"
UNIT_ECHO = "ECHO"
UNIT_REVERTED = "REVERTED"
REVERT_MESSAGE = (
    "The previous answer put a unit back to the unmodified target code although an earlier "
    "attempt had changed it. Going back to the original is not an adaptation."
)


def unit_states(expanded: list[str], targets: list[str],
                first_expanded: list[str] | None = None) -> list[str]:
    """Per UNIT: CHANGED, ECHO (the fork's code once comments and layout are
    ignored) or REVERTED (the fork's code now, while the unit's first attempt
    differed from it: a later attempt went back).

    The echo guard used to look at the row only: in a multi-unit row a
    re-prompted unit could return the fork's original while its siblings kept
    the row different from the fork, and the row passed. Most raw VALIDs of the
    2026-10-06 re-prompt evaluation were that (PR-12789 AbstractHerder,
    PR-12478 / PR-15154 RetryWithToleranceOperator, PR-14483)."""
    out = []
    for i, cand in enumerate(expanded):
        if not code_equal(cand, targets[i]):
            out.append(UNIT_CHANGED)
        elif first_expanded is not None and not code_equal(first_expanded[i], targets[i]):
            out.append(UNIT_REVERTED)
        else:
            out.append(UNIT_ECHO)
    return out


def shrink_reasons(vs_attempt1: dict[str, object]) -> list[str]:
    reasons = []
    ratio = vs_attempt1.get("line_ratio")
    if isinstance(ratio, float) and ratio < SHRINK_LINE_RATIO:
        reasons.append(f"keeps {ratio:.0%} of attempt 1's lines")
    if vs_attempt1["test_methods_delta"] < 0:
        reasons.append(f"{-vs_attempt1['test_methods_delta']} test method(s) removed")
    if vs_attempt1["assertions_delta"] < 0:
        reasons.append(f"{-vs_attempt1['assertions_delta']} assertion(s) removed")
    return reasons


def fit_feedback(context: AdaptationContext, feedback: RepairFeedback,
                 limit_chars: int | None) -> tuple[RepairContext, AdaptationRequest]:
    """Build the repair context/request, cutting supporting material (never
    the code) so the request fits `limit_chars`: first the fork-side context,
    then the co-change list from its low-priority end, then the diagnostics.
    If the code alone exceeds the limit the request is returned oversized and
    `generate` refuses it."""
    full = feedback.error_text
    feedback = feedback.model_copy(update={"full_chars": len(full)})

    def build(fb: RepairFeedback):
        ctx = RepairContext(**dict(context), feedback=fb)
        return ctx, synthesize(ctx)

    ctx, request = build(feedback)
    if not limit_chars or len(request.prompt) <= limit_chars:
        return ctx, request
    cut: list[str] = []
    while feedback.fork_context and len(request.prompt) > limit_chars:
        feedback = feedback.model_copy(update={"fork_context": feedback.fork_context[:-1]})
        if "fork_context" not in cut:
            cut.append("fork_context")
        ctx, request = build(feedback.model_copy(update={"context_truncated": cut}))
    while feedback.cochanges and len(request.prompt) > limit_chars:
        feedback = feedback.model_copy(update={
            "cochanges": feedback.cochanges[:-1],
            "cochanges_omitted": feedback.cochanges_omitted + 1})
        if "cochanges" not in cut:
            cut.append("cochanges")
        ctx, request = build(feedback.model_copy(update={"context_truncated": cut}))
    # Symbol context last among the supporting material (it is what the v3
    # request is for): bodies are cut to their declaration first, largest
    # first; then whole entries from the end.
    while len(request.prompt) > limit_chars:
        bodies = [i for i, sc in enumerate(feedback.symbol_context)
                  if sc.show_body and sc.upstream_text and sc.upstream_signature
                  and len(sc.upstream_text) > len(sc.upstream_signature)]
        if not bodies:
            break
        i = max(bodies, key=lambda j: len(feedback.symbol_context[j].upstream_text))
        scs = list(feedback.symbol_context)
        scs[i] = scs[i].model_copy(update={"show_body": False})
        feedback = feedback.model_copy(update={"symbol_context": scs})
        if "symbol_bodies" not in cut:
            cut.append("symbol_bodies")
        ctx, request = build(feedback.model_copy(update={"context_truncated": cut}))
    while len(feedback.symbol_context) > 1 and len(request.prompt) > limit_chars:
        feedback = feedback.model_copy(update={
            "symbol_context": feedback.symbol_context[:-1],
            "symbol_context_omitted": feedback.symbol_context_omitted + 1})
        if "symbol_context" not in cut:
            cut.append("symbol_context")
        ctx, request = build(feedback.model_copy(update={"context_truncated": cut}))
    feedback = feedback.model_copy(update={"context_truncated": cut})
    # The truncation note costs a line itself, so cut with a margin and at a
    # line boundary; two passes settle it.
    for _ in range(3):
        over = len(request.prompt) - limit_chars
        if over <= 0:
            break
        keep = max(0, len(feedback.error_text) - over - 160)
        text = full[:keep]
        if "\n" in text:
            text = text[:text.rfind("\n")]
        feedback = feedback.model_copy(update={"error_text": text, "truncated": True})
        ctx, request = build(feedback)
        if keep == 0:
            break
    return ctx, request


def parse_decline(diagnostics: list[str]) -> dict[str, str | None]:
    """`CANNOT_ADAPT: needs <symbol> from <file>` -> its parts (None when the
    model did not follow the shape; the raw line is kept either way)."""
    line = next((d for d in diagnostics if "CANNOT_ADAPT" in d), "")
    m = _DECLINE_DETAIL.search(line)
    return {"line": line,
            "symbol": m.group("symbol").strip("`'\" ") if m else None,
            "file": m.group("file").strip("`'\" .") if m else None}


DIALOGUE_ANSWER_SLACK = 2.0      # a follow-up answer may explain itself before the code
DIALOGUE_ANSWER_EXTRA = 1500     # characters


def _dialogue_answer_chars(expected: int | None, history: list[dict[str, str]]) -> int | None:
    """Expected length of a follow-up answer. In a conversation the model often
    reasons in prose before the code block, and earlier answers show how long
    the function really is: the single-turn estimate cut such an answer at the
    output floor (Vim->Neovim case 187). None stays None (no cap)."""
    if expected is None:
        return None
    longest = max((len(m["content"]) for m in history if m["role"] == "assistant"), default=0)
    return int(max(expected, longest) * DIALOGUE_ANSWER_SLACK) + DIALOGUE_ANSWER_EXTRA


def _dialogue_reprompt(units, targets, transcripts, kind, unit_feedback, feedback, backend,
                       rejects, gens, gen_seconds, requests, raw, *,
                       earlier: Validation | None = None, repeat_of: int | None = None) -> None:
    """Send the next user turn of each targeted unit's conversation; results
    go into the loop's per-unit lists in place."""
    for i in targets:
        u = units[i]
        lang = u.request.language
        text = dialogue_turn(
            kind, unit_feedback.get(i, feedback), lang, repeat_of=repeat_of,
            earlier_errors=(earlier.unit_feedback.get(i, earlier.feedback) if earlier else None))
        history = list(transcripts[i])
        request = AdaptationRequest(
            case_id=u.request.case_id, language=lang,
            template_id=DIALOGUE_TEMPLATE_ID, template_version=DIALOGUE_TEMPLATE_VERSION,
            prompt=text, request_hash=conversation_hash(history, text),
            answer_kind=u.request.answer_kind,
            expected_answer_chars=_dialogue_answer_chars(u.request.expected_answer_chars, history),
            history=history)
        t0 = time.time()
        gens[i] = generate(request, backend, expected_placeholders=u.expected_placeholders,
                           allow_decline=True, reject=rejects[i])
        gen_seconds[i] = round(time.time() - t0, 1)
        requests[i] = request
        if gens[i].outcome == "completion":
            raw[i] = gens[i].candidate


_STOP_FOR_OUTCOME = {
    # the model declined: a typed refusal, neither a parse failure nor a success
    "needs_cochange": "needs_cochange",
    "limits_exceeded": "limits",
    "unparseable": "unparseable",
    "shape_mismatch": "unparseable",
    # the paper breaks on any extraction failure; these are the same class
    "placeholder_violation": "unparseable",
    "duplicated_block": "unparseable",
    "no_output": "unparseable",
    "backend_error": "unparseable",
}


def run_reprompt_loop(
    units: list[UnitState], backend: Backend,
    validate: Callable[[int, list[str]], Validation],
    *, k: int = DEFAULT_K, echo_policy: str = "retry",
    on_attempt: Callable[[int, list[dict[str, object]]], None] | None = None,
    precheck: Callable[[int, Validation], list[dict[str, object]]] | None = None,
    repair_extras: Callable[[int, Validation], dict[str, object]] | None = None,
    dialogue: bool = False, first_prompts: list[str] | None = None,
) -> dict[str, object]:
    """Run the loop for one case and return its record.

    `validate(attempt, expanded_candidates)` applies the candidates (one per
    unit, placeholders expanded), builds, tests and restores, and is called
    after EVERY attempt. `on_attempt(attempt, unit_artifacts)` lets the driver
    persist each attempt's prompt and candidate before validation.

    `echo_policy` says what follows an ECHO (the model returned the unmodified
    target). "feedback": the echo becomes the current candidate and the next
    request only says the change was not applied — memoryless, as in the
    paper, so the original diagnostics are gone. "retry": the echo is
    discarded; the next request restates the last FAILING candidate and its
    diagnostics plus a note that reverting was rejected.

    `precheck(unit_index, validation)` is the deterministic co-change check:
    it returns the evidence for every missing name whose co-change is NOT
    available; a non-empty answer ends the loop with `needs_cochange` and no
    model call. `repair_extras(unit_index, validation)` supplies the round-2
    request context (`cochanges`, `allowed_symbols`, `fork_context`, ...).
    A model answer `CANNOT_ADAPT: ...` also ends the loop with
    `needs_cochange` (`needs_cochange_source` says which of the two it was).

    `dialogue=True`: each unit keeps one conversation instead of getting a
    fresh request per attempt (see DIALOGUE_TEMPLATE_ID above). The first user
    turn is the unit's own request, or `first_prompts[i]` when attempt 1 was
    produced by another prompt (the driver then passes that prompt, so the
    history is what the model actually saw). A candidate identical to an
    earlier one does not end the loop at once: the model is told so, once; a
    second repeat in a row ends it with `cycle`. `echo_policy` and
    `repair_extras` do not apply (the history already holds the diagnostics).
    A conversation that no longer fits the backend's window ends with `limits`.
    """
    if echo_policy not in ECHO_POLICIES:
        raise ValueError(f"echo_policy must be one of {ECHO_POLICIES}")
    attempts: list[dict[str, object]] = []
    record: dict[str, object] = {
        "k": k, "echo_policy": echo_policy, "units": len(units), "attempts": attempts,
        "first_success_attempt": None, "stop_reason": None, "final_verdict": None,
        "success_class": None,
    }
    raw: list[str | None] = [u.generation.candidate for u in units]
    requests = [u.request for u in units]
    gens = [u.generation for u in units]
    gen_seconds: list[float | None] = [None] * len(units)
    truncated = [False] * len(units)
    fb_chars: list[tuple[int, int] | None] = [None] * len(units)
    reprompted = [True] * len(units)
    first_expanded: list[str] | None = None
    seen: list[tuple[str, ...]] = []
    limit = None
    last_failing: tuple[list[str | None], Validation, int] | None = None
    from rcap.pipeline import duplicate_check

    rejects = [duplicate_check(u.context, u.request.language) for u in units]
    # dialogue mode: one transcript per unit, [user, assistant, user, ...]
    transcripts: list[list[dict[str, str]]] = [
        [{"role": "user", "content": (first_prompts[i] if first_prompts else u.request.prompt)}]
        for i, u in enumerate(units)] if dialogue else []
    said: list[int] = [0] * len(units)     # attempt whose answer is the transcript's last turn
    if dialogue:
        record["dialogue"] = {"template_id": DIALOGUE_TEMPLATE_ID,
                              "template_version": DIALOGUE_TEMPLATE_VERSION,
                              "first_turn": "driver-supplied" if first_prompts else "unit request"}
    repeated_last = False

    for attempt in range(1, k + 1):
        row: dict[str, object] = {
            "attempt": attempt,
            "units": [{
                "unit": u.index, "entity": u.entity,
                "reprompted": reprompted[i] if attempt > 1 else None,
                "outcome": gens[i].outcome,
                "prompt_chars": len(requests[i].prompt),
                "conversation_chars": request_chars(requests[i]),
                "history_turns": len(requests[i].history),
                "prompt_sha256": hashlib.sha256(requests[i].prompt.encode()).hexdigest(),
                "request_hash": requests[i].request_hash,
                "exec_id": gens[i].exec_id,
                "template_id": requests[i].template_id,
                "gen_seconds": gen_seconds[i],
                "usage": dict(gens[i].usage),
                "diagnostics": list(gens[i].diagnostics)[:3],
                "feedback_truncated": truncated[i],
                "feedback_chars": fb_chars[i],
            } for i, u in enumerate(units)],
            "feedback_truncated": any(truncated),
        }
        attempts.append(row)
        bad = next((g.outcome for g in gens if g.outcome != "completion"), None)
        if bad:
            row["verdict"] = "NO_CANDIDATE"
            record["stop_reason"] = _STOP_FOR_OUTCOME.get(bad, "unparseable")
            record["stop_detail"] = bad
            if bad == "needs_cochange":
                row["verdict"] = "DECLINED"
                record["needs_cochange_source"] = "model"
                record["needs_cochange"] = [
                    parse_decline(g.diagnostics) for g in gens if g.outcome == bad]
            break

        expanded = [u.expand(raw[i]) for i, u in enumerate(units)]
        if first_expanded is None:
            first_expanded = expanded
        if on_attempt:
            on_attempt(attempt, [{"unit": u.index, "entity": u.entity,
                                  "prompt": (render_transcript(requests[i].history, requests[i].prompt)
                                             if requests[i].history else requests[i].prompt),
                                  "candidate_raw": raw[i],
                                  "candidate_expanded": expanded[i]}
                                 for i, u in enumerate(units)])

        vs1 = measure("\n".join(expanded), "\n".join(first_expanded))
        vst = measure("\n".join(expanded), "\n".join(u.target_original for u in units))
        echo = all(is_echo(expanded[i], u.target_original) for i, u in enumerate(units))
        states = unit_states(expanded, [u.target_original for u in units], first_expanded)
        reasons = shrink_reasons(vs1)
        row.update({
            "metrics": {"vs_attempt1": vs1, "vs_target_original": vst},
            "echo": echo, "unit_states": states,
            "code_equal_to_target": all(code_equal(expanded[i], u.target_original)
                                        for i, u in enumerate(units)),
            "suspicious_shrink": bool(reasons), "shrink_reasons": reasons,
        })

        if dialogue:
            # the answers just judged join the conversation
            for i, u in enumerate(units):
                if said[i] < attempt and (attempt == 1 or reprompted[i]):
                    if attempt > 1:
                        transcripts[i].append({"role": "user", "content": requests[i].prompt})
                    transcripts[i].append({"role": "assistant",
                                           "content": assistant_turn(raw[i], u.request.language)})
                    said[i] = attempt

        signature = tuple(_norm(c) for c in expanded)
        repeat_of = seen.index(signature) + 1 if signature in seen else None
        if dialogue and repeat_of is not None and not repeated_last and attempt < k:
            # Same answer again, but the next request is not the same request:
            # the conversation is longer. Say so once instead of stopping.
            row.update({"verdict": "REPEAT", "cycle_of_attempt": repeat_of})
            repeated_last = True
            # Only the units that were asked for a fix and answered as before
            # repeated themselves; a unit nobody asked kept its answer and has
            # no error to fix (PR-12250 ClusterTestExtensions).
            targets = [i for i in range(len(units)) if reprompted[i]] or list(range(len(units)))
            row["reprompt_units"] = targets
            reprompted = [i in targets for i in range(len(units))]
            _dialogue_reprompt(units, targets, transcripts, "repeat", {}, "", backend,
                               rejects, gens, gen_seconds, requests, raw, repeat_of=repeat_of)
            truncated, fb_chars = [False] * len(units), [None] * len(units)
            for i in range(len(units)):
                if i not in targets:
                    gen_seconds[i] = None
            continue
        repeated_last = False
        if signature in seen:
            # Deterministic generation: the same candidate fails the same way
            # and yields the same next prompt. Validating it again is wasted.
            row["verdict"] = "CYCLE"
            row["cycle_of_attempt"] = seen.index(signature) + 1
            record["stop_reason"] = "cycle"
            break
        seen.append(signature)

        leaks = [i for i, c in enumerate(expanded) if has_placeholder_leak(c)]
        if leaks:
            # never written into the fork, whatever attempt it comes from
            row.update({"verdict": "PLACEHOLDER_LEAK", "placeholder_leak_units": leaks})
            record["stop_reason"] = "placeholder_leak"
            break
        if echo:
            validation = Validation(verdict="ECHO", kind="echo", feedback=ECHO_MESSAGE,
                                    summary="candidate identical to the unmodified target")
        else:
            validation = validate(attempt, expanded)
            reverted = [i for i, st in enumerate(states) if st == UNIT_REVERTED]
            if validation.verdict == "VALID" and (reverted or UNIT_CHANGED not in states):
                # The row builds and passes because a unit went back to the fork's
                # code (or because nothing but comments differs): not a port.
                row["raw_verdict"] = "VALID"
                validation = Validation(
                    verdict="REVERTED" if reverted else "ECHO", kind="echo",
                    feedback=REVERT_MESSAGE if reverted else ECHO_MESSAGE,
                    units_with_errors=reverted,
                    summary=(f"unit(s) {reverted} returned to the unmodified target" if reverted
                             else "no unit differs from the unmodified target in code"),
                    seconds=validation.seconds, detail=validation.detail)
        row.update({"verdict": validation.verdict, "error_summary": validation.summary,
                    "validation_seconds": validation.seconds,
                    "validation": validation.detail})

        if validation.verdict == "VALID":
            record["first_success_attempt"] = attempt
            record["stop_reason"] = "success"
            record["success_class"] = "suspicious_shrink" if reasons else "clean"
            break
        if validation.verdict not in ("BUILD_FAIL", "TEST_FAIL", "ECHO", "REVERTED"):
            # Nothing actionable to feed forward (no test class, timeout,
            # splice failure): stop rather than re-prompt blind.
            record["stop_reason"] = validation.verdict.lower()
            break
        if attempt == k:
            record["stop_reason"] = "k_exhausted"
            break

        if dialogue:
            # ---- dialogue mode: the next turn of each unit's conversation
            targets = validation.units_with_errors or list(range(len(units)))
            row["reprompt_units"] = targets
            reprompted = [i in targets for i in range(len(units))]
            echoed = validation.verdict in ("ECHO", "REVERTED")
            earlier = last_failing[1] if echoed and last_failing is not None else None
            if not echoed:
                last_failing = (list(raw), validation, attempt)
            _dialogue_reprompt(
                units, targets, transcripts, "echo" if echoed else (validation.kind or "build"),
                validation.unit_feedback, validation.feedback, backend, rejects, gens,
                gen_seconds, requests, raw,
                earlier=earlier)
            for i in range(len(units)):
                truncated[i] = False
                fb_chars[i] = ((len(validation.unit_feedback.get(i, validation.feedback)),) * 2
                               if i in targets and not echoed else None)
                if i not in targets:
                    gen_seconds[i] = None
            continue

        basis, basis_attempt = list(raw), attempt
        echo_notes: list[str] = []
        if validation.verdict in ("ECHO", "REVERTED"):
            if echo_policy == "retry" and last_failing is not None:
                # (f) the follow-up to an echo keeps the diagnostics: the echo
                # is discarded and the failing candidate is restated with them.
                basis, validation, basis_attempt = last_failing
                echo_notes = [ECHO_RETRY_NOTE]
                row["reprompt_basis"] = "last failing candidate (echo discarded)"
        else:
            last_failing = (list(raw), validation, attempt)

        # ---- re-prompt: a fresh single-turn request per unit that needs one
        targets = validation.units_with_errors or list(range(len(units)))
        if precheck is not None and validation.kind == "build":
            unavailable = [e for i in targets for e in precheck(i, validation)]
            if unavailable:
                row["precheck"] = unavailable
                record["stop_reason"] = "needs_cochange"
                record["needs_cochange_source"] = "precheck"
                record["needs_cochange"] = unavailable
                break
        row["reprompt_units"] = targets
        row["reprompt_mapping"] = ("errors mapped to unit spans"
                                   if validation.units_with_errors else
                                   "unmappable: all units re-prompted")
        reprompted = [i in targets for i in range(len(units))]
        for i in targets:
            u = units[i]
            if limit is None:
                limit = effective_limit_chars(backend, u.request.language)
            fb = RepairFeedback(
                attempt=attempt + 1, kind=validation.kind or "build",
                previous_candidate=basis[i], previous_attempt=basis_attempt,
                error_text=validation.unit_feedback.get(i, validation.feedback),
                notes=echo_notes,
                **(repair_extras(i, validation) if repair_extras else {}))
            ctx, request = fit_feedback(u.context, fb, limit)
            t0 = time.time()
            gens[i] = generate(request, backend,
                               expected_placeholders=u.expected_placeholders,
                               allow_decline=True, reject=rejects[i])
            gen_seconds[i] = round(time.time() - t0, 1)
            requests[i] = request
            truncated[i] = ctx.feedback.truncated
            fb_chars[i] = (len(ctx.feedback.error_text), ctx.feedback.full_chars)
            if gens[i].outcome == "completion":
                raw[i] = gens[i].candidate
        for i in range(len(units)):
            if i not in targets:
                truncated[i], fb_chars[i], gen_seconds[i] = False, None, None

    # The final verdict is that of the last attempt that was actually judged:
    # a cycle or an unusable re-prompt leaves the previous verdict standing.
    judged = [a["verdict"] for a in attempts
              if a.get("verdict") not in (None, "CYCLE", "REPEAT", "NO_CANDIDATE", "DECLINED")]
    record["final_verdict"] = judged[-1] if judged else "NO_CANDIDATE"
    record["attempts_run"] = len(attempts)
    return record
