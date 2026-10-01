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

from rcap.context import AdaptationContext
from rcap.generate import Backend, GenerationRecord, effective_limit_chars, generate
from rcap.request import AdaptationRequest, synthesize

DEFAULT_K = 10
SHRINK_LINE_RATIO = 0.70   # flag when < 70% of attempt 1's lines remain (> 30% dropped)
ECHO_MESSAGE = (
    "The previous attempt is identical to the ORIGINAL target function: the "
    "source change was not applied at all. Apply the change shown in the "
    "before/after pair to the target function.")

_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"[A-Za-z_$][\w$]*|\d+|\S")
_COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*", re.DOTALL)
_TEST_METHOD = re.compile(r"@(?:Test|ParameterizedTest|RepeatedTest|TestTemplate|TestFactory)\b")
_ASSERTION = re.compile(r"\b(?:assert\w*|verify\w*|fail|intercept|expectThrows)\s*[(\[{]")


class RepairFeedback(BaseModel):
    """What one failed attempt feeds forward into the next request."""

    attempt: int                 # the attempt this feedback is FOR (2..k)
    kind: str                    # build | test | echo
    previous_candidate: str      # the failing candidate, as the model wrote it
    error_text: str              # rendered diagnostics, possibly truncated
    truncated: bool = False
    full_chars: int = 0          # length of the diagnostics before truncation


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
    """Build the repair context/request, truncating the FEEDBACK (never the
    code) so the request fits `limit_chars`. If the code alone exceeds the
    limit the request is returned oversized and `generate` refuses it."""
    full = feedback.error_text
    feedback = feedback.model_copy(update={"full_chars": len(full)})

    def build(fb: RepairFeedback):
        ctx = RepairContext(**dict(context), feedback=fb)
        return ctx, synthesize(ctx)

    ctx, request = build(feedback)
    if not limit_chars or len(request.prompt) <= limit_chars:
        return ctx, request
    # The truncation note costs a line itself, so cut with a margin and at a
    # line boundary; two passes settle it.
    for _ in range(3):
        over = len(request.prompt) - limit_chars
        if over <= 0:
            break
        keep = max(0, len(feedback.error_text) - over - 160)
        cut = full[:keep]
        if "\n" in cut:
            cut = cut[:cut.rfind("\n")]
        feedback = feedback.model_copy(update={"error_text": cut, "truncated": True})
        ctx, request = build(feedback)
        if keep == 0:
            break
    return ctx, request


_STOP_FOR_OUTCOME = {
    "limits_exceeded": "limits",
    "unparseable": "unparseable",
    # the paper breaks on any extraction failure; these are the same class
    "placeholder_violation": "unparseable",
    "no_output": "unparseable",
    "backend_error": "unparseable",
}


def run_reprompt_loop(
    units: list[UnitState], backend: Backend,
    validate: Callable[[int, list[str]], Validation],
    *, k: int = DEFAULT_K,
    on_attempt: Callable[[int, list[dict[str, object]]], None] | None = None,
) -> dict[str, object]:
    """Run the loop for one case and return its record.

    `validate(attempt, expanded_candidates)` applies the candidates (one per
    unit, placeholders expanded), builds, tests and restores, and is called
    after EVERY attempt. `on_attempt(attempt, unit_artifacts)` lets the driver
    persist each attempt's prompt and candidate before validation.
    """
    attempts: list[dict[str, object]] = []
    record: dict[str, object] = {
        "k": k, "units": len(units), "attempts": attempts,
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

    for attempt in range(1, k + 1):
        row: dict[str, object] = {
            "attempt": attempt,
            "units": [{
                "unit": u.index, "entity": u.entity,
                "reprompted": reprompted[i] if attempt > 1 else None,
                "outcome": gens[i].outcome,
                "prompt_chars": len(requests[i].prompt),
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
            break

        expanded = [u.expand(raw[i]) for i, u in enumerate(units)]
        if first_expanded is None:
            first_expanded = expanded
        if on_attempt:
            on_attempt(attempt, [{"unit": u.index, "entity": u.entity,
                                  "prompt": requests[i].prompt, "candidate_raw": raw[i],
                                  "candidate_expanded": expanded[i]}
                                 for i, u in enumerate(units)])

        vs1 = measure("\n".join(expanded), "\n".join(first_expanded))
        vst = measure("\n".join(expanded), "\n".join(u.target_original for u in units))
        echo = all(is_echo(expanded[i], u.target_original) for i, u in enumerate(units))
        reasons = shrink_reasons(vs1)
        row.update({
            "metrics": {"vs_attempt1": vs1, "vs_target_original": vst},
            "echo": echo,
            "code_equal_to_target": all(code_equal(expanded[i], u.target_original)
                                        for i, u in enumerate(units)),
            "suspicious_shrink": bool(reasons), "shrink_reasons": reasons,
        })

        signature = tuple(_norm(c) for c in expanded)
        if signature in seen:
            # Deterministic generation: the same candidate fails the same way
            # and yields the same next prompt. Validating it again is wasted.
            row["verdict"] = "CYCLE"
            row["cycle_of_attempt"] = seen.index(signature) + 1
            record["stop_reason"] = "cycle"
            break
        seen.append(signature)

        if echo:
            validation = Validation(verdict="ECHO", kind="echo", feedback=ECHO_MESSAGE,
                                    summary="candidate identical to the unmodified target")
        else:
            validation = validate(attempt, expanded)
        row.update({"verdict": validation.verdict, "error_summary": validation.summary,
                    "validation_seconds": validation.seconds,
                    "validation": validation.detail})

        if validation.verdict == "VALID":
            record["first_success_attempt"] = attempt
            record["stop_reason"] = "success"
            record["success_class"] = "suspicious_shrink" if reasons else "clean"
            break
        if validation.verdict not in ("BUILD_FAIL", "TEST_FAIL", "ECHO"):
            # Nothing actionable to feed forward (no test class, timeout,
            # splice failure): stop rather than re-prompt blind.
            record["stop_reason"] = validation.verdict.lower()
            break
        if attempt == k:
            record["stop_reason"] = "k_exhausted"
            break

        # ---- re-prompt: a fresh single-turn request per unit that needs one
        targets = validation.units_with_errors or list(range(len(units)))
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
                previous_candidate=raw[i],
                error_text=validation.unit_feedback.get(i, validation.feedback))
            ctx, request = fit_feedback(u.context, fb, limit)
            t0 = time.time()
            gens[i] = generate(request, backend,
                               expected_placeholders=u.expected_placeholders)
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
              if a.get("verdict") not in (None, "CYCLE", "NO_CANDIDATE")]
    record["final_verdict"] = judged[-1] if judged else "NO_CANDIDATE"
    record["attempts_run"] = len(attempts)
    return record
