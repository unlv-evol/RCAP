"""Candidate Generation (RCAP ref section 11).

Invokes the configured backend on the request — the only stage allowed to be
nondeterministic — and normalizes the returned artifact. Records an explicit
outcome: completion (a packageable candidate) or a distinct failure/abstention
state. Completion is not correctness (I11); correctness is SVRP's.
"""

from __future__ import annotations

import datetime
import hashlib
import re
from typing import Protocol

from pydantic import BaseModel, Field
from salp.structural import grammar_for, parse

from rcap.backend_ollama import CHARS_PER_TOKEN_IMPLAUSIBLE, admissible_chars

from rcap.request import AdaptationRequest

_CODE_BLOCK = re.compile(r"```[A-Za-z0-9_+-]*\s*\n(.*?)```", re.DOTALL)
_PH = re.compile(r"/\* RCAP_PH_[^*]+\*/")
# Any mention of the placeholder prefix, however it is written (`/*RCAP_PH_1*/`,
# `// RCAP_PH_1`): outside the exact comment form it is still a leak.
_PH_ANY = re.compile(r"RCAP_PH_[\w.]*")
# Typed refusal (repair requests only): `CANNOT_ADAPT: needs <symbol> from <file>`.
_DECLINE = re.compile(r"^[\s`*>-]*CANNOT_ADAPT\b\s*:?\s*(.*?)[\s`*]*$", re.MULTILINE)


def placeholder_report(candidate: str, expected: set[str]) -> dict[str, list[str]]:
    """How a candidate handles its placeholder comments.

    `missing`: expected, absent. `duplicated`: expected, more than once (only
    the first is expanded; the others would stay in the fork as comments).
    `unknown`: placeholder comments the target does not have (invented), and
    mentions of the prefix that are not a well-formed placeholder comment."""
    got = _PH.findall(candidate)
    counts: dict[str, int] = {}
    for g in got:
        counts[g] = counts.get(g, 0) + 1
    unknown = sorted(set(counts) - expected)
    loose = len(_PH_ANY.findall(candidate)) - len(got)
    if loose > 0:
        unknown.append(f"{loose} malformed mention(s) of RCAP_PH_")
    return {"missing": sorted(expected - set(counts)),
            "duplicated": sorted(g for g, n in counts.items() if n > 1 and g in expected),
            "unknown": unknown}


def has_placeholder_leak(text: str | None) -> bool:
    """A placeholder marker in text that is about to be written into the fork
    (i.e. AFTER expansion) is never acceptable."""
    return bool(text) and "RCAP_PH_" in text


class Backend(Protocol):
    name: str
    model: str

    def generate(self, request: AdaptationRequest) -> str: ...


class StubBackend:
    """Deterministic test backend: returns a fixed artifact (spec section 11 tests)."""

    name = "stub"
    model = "stub-0"

    def __init__(self, response: str):
        self._response = response

    def generate(self, request: AdaptationRequest) -> str:
        return self._response


class GenerationRecord(BaseModel):
    case_id: str
    outcome: str  # completion | no_output | unparseable | placeholder_violation | limits_exceeded | backend_error
    candidate: str | None = None
    diagnostics: list[str] = Field(default_factory=list)
    generation_config: dict[str, object] = Field(default_factory=dict)  # theta
    # Backend-reported token counts (section 15: Request/Input and Output
    # Tokens, measured separately). Recorded only when the runtime reports
    # them — never estimated. Empty for backends without a tokenizer.
    usage: dict[str, int] = Field(default_factory=dict)
    request_hash: str
    # exec_id is the deterministic (backend, model, request) anchor; the
    # timestamp (section 11: "a timestamp may be added") makes repeated runs —
    # separate executions by definition — distinguishable in stored artifacts.
    exec_id: str
    executed_at: str = ""


def effective_limit_chars(backend: Backend, language: str | None) -> int | None:
    """The request-size bound `generate` enforces for this backend and language.

    The backend's own limit is language-agnostic. When it declares a context
    window, the bound is recomputed for the request's language, which is both
    tighter for dense-token languages and safer for sparse ones."""
    params = getattr(backend, "params", None)
    limit_chars = getattr(backend, "request_limit_chars", None)
    _num_ctx = params.get("num_ctx") if isinstance(params, dict) else None
    if _num_ctx and getattr(backend, "request_limit_chars_is_explicit", False) is False:
        limit_chars = admissible_chars(_num_ctx, language)
    return limit_chars


def generate(request: AdaptationRequest, backend: Backend,
             *, expected_placeholders: set[str] | None = None,
             allow_decline: bool = False) -> GenerationRecord:
    """`allow_decline` (repair requests): an answer `CANNOT_ADAPT: ...` instead
    of code is the distinct outcome `needs_cochange` -- a typed refusal, not an
    unparseable answer and not a completion. Off by default, so the ordinary
    pipeline's outcomes are unchanged."""
    # θ must make a weight change distinguishable from an evidence change
    # (sections 10-12): record generation params and model version when the
    # backend exposes them, not just its name.
    theta: dict[str, object] = {"backend": backend.name, "model": backend.model}
    params = getattr(backend, "params", None)
    if params:
        theta["params"] = dict(params)
    digest = getattr(backend, "model_digest", None)
    if digest:
        theta["version"] = digest
    limit_chars = effective_limit_chars(backend, getattr(request, "language", None))
    if limit_chars:
        theta["request_limit_chars"] = limit_chars
    exec_id = hashlib.sha256(
        f"{backend.name}:{backend.model}:{request.request_hash}".encode()).hexdigest()[:16]

    usage: dict[str, int] = {}

    def record(outcome: str, candidate: str | None = None, *, diag: list[str] | None = None):
        return GenerationRecord(
            case_id=request.case_id, outcome=outcome, candidate=candidate,
            diagnostics=diag or [], generation_config=theta, usage=dict(usage),
            request_hash=request.request_hash, exec_id=exec_id,
            executed_at=datetime.datetime.now(datetime.UTC)
            .isoformat(timespec="seconds"))

    # Section 11: "request or context exceeding backend limits" is a distinct
    # outcome, refused BEFORE invocation against the backend's own declared
    # acceptance limit — an oversized request silently truncated by the runtime
    # is a garbage completion, not a measurement.
    if limit_chars and len(request.prompt) > limit_chars:
        return record("limits_exceeded", diag=[
            (f"request is {len(request.prompt)} chars; backend declares an "
             f"acceptance limit of {limit_chars} chars")])

    try:
        raw = backend.generate(request)
    except Exception as exc:  # noqa: BLE001 - any backend crash is a generation failure state
        return record("backend_error", diag=[str(exc)])
    reported = getattr(backend, "last_usage", None)
    if reported:
        usage.update({k: v for k, v in reported.items() if isinstance(v, int)})

    # Post-hoc runtime evidence of the same limit. The obvious test --
    # input_tokens >= num_ctx -- CANNOT EVER FIRE: when the runtime truncates it
    # does not report a saturated count, it reports a pinned one at
    # num_ctx // 2 + 2, which is far below the window. Two signals that do work:
    #
    #   1. the mechanical signature: a count sitting exactly on that pinned
    #      value;
    #   2. a chars-per-token ratio too high to be real code in any language we
    #      handle (C peaks at 3.4, Scala/Java at 4.9), which catches a pinned
    #      count at some other value.
    #
    # The ratio alone is not enough and must stay language-independent: a 4.5
    # threshold calibrated on C flags ordinary Scala prompts as truncated.
    num_ctx = params.get("num_ctx") if isinstance(params, dict) else None
    in_tok = usage.get("input_tokens", 0)
    if num_ctx and in_tok:
        pinned = in_tok == num_ctx // 2 + 2
        ratio = len(request.prompt) / in_tok
        # The ratio test needs a precondition: truncation is only possible if the
        # prompt filled the window, and it pins the reported count at
        # num_ctx // 2 + 2, so a count far below that cannot be a truncation
        # however unusual its ratio. Without this, ordinary short Java prompts
        # (6.0-6.8 chars/token at ~650-1,560 tokens in a 16,384 window) were
        # labelled truncated -- 10 of 96 limits_exceeded rows in the 477 run.
        could_be_truncated = in_tok >= num_ctx * 0.4
        if pinned or in_tok >= num_ctx or (
                could_be_truncated and ratio > CHARS_PER_TOKEN_IMPLAUSIBLE):
            return record("limits_exceeded", diag=[
                (f"runtime evaluated {in_tok} prompt tokens for a "
                 f"{len(request.prompt)}-char prompt ({ratio:.2f} chars/token, "
                 f"window {num_ctx}"
                 + (", pinned at the truncation value" if pinned else "")
                 + "); the input was truncated by the runtime")])

    if not raw or not raw.strip():
        return record("no_output")

    match = _CODE_BLOCK.search(raw)
    if allow_decline:
        declined = _DECLINE.search(raw)
        # A refusal is the answer when it stands alone or opens the answer; a
        # code block that merely mentions the token in a comment is still code.
        if declined and (match is None or raw.lstrip(" \t\r\n`*>-").startswith("CANNOT_ADAPT")):
            detail = declined.group(1).strip()
            return record("needs_cochange",
                          diag=["CANNOT_ADAPT: " + detail if detail else "CANNOT_ADAPT"])
    candidate = (match.group(1) if match else raw).strip() + "\n"

    # τ may be a bare method (probe-wrap it in a class) or, for class-level
    # changes, a whole type/compilation unit (parse it bare). Accept either.
    # The grammar is the request's language, not an assumed one: Java's grammar
    # rejects every well-formed Scala function, which would report a language
    # mismatch as a model failure.
    lang = request.language or "java"
    grammar = grammar_for(lang)
    if grammar is None:
        return record("unparseable",
                      diag=[f"no tree-sitter grammar available for {lang}"])
    def parses(text: str) -> bool:
        tree = parse(text, grammar)
        return tree is not None and not tree.root_node.has_error
    if not (parses("class __RcapProbe {\n" + candidate + "\n}") or parses(candidate)):
        diag = (f"candidate parses neither as a {lang} method nor as a "
                "compilation unit")
        return record("unparseable", diag=[diag])

    if expected_placeholders is not None:
        report = placeholder_report(candidate, expected_placeholders)
        if report["missing"]:
            # wording unchanged since the 477 run, so stored outcomes stay comparable
            return record("placeholder_violation",
                          diag=[f"missing placeholders: {report['missing']}"])
        if report["duplicated"] or report["unknown"]:
            # A hard rejection: an unknown or repeated placeholder is not
            # expanded and would be written into the fork as a stray comment
            # (or stand where the model deleted code).
            return record("placeholder_violation", diag=[
                f"{kind} placeholders: {report[kind]}"
                for kind in ("duplicated", "unknown") if report[kind]])

    return record("completion", candidate)


def generate_with_placeholder_retry(
    request: AdaptationRequest, backend: Backend, *,
    expected_placeholders: set[str] | None = None,
) -> tuple[GenerationRecord, dict[str, object] | None]:
    """`generate`, plus ONE targeted retry when the answer is rejected only for
    its placeholders.

    The retry is a new single-turn request: the original request followed by
    the exact list of what was wrong (missing / duplicated / unknown) and the
    rule (each original placeholder exactly once, no new one). It is issued
    once; if its answer is anything but a completion the ORIGINAL refusal is
    returned unchanged. Returns (record, retry info or None when no retry was
    issued). Deterministic given a deterministic backend."""
    from rcap.request import synthesize_placeholder_repair

    first = generate(request, backend, expected_placeholders=expected_placeholders)
    if first.outcome != "placeholder_violation" or expected_placeholders is None:
        return first, None
    report = _report_from_diagnostics(first.diagnostics, expected_placeholders)
    repair = synthesize_placeholder_repair(request, sorted(expected_placeholders), report)
    second = generate(repair, backend, expected_placeholders=expected_placeholders)
    info: dict[str, object] = {
        "template_id": repair.template_id, "request_hash": repair.request_hash,
        "prompt_chars": len(repair.prompt), "first_diagnostics": list(first.diagnostics),
        "report": report, "retry_outcome": second.outcome,
        "retry_diagnostics": list(second.diagnostics), "retry_usage": dict(second.usage),
        "fixed": second.outcome == "completion",
    }
    return (second if second.outcome == "completion" else first), info


def _report_from_diagnostics(diags: list[str], expected: set[str]) -> dict[str, list[str]]:
    """The violation lists as `generate` recorded them (the record keeps no
    candidate for a rejected answer)."""
    import ast

    out: dict[str, list[str]] = {"missing": [], "duplicated": [], "unknown": []}
    for d in diags:
        m = re.match(r"^(missing|duplicated|unknown) placeholders: (\[.*\])$", d)
        if m:
            out[m.group(1)] = list(ast.literal_eval(m.group(2)))
    return out
