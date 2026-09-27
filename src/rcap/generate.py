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


def generate(request: AdaptationRequest, backend: Backend,
             *, expected_placeholders: set[str] | None = None) -> GenerationRecord:
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
    # The backend's own limit is language-agnostic. When it declares a context
    # window, the bound is recomputed for the request's language, which is both
    # tighter for dense-token languages and safer for sparse ones.
    limit_chars = getattr(backend, "request_limit_chars", None)
    _num_ctx = params.get("num_ctx") if isinstance(params, dict) else None
    if _num_ctx and getattr(backend, "request_limit_chars_is_explicit", False) is False:
        limit_chars = admissible_chars(_num_ctx, getattr(request, "language", None))
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
        if pinned or ratio > CHARS_PER_TOKEN_IMPLAUSIBLE or in_tok >= num_ctx:
            return record("limits_exceeded", diag=[
                (f"runtime evaluated {in_tok} prompt tokens for a "
                 f"{len(request.prompt)}-char prompt ({ratio:.2f} chars/token, "
                 f"window {num_ctx}"
                 + (", pinned at the truncation value" if pinned else "")
                 + "); the input was truncated by the runtime")])

    if not raw or not raw.strip():
        return record("no_output")

    match = _CODE_BLOCK.search(raw)
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
        got = set(_PH.findall(candidate))
        missing = expected_placeholders - got
        if missing:
            return record("placeholder_violation",
                          diag=[f"missing placeholders: {sorted(missing)}"])

    return record("completion", candidate)
