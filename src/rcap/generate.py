"""Candidate Generation (RCAP ref section 11).

Invokes the configured backend on the request — the only stage allowed to be
nondeterministic — and normalizes the returned artifact. Records an explicit
outcome: completion (a packageable candidate) or a distinct failure/abstention
state. Completion is not correctness (I11); correctness is SVRP's.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

from pydantic import BaseModel, Field
from salp.structural import grammar_for, parse

from rcap.request import AdaptationRequest

_CODE_BLOCK = re.compile(r"```(?:java)?\s*\n(.*?)```", re.DOTALL)
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
    exec_id: str


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
    limit_chars = getattr(backend, "request_limit_chars", None)
    if limit_chars:
        theta["request_limit_chars"] = limit_chars
    exec_id = hashlib.sha256(
        f"{backend.name}:{backend.model}:{request.request_hash}".encode()).hexdigest()[:16]

    usage: dict[str, int] = {}

    def record(outcome: str, candidate: str | None = None, *, diag: list[str] | None = None):
        return GenerationRecord(
            case_id=request.case_id, outcome=outcome, candidate=candidate,
            diagnostics=diag or [], generation_config=theta, usage=dict(usage),
            request_hash=request.request_hash, exec_id=exec_id)

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

    # Post-hoc runtime evidence of the same limit: a reported input-token
    # count at (or past) the declared context window means the runtime
    # saturated — ollama truncates the prompt silently in that case.
    num_ctx = params.get("num_ctx") if isinstance(params, dict) else None
    if num_ctx and usage.get("input_tokens", 0) >= num_ctx:
        return record("limits_exceeded", diag=[
            (f"runtime evaluated {usage['input_tokens']} prompt tokens, "
             f"saturating the declared context window ({num_ctx}); "
             "the input was truncated by the runtime")])

    if not raw or not raw.strip():
        return record("no_output")

    match = _CODE_BLOCK.search(raw)
    candidate = (match.group(1) if match else raw).strip() + "\n"

    # τ may be a bare method (probe-wrap it in a class) or, for class-level
    # changes, a whole type/compilation unit (parse it bare). Accept either.
    grammar = grammar_for("java")
    def parses(text: str) -> bool:
        tree = parse(text, grammar)
        return tree is not None and not tree.root_node.has_error
    if not (parses("class __RcapProbe {\n" + candidate + "\n}") or parses(candidate)):
        diag = "candidate parses neither as a Java method nor as a compilation unit"
        return record("unparseable", diag=[diag])

    if expected_placeholders is not None:
        got = set(_PH.findall(candidate))
        missing = expected_placeholders - got
        if missing:
            return record("placeholder_violation",
                          diag=[f"missing placeholders: {sorted(missing)}"])

    return record("completion", candidate)
