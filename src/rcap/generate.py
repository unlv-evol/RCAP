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
    outcome: str  # completion | no_output | unparseable | placeholder_violation | backend_error
    candidate: str | None = None
    diagnostics: list[str] = Field(default_factory=list)
    generation_config: dict[str, object] = Field(default_factory=dict)  # theta
    request_hash: str
    exec_id: str


def generate(request: AdaptationRequest, backend: Backend,
             *, expected_placeholders: set[str] | None = None) -> GenerationRecord:
    theta = {"backend": backend.name, "model": backend.model}
    exec_id = hashlib.sha256(
        f"{backend.name}:{backend.model}:{request.request_hash}".encode()).hexdigest()[:16]

    def record(outcome: str, candidate: str | None = None, *, diag: list[str] | None = None):
        return GenerationRecord(
            case_id=request.case_id, outcome=outcome, candidate=candidate,
            diagnostics=diag or [], generation_config=theta,
            request_hash=request.request_hash, exec_id=exec_id)

    try:
        raw = backend.generate(request)
    except Exception as exc:  # noqa: BLE001 - any backend crash is a generation failure state
        return record("backend_error", diag=[str(exc)])

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
