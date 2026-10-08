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
from collections.abc import Callable
from typing import Protocol

from pydantic import BaseModel, Field
from rcap.grammars import grammar_for, parse, tolerates_parse_errors

from rcap.accept import has_placeholder_leak  # noqa: F401  (re-exported: its home is rcap.accept)
from rcap.backend_ollama import CHARS_PER_TOKEN_IMPLAUSIBLE, admissible_chars

from rcap.regions import merge_regions, region_shape_error
from rcap.request import AdaptationRequest
from rcap.unitkind import FILE, FUNCTION, REGION, TYPE, WHOLE_KINDS, text_kind

# Why a unit was not attempted, or its answer not used, for a reason that lies
# in the backend's capacity (window, output budget, speed) and not in what the
# model wrote. Always with outcome `limits_exceeded`. A comparison of two
# models reports results over all units and over the units neither model
# refused for capacity.
CAPACITY_KINDS = (
    "request_too_large",          # refused up front: the request alone passes the request limit
    "answer_cannot_fit_window",   # refused up front: prompt + expected answer pass the window
    "input_truncated",            # the runtime evaluated fewer tokens than the prompt has
    "window_overflow",            # prompt + answer passed the window while answering
    "output_budget_exhausted",    # the answer reached the output budget (num_predict) unfinished
    "generation_timeout",         # no answer within the timeout derived from the budget
)

_CODE_BLOCK = re.compile(r"```[A-Za-z0-9_+-]*\s*\n(.*?)```", re.DOTALL)
_PH = re.compile(r"/\* RCAP_PH_[^*]+\*/")
# Any mention of the placeholder prefix, however it is written (`/*RCAP_PH_1*/`,
# `// RCAP_PH_1`, and without the underscore: `/* RCAP_PHname */`): outside the
# exact comment form it is still a leak.
_PH_ANY = re.compile(r"RCAP_PH[\w.]*")
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
    # completion | no_output | unparseable | shape_mismatch | placeholder_violation |
    # duplicated_block | limits_exceeded | backend_error | needs_cochange (repair requests only)
    outcome: str
    candidate: str | None = None
    diagnostics: list[str] = Field(default_factory=list)
    generation_config: dict[str, object] = Field(default_factory=dict)  # theta
    # Backend-reported token counts (section 15: Request/Input and Output
    # Tokens, measured separately). Recorded only when the runtime reports
    # them — never estimated. Empty for backends without a tokenizer.
    usage: dict[str, int] = Field(default_factory=dict)
    # The limits this request was generated (or refused) under: request size and
    # its bound, expected prompt / answer tokens, output budget, timeout, and
    # the runtime's timings when it reports them. Present whenever the backend
    # declares limits; for a unit sent as regions: {"regions": [one per region]}.
    limits: dict[str, object] = Field(default_factory=dict)
    # One of CAPACITY_KINDS when the outcome is a capacity refusal, else None.
    capacity: str | None = None
    # A whole-target unit answered region by region: one entry per region
    # request (lines, outcome, request hash, usage).
    regions: list[dict[str, object]] = Field(default_factory=list)
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
             allow_decline: bool = False,
             reject: Callable[[str], tuple[str, list[str]] | None] | None = None,
             parse_reference: str | None = None) -> GenerationRecord:
    """`parse_reference`: the text the answer replaces (the fork's function,
    in the form the request showed it). Only read for a language whose valid
    code may hold tree-sitter ERROR nodes (C, rcap.grammars): there an answer
    with a parse error is `unparseable` only if this reference parses clean,
    so the answer is never held to more than the code it was made from. For
    Java and Scala it changes nothing.

    `allow_decline` (repair requests): an answer `CANNOT_ADAPT: ...` instead
    of code is the distinct outcome `needs_cochange` -- a typed refusal, not an
    unparseable answer and not a completion. Off by default, so the ordinary
    pipeline's outcomes are unchanged.

    `reject(candidate)` is the caller's last acceptance check on an answer that
    passed everything else; it returns None, or (outcome, diagnostics) for a
    typed failure (the pipeline uses it for `duplicated_block`, which needs the
    fork and upstream texts this function does not have)."""
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
    # Explicit limits (rcap.backend_ollama.GenerationLimits): the budget of
    # this request. A backend without them keeps the request-size check only.
    declared = getattr(backend, "limits", None)
    plan: dict[str, object] = {}
    if declared is not None and hasattr(declared, "plan"):
        plan = declared.plan(len(request.prompt), getattr(request, "expected_answer_chars", None),
                             getattr(request, "language", None))
        if limit_chars:
            plan["request_limit_chars"] = limit_chars

    def record(outcome: str, candidate: str | None = None, *, diag: list[str] | None = None,
               capacity: str | None = None):
        return GenerationRecord(
            case_id=request.case_id, outcome=outcome, candidate=candidate,
            diagnostics=diag or [], generation_config=theta, usage=dict(usage),
            limits=dict(plan), capacity=capacity,
            request_hash=request.request_hash, exec_id=exec_id,
            executed_at=datetime.datetime.now(datetime.UTC)
            .isoformat(timespec="seconds"))

    # Section 11: "request or context exceeding backend limits" is a distinct
    # outcome, refused BEFORE invocation against the backend's own declared
    # acceptance limit — an oversized request silently truncated by the runtime
    # is a garbage completion, not a measurement.
    if limit_chars and len(request.prompt) > limit_chars:
        return record("limits_exceeded", capacity="request_too_large", diag=[
            (f"request is {len(request.prompt)} chars; backend declares an "
             f"acceptance limit of {limit_chars} chars")])
    # The window holds the prompt AND the answer. When the two cannot fit, the
    # request is refused here instead of being generated until the runtime
    # shifts the context (the answer is then written without the text it
    # answers, and with no cap it runs until the client times out).
    if plan.get("fits_window") is False:
        return record("limits_exceeded", capacity="answer_cannot_fit_window", diag=[
            (f"prompt (~{plan['expected_prompt_tokens']} tokens for {len(request.prompt)} chars) plus the "
             f"expected answer (~{plan['expected_output_tokens']} tokens for "
             f"{plan['expected_answer_chars']} chars) exceed the {plan['num_ctx']}-token window; "
             "not generated")])

    try:
        if plan and getattr(backend, "accepts_budget", False):
            raw = backend.generate(request, num_predict=plan["output_budget_tokens"],
                                   timeout=plan["timeout_s"])
        else:
            raw = backend.generate(request)
    except Exception as exc:  # noqa: BLE001 - any backend crash is a generation failure state
        if "timed out" in str(exc).lower():
            # the backend was too slow for this answer: capacity, not a model error
            return record("limits_exceeded", capacity="generation_timeout", diag=[str(exc)])
        return record("backend_error", diag=[str(exc)])
    reported = getattr(backend, "last_usage", None)
    if reported:
        usage.update({k: v for k, v in reported.items() if isinstance(v, int)})
    if plan:
        timing = getattr(backend, "last_timing", None)
        if timing:
            plan["timing"] = dict(timing)
        source = getattr(backend, "last_source", None)
        if source:
            plan["answer_source"] = source

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
            return record("limits_exceeded", capacity="input_truncated", diag=[
                (f"runtime evaluated {in_tok} prompt tokens for a "
                 f"{len(request.prompt)}-char prompt ({ratio:.2f} chars/token, "
                 f"window {num_ctx}"
                 + (", pinned at the truncation value" if pinned else "")
                 + "); the input was truncated by the runtime")])

    # The same limit on the way out. The window holds the prompt AND the answer:
    # when their sum passes it the runtime shifts the context while it is still
    # generating (it drops part of the prompt) and keeps going, so the answer
    # was written without the text it answers. Seen on whole-file units only
    # (15 of 1,110 stored unit candidates of the 477 run, no function unit);
    # e.g. a 997-line file came back 513 lines shorter.
    out_tok = usage.get("output_tokens", 0)
    if num_ctx and in_tok and out_tok and in_tok + out_tok > num_ctx:
        return record("limits_exceeded", capacity="window_overflow", diag=[
            (f"prompt ({in_tok} tokens) plus answer ({out_tok} tokens) exceed the {num_ctx}-token "
             "window; the runtime shifted the context while answering")])
    # The answer reached the output budget: it was cut, whatever it parses as.
    if getattr(backend, "last_done_reason", None) == "length" and plan:
        return record("limits_exceeded", capacity="output_budget_exhausted", diag=[
            (f"the answer reached the output budget of {plan['output_budget_tokens']} tokens "
             f"(expected ~{plan.get('expected_output_tokens')}) and was cut by the runtime")])

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
    wanted = getattr(request, "answer_kind", FUNCTION)
    if wanted == REGION:
        # One part of a file: its first line's indentation is part of the answer
        # (only blank lines around it are dropped), it is not parsed on its own
        # (the spliced file is, by generate_regions), and it must still begin and
        # end with the part's own first and last line.
        body = match.group(1) if match else raw
        lines = body.splitlines()
        while lines and not lines[0].strip():
            lines.pop(0)
        while lines and not lines[-1].strip():
            lines.pop()
        candidate = "".join(x.rstrip() + "\n" for x in lines)
        if not candidate:
            return record("no_output")
        # blank lines at the part's own edges are context like any other line: put them back
        part = (request.region or {}).get("text", "").splitlines()
        lead = next((n for n, x in enumerate(part) if x.strip()), 0)
        trail = next((n for n, x in enumerate(reversed(part)) if x.strip()), 0)
        candidate = "\n" * lead + candidate + "\n" * trail
        why = region_shape_error((request.region or {}).get("text", ""), candidate)
        if why:
            return record("shape_mismatch", diag=[f"the answer is not the adapted part: {why}"])
        if expected_placeholders is not None:
            report = placeholder_report(candidate, expected_placeholders)
            if report["missing"]:
                return record("placeholder_violation",
                              diag=[f"missing placeholders: {report['missing']}"])
            if report["duplicated"] or report["unknown"]:
                return record("placeholder_violation", diag=[
                    f"{kind} placeholders: {report[kind]}"
                    for kind in ("duplicated", "unknown") if report[kind]])
        return record("completion", candidate)
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
        excused = (tolerates_parse_errors(lang) and parse_reference is not None
                   and not parses(parse_reference))
        if not excused:
            diag = (f"candidate parses neither as a {lang} method nor as a "
                    "compilation unit")
            return record("unparseable", diag=[diag])

    # Answer shape. A request whose target is the fork's whole file (or a whole
    # type) is answered by the whole file (type). An answer that is one function
    # or an excerpt parses fine, and splicing it would replace the whole file
    # with it: a typed failure, never a completion. Function requests are not
    # checked (answer_kind defaults to "function").
    if wanted in WHOLE_KINDS:
        got = text_kind(candidate, lang)
        if not (got == FILE or (wanted == TYPE and got == TYPE)):
            return record("shape_mismatch", diag=[
                f"the target is a whole {'file' if wanted == FILE else 'type'} but the answer "
                f"is {'one function' if got == FUNCTION else 'not a complete ' + wanted} "
                f"({candidate.count(chr(10))} lines, kind {got})"])

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

    if reject is not None:
        refused = reject(candidate)
        if refused:
            return record(refused[0], diag=list(refused[1]))

    return record("completion", candidate)


def generate_regions(
    whole: AdaptationRequest, regions: list[AdaptationRequest], target: str, backend: Backend, *,
    expected_placeholders: set[str] | None = None,
    reject: Callable[[str], tuple[str, list[str]] | None] | None = None,
    placeholder_retry: bool = False,
) -> tuple[GenerationRecord, dict[str, object] | None]:
    """A whole-target unit answered region by region (rcap.regions).

    Each region request is generated on its own (with its own budget and, when
    `placeholder_retry`, its own targeted retry); the answers are spliced over
    their exact spans of `target` (rcap.splice) and the spliced text is then
    held to what a whole-target answer is held to: it parses when the target
    did, it has every placeholder exactly once, and `reject` accepts it. The
    returned record is the UNIT's: its candidate is the complete target text,
    so everything downstream is unchanged. The first region that fails decides
    the outcome and the remaining regions are not generated.

    Returns (record, placeholder-retry info of the last region retried or None)."""
    records: list[GenerationRecord] = []
    retry_info: dict[str, object] | None = None
    usage: dict[str, int] = {}
    for r in regions:
        text = (r.region or {}).get("text", "")
        exp = None if expected_placeholders is None else {ph for ph in expected_placeholders if ph in text}
        if placeholder_retry:
            rec, info = generate_with_placeholder_retry(r, backend, expected_placeholders=exp)
            retry_info = info or retry_info
        else:
            rec = generate(r, backend, expected_placeholders=exp)
        records.append(rec)
        for k, v in rec.usage.items():
            usage[k] = usage.get(k, 0) + v
        if rec.outcome != "completion":
            break
    first = records[0]
    combined_hash = hashlib.sha256(
        ("regions:" + "\n".join(r.request_hash for r in regions)).encode()).hexdigest()

    def record(outcome: str, candidate: str | None = None, diag: list[str] | None = None,
               capacity: str | None = None) -> GenerationRecord:
        return GenerationRecord(
            case_id=whole.case_id, outcome=outcome, candidate=candidate, diagnostics=diag or [],
            generation_config=first.generation_config, usage=dict(usage),
            limits={"regions": [rec.limits for rec in records]}, capacity=capacity,
            regions=[{"index": n + 1, "of": len(regions), "lines": (rq.region or {}).get("lines"),
                      "outcome": rec.outcome, "request_hash": rq.request_hash,
                      "template_id": rq.template_id, "prompt_chars": len(rq.prompt),
                      "usage": dict(rec.usage), "capacity": rec.capacity}
                     for n, (rq, rec) in enumerate(zip(regions, records))],
            request_hash=combined_hash,
            exec_id=hashlib.sha256(
                f"{backend.name}:{backend.model}:{combined_hash}".encode()).hexdigest()[:16],
            executed_at=records[-1].executed_at)

    bad = records[-1]
    if bad.outcome != "completion":
        return record(bad.outcome, diag=[f"region {len(records)} of {len(regions)}: {d}"
                                         for d in bad.diagnostics] or
                      [f"region {len(records)} of {len(regions)}: {bad.outcome}"],
                      capacity=bad.capacity), retry_info
    from rcap.regions import Region

    spans = [Region(rq.region["start"], rq.region["end"], rq.region["lines"][0], rq.region["lines"][1],
                    rq.region["text"]) for rq in regions]
    merged, why = merge_regions(target, spans, [rec.candidate or "" for rec in records])
    if merged is None:
        return record("unparseable", diag=["the region answers could not be spliced: " + why]), retry_info
    lang = whole.language or "java"
    grammar = grammar_for(lang)

    def parses(text: str) -> bool:
        tree = parse(text, grammar) if grammar is not None else None
        return tree is not None and not tree.root_node.has_error
    if grammar is not None and parses(target) and not parses(merged):
        return record("unparseable", diag=[
            f"the target with the adapted part(s) spliced in no longer parses as {lang}"]), retry_info
    if expected_placeholders is not None:
        report = placeholder_report(merged, expected_placeholders)
        if report["missing"] or report["duplicated"] or report["unknown"]:
            return record("placeholder_violation", diag=[
                f"{kind} placeholders: {report[kind]}"
                for kind in ("missing", "duplicated", "unknown") if report[kind]]), retry_info
    if reject is not None:
        refused = reject(merged)
        if refused:
            return record(refused[0], diag=list(refused[1])), retry_info
    return record("completion", merged), retry_info


def generate_with_placeholder_retry(
    request: AdaptationRequest, backend: Backend, *,
    expected_placeholders: set[str] | None = None,
    reject: Callable[[str], tuple[str, list[str]] | None] | None = None,
) -> tuple[GenerationRecord, dict[str, object] | None]:
    """`generate`, plus ONE targeted retry when the answer is rejected only for
    its placeholders.

    The retry is a new single-turn request: the original request followed by
    the exact list of what was wrong (missing / duplicated / unknown) and the
    rule (each original placeholder exactly once, no new one). It is issued
    once; if its answer is anything but a completion the ORIGINAL refusal is
    returned unchanged. The retry's answer passes the same checks as any other
    (placeholders of any spelling, and `reject`, i.e. no duplicated block): both
    wrong passes the first version of the retry let through (PR-14026
    BatchBuilder, PR-13212 OAuthBearerValidationUtils) wrote the changed loop
    in front of the loop it was to change. Returns (record, retry info or None when no retry was
    issued). Deterministic given a deterministic backend."""
    from rcap.request import synthesize_placeholder_repair

    first = generate(request, backend, expected_placeholders=expected_placeholders, reject=reject)
    if first.outcome != "placeholder_violation" or expected_placeholders is None:
        return first, None
    report = _report_from_diagnostics(first.diagnostics, expected_placeholders)
    repair = synthesize_placeholder_repair(request, sorted(expected_placeholders), report)
    second = generate(repair, backend, expected_placeholders=expected_placeholders, reject=reject)
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
