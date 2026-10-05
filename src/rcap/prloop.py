"""PR-level re-prompt loop (round 2).

Round 1 validated one function at a time and found that most failing
candidates cannot be fixed by editing that one function: the fix needs a
co-change of the same upstream change. Here the unit of VALIDATION is the
whole upstream change:

1. every processing unit of every SAP of the PR that produced a candidate is
   applied to the fork together, then built and tested once;
2. each diagnostic is charged to the unit it belongs to (the driver does the
   attribution and hands back per-unit diagnostics);
3. only charged units are re-prompted, each with its own fresh single-turn
   repair request, and the combined state is validated again;
4. a unit that cannot make progress (k attempts used, a repeated candidate,
   an echo, a typed refusal, a co-change that is not available) is DROPPED:
   reverted to the fork's code, so the build can go on judging the others.
   Dropping is recorded with its reason and counts as that unit's failure.

The loop ends when a validation charges nothing to anyone (`settled`), or
when the build still fails but no diagnostic can be charged (`unattributed`).

Attribution honesty: a unit that ends VALID here is valid IN PR CONTEXT —
with its siblings' candidates applied next to it — which is a different claim
from round 1's standalone VALID. The record keeps what was applied alongside,
what was missing from the start and what was dropped on the way.

Like `rcap.reprompt`, this module knows nothing about build systems.
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from pydantic import BaseModel, Field

from rcap.context import AdaptationContext
from rcap.generate import (
    Backend,
    GenerationRecord,
    effective_limit_chars,
    generate,
    has_placeholder_leak,
)
from rcap.reprompt import (
    _STOP_FOR_OUTCOME,
    DEFAULT_K,
    ECHO_RETRY_NOTE,
    RepairFeedback,
    _norm,
    code_equal,
    fit_feedback,
    is_echo,
    measure,
    parse_decline,
    shrink_reasons,
)
from rcap.request import AdaptationRequest

VALID_PR = "VALID_PR"   # valid with the PR's other applied changes next to it


class UnitDiagnostic(BaseModel):
    """What one validation charges to one unit."""

    kind: str                                  # build | test
    text: str                                  # rendered, as the request shows it
    summary: str = ""
    missing: list[dict[str, object]] = Field(default_factory=list)   # MissingSymbol dicts
    fork_context: list[str] = Field(default_factory=list)
    how: list[str] = Field(default_factory=list)                     # span | name | file | ...


class PRValidation(BaseModel):
    """Outcome of validating one combined state, returned by the driver."""

    verdict: str                               # VALID | BUILD_FAIL | TEST_FAIL | INCONCLUSIVE
    unit_verdicts: dict[str, str] = Field(default_factory=dict)      # every applied unit
    unit_diags: dict[str, UnitDiagnostic] = Field(default_factory=dict)
    mech_diags: dict[str, str] = Field(default_factory=dict)         # mechanical hunk -> why
    # Requested changes that could not be written into the tree. The validation
    # ran WITHOUT them, so its verdict already describes the state without them.
    not_spliced: dict[str, str] = Field(default_factory=dict)        # unit -> why
    mech_not_applied: dict[str, str] = Field(default_factory=dict)   # mechanical hunk -> why
    unattributed: list[str] = Field(default_factory=list)
    summary: str = ""
    seconds: float = 0.0
    detail: dict[str, object] = Field(default_factory=dict)


@dataclass
class PRUnit:
    """One processing unit of one SAP of the PR, with its attempt-1 artifacts."""

    key: str                                   # "<sap>#<unit index>"
    sap: str
    file: str                                  # fork-relative target file
    index: int
    entity: str | None
    context: AdaptationContext
    request: AdaptationRequest
    generation: GenerationRecord
    target_original: str
    expand: Callable[[str], str]
    expected_placeholders: set[str] | None = None
    is_test: bool = False
    # loop state
    raw: str | None = None
    expanded: str | None = None
    first_expanded: str | None = None
    status: str = "active"                     # active | dropped | not_applied
    reason: str | None = None
    reason_detail: dict[str, object] = field(default_factory=dict)
    attempts: int = 1
    current_attempt: int = 1                   # attempt that produced the applied candidate
    echoes: int = 0
    rows: list[dict[str, object]] = field(default_factory=list)
    validated: dict[tuple[str, str], int] = field(default_factory=dict)
    prompts: set[str] = field(default_factory=set)
    last_verdict: str | None = None
    round1_verdict: str | None = None


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _attempt_row(u: PRUnit, attempt: int, rnd: int, request: AdaptationRequest,
                 gen: GenerationRecord, **extra: object) -> dict[str, object]:
    row: dict[str, object] = {
        "attempt": attempt, "round": rnd, "outcome": gen.outcome,
        "template_id": request.template_id, "prompt_chars": len(request.prompt),
        "prompt_sha256": _sha(request.prompt), "request_hash": request.request_hash,
        "exec_id": gen.exec_id, "usage": dict(gen.usage),
        "diagnostics": list(gen.diagnostics)[:3], "verdicts": [],
    }
    row.update(extra)
    u.rows.append(row)
    return row


def run_pr_loop(
    units: Sequence[PRUnit], backend: Backend,
    validate: Callable[[int, dict[str, str], list[str]], PRValidation],
    *, k: int = DEFAULT_K, mech: Sequence[str] = (),
    precheck: Callable[[PRUnit, UnitDiagnostic, set[str], list[str]],
                       list[dict[str, object]]] | None = None,
    repair_extras: Callable[[PRUnit, UnitDiagnostic, set[str], list[str]],
                            dict[str, object]] | None = None,
    on_attempt: Callable[[PRUnit, int, dict[str, object]], None] | None = None,
    on_round: Callable[[int, dict[str, object]], None] | None = None,
    max_rounds: int = 60,
) -> dict[str, object]:
    """Run the PR-level loop and return its record.

    `validate(round, {unit key: expanded candidate}, [mechanical hunk ids])`
    applies exactly that state, builds, tests, attributes and restores.
    `precheck(unit, diagnostic, applied unit keys, mech ids)` returns the
    evidence for every missing name whose co-change is NOT available (a
    non-empty answer drops the unit as `needs_cochange` with no model call).
    `repair_extras(...)` returns the co-change view for the repair request.
    `mech` are mechanically applied hunks of the upstream change: they take
    part in validation, cannot be re-prompted, and are dropped when a
    diagnostic is charged to them.
    """
    by_key = {u.key: u for u in units}
    limit = None
    for u in units:
        u.raw = u.generation.candidate
        u.expanded = u.expand(u.raw or "")
        u.first_expanded = u.expanded
        row = _attempt_row(u, 1, 1, u.request, u.generation)
        if on_attempt:
            on_attempt(u, 1, {"prompt": u.request.prompt, "candidate_raw": u.raw,
                              "candidate_expanded": u.expanded})
        if has_placeholder_leak(u.expanded):
            # A placeholder marker survives expansion: it would be written into
            # the fork as a stray comment. Never applied, and never an "echo".
            u.status, u.reason = "not_applied", "placeholder_leak"
            row["placeholder_leak"] = True
            row["verdicts"].append({"round": 1, "verdict": "PLACEHOLDER_LEAK"})
            u.last_verdict = u.round1_verdict = "PLACEHOLDER_LEAK"
        elif is_echo(u.expanded, u.target_original):
            # Nothing to apply: the attempt-1 answer is the fork's own code.
            u.status, u.reason = "not_applied", "echo_attempt1"
            row["echo"] = True
            row["verdicts"].append({"round": 1, "verdict": "ECHO"})
            u.last_verdict = u.round1_verdict = "ECHO"

    mech_active = list(mech)
    mech_state: dict[str, dict[str, object]] = {m: {"status": "active"} for m in mech}
    rounds: list[dict[str, object]] = []
    stop: str | None = None
    last: PRValidation | None = None

    def active() -> list[PRUnit]:
        return [u for u in units if u.status == "active"]

    def others_sig(u: PRUnit) -> str:
        parts = sorted(f"{o.key}={_norm(o.expanded)}" for o in active() if o is not u)
        return _sha("\n".join(parts + sorted(mech_active)))

    def drop(u: PRUnit, reason: str, rnd: int, **detail: object) -> None:
        u.status, u.reason, u.reason_detail = "dropped", reason, dict(detail)
        u.reason_detail["round"] = rnd

    for rnd in range(1, max_rounds + 1):
        applied = {u.key: u.expanded or "" for u in active()}
        for u in active():
            u.validated.setdefault((_norm(u.expanded), others_sig(u)), u.current_attempt)
        last = validate(rnd, applied, list(mech_active))
        row: dict[str, object] = {
            "round": rnd, "applied_units": sorted(applied), "mech_active": list(mech_active),
            "verdict": last.verdict, "summary": last.summary, "seconds": last.seconds,
            "unit_verdicts": dict(last.unit_verdicts),
            "charged": {key: {"kind": d.kind, "summary": d.summary, "how": d.how}
                        for key, d in last.unit_diags.items()},
            "mech_charged": dict(last.mech_diags), "not_spliced": dict(last.not_spliced),
            "mech_not_applied": dict(last.mech_not_applied),
            "unattributed": list(last.unattributed), "detail": last.detail, "actions": [],
        }
        rounds.append(row)
        actions: list[dict[str, object]] = row["actions"]  # type: ignore[assignment]
        for u in active():
            verdict = last.unit_verdicts.get(u.key, "BLOCKED")
            if u.key in last.not_spliced:
                verdict = "INTEGRATION_FAIL"
            u.last_verdict = verdict
            if rnd == 1:
                u.round1_verdict = verdict
            u.rows[-1]["verdicts"].append({"round": rnd, "verdict": verdict})  # type: ignore[union-attr]
            if u.key in last.unit_diags:
                u.rows[-1]["error_summary"] = last.unit_diags[u.key].summary

        for key, why in last.not_spliced.items():
            if key in by_key and by_key[key].status == "active":
                drop(by_key[key], "integration_fail", rnd, why=why)
                actions.append({"unit": key, "action": "dropped", "reason": "integration_fail"})
        for m, why in last.mech_not_applied.items():
            if m in mech_active:
                mech_active.remove(m)
                mech_state[m] = {"status": "dropped", "round": rnd, "reason": why}
                actions.append({"mech": m, "action": "dropped", "reason": why[:200]})
        failing = [u for u in active() if u.key in last.unit_diags]
        failing_mech = [m for m in mech_active if m in last.mech_diags]

        if not failing and not failing_mech:
            if last.verdict in ("BUILD_FAIL", "TEST_FAIL"):
                if mech_active:
                    # Nothing chargeable, and mechanical hunks are the one
                    # thing nobody ported deliberately: take them all out.
                    for m in mech_active:
                        mech_state[m] = {"status": "dropped", "round": rnd,
                                         "reason": "unattributed failure"}
                    actions.append({"mech": list(mech_active), "action": "dropped",
                                    "reason": "unattributed failure"})
                    mech_active = []
                    if on_round:
                        on_round(rnd, row)
                    continue
                stop = "unattributed"
            else:
                stop = "settled"
            if on_round:
                on_round(rnd, row)
            break

        for m in failing_mech:
            mech_active.remove(m)
            mech_state[m] = {"status": "dropped", "round": rnd, "reason": last.mech_diags[m]}
            actions.append({"mech": m, "action": "dropped", "reason": last.mech_diags[m][:200]})

        applied_keys = set(applied) - set(last.not_spliced)
        reprompted: list[PRUnit] = []
        for u in failing:
            diag = last.unit_diags[u.key]
            if u.attempts >= k:
                drop(u, "k_exhausted", rnd)
                actions.append({"unit": u.key, "action": "dropped", "reason": "k_exhausted"})
                continue
            if precheck is not None and diag.kind == "build":
                unavailable = precheck(u, diag, applied_keys, list(mech_active))
                if unavailable:
                    drop(u, "needs_cochange", rnd, source="precheck", evidence=unavailable)
                    actions.append({"unit": u.key, "action": "dropped",
                                    "reason": "needs_cochange", "source": "precheck",
                                    "evidence": unavailable})
                    continue
            basis_raw, basis_attempt = u.raw or "", u.current_attempt
            notes: list[str] = []
            for retry in (0, 1):
                if u.attempts >= k:
                    drop(u, "k_exhausted", rnd)
                    actions.append({"unit": u.key, "action": "dropped", "reason": "k_exhausted"})
                    break
                extras = (repair_extras(u, diag, applied_keys, list(mech_active))
                          if repair_extras else {})
                if limit is None:
                    limit = effective_limit_chars(backend, u.request.language)
                fb = RepairFeedback(
                    attempt=u.attempts + 1, kind=diag.kind, previous_candidate=basis_raw,
                    previous_attempt=basis_attempt, error_text=diag.text, notes=notes,
                    fork_context=list(diag.fork_context),
                    # only a driver that supplies the co-change view may let the
                    # request say what the upstream change does outside the unit
                    pr_context=repair_extras is not None, **extras)
                ctx, request = fit_feedback(u.context, fb, limit)
                prompt_sha = _sha(request.prompt)
                if prompt_sha in u.prompts:
                    # Deterministic backend: the same request gives the same
                    # answer, which was already tried.
                    drop(u, "cycle", rnd, detail="identical repair request already issued")
                    actions.append({"unit": u.key, "action": "dropped", "reason": "cycle",
                                    "detail": "identical repair request"})
                    break
                u.prompts.add(prompt_sha)
                t0 = time.time()
                gen = generate(request, backend, expected_placeholders=u.expected_placeholders,
                               allow_decline=True)
                u.attempts += 1
                arow = _attempt_row(
                    u, u.attempts, rnd + 1, request, gen,
                    gen_seconds=round(time.time() - t0, 1), feedback_kind=diag.kind,
                    feedback_truncated=ctx.feedback.truncated,
                    context_truncated=list(ctx.feedback.context_truncated),
                    feedback_chars=[len(ctx.feedback.error_text), ctx.feedback.full_chars],
                    cochanges=[f"{c.status}|{c.hunk_id}" for c in ctx.feedback.cochanges],
                    echo_retry=bool(notes))
                expanded = u.expand(gen.candidate) if gen.outcome == "completion" else None
                if on_attempt:
                    on_attempt(u, u.attempts, {
                        "prompt": request.prompt, "candidate_raw": gen.candidate,
                        "candidate_expanded": expanded, "outcome": gen.outcome,
                        "diagnostics": list(gen.diagnostics)})
                if gen.outcome == "needs_cochange":
                    decline = parse_decline(gen.diagnostics)
                    arow["verdicts"].append({"round": rnd, "verdict": "DECLINED"})  # type: ignore[union-attr]
                    drop(u, "needs_cochange", rnd, source="model", evidence=[decline])
                    actions.append({"unit": u.key, "action": "dropped",
                                    "reason": "needs_cochange", "source": "model",
                                    "evidence": [decline]})
                    break
                if gen.outcome != "completion":
                    reason = _STOP_FOR_OUTCOME.get(gen.outcome, "unparseable")
                    arow["verdicts"].append({"round": rnd, "verdict": "NO_CANDIDATE"})  # type: ignore[union-attr]
                    drop(u, reason, rnd, outcome=gen.outcome)
                    actions.append({"unit": u.key, "action": "dropped", "reason": reason})
                    break
                assert expanded is not None
                if has_placeholder_leak(expanded):
                    arow["placeholder_leak"] = True
                    arow["verdicts"].append({"round": rnd, "verdict": "PLACEHOLDER_LEAK"})  # type: ignore[union-attr]
                    drop(u, "placeholder_leak", rnd)
                    actions.append({"unit": u.key, "action": "dropped",
                                    "reason": "placeholder_leak"})
                    break
                if is_echo(expanded, u.target_original):
                    u.echoes += 1
                    arow["echo"] = True
                    arow["verdicts"].append({"round": rnd, "verdict": "ECHO"})  # type: ignore[union-attr]
                    if retry == 0:
                        # (f) keep the diagnostics: restate the failing
                        # candidate and say the revert was rejected.
                        notes = [ECHO_RETRY_NOTE]
                        continue
                    drop(u, "echo", rnd)
                    actions.append({"unit": u.key, "action": "dropped", "reason": "echo"})
                    break
                u.raw, u.expanded, u.current_attempt = gen.candidate, expanded, u.attempts
                reprompted.append(u)
                actions.append({"unit": u.key, "action": "reprompted", "attempt": u.attempts})
                break

        # A candidate this unit already had validated in this same surrounding
        # state fails the same way: stop it instead of rebuilding. Dropping one
        # unit changes the state the others are in, so iterate to a fixpoint.
        changed = True
        while changed:
            changed = False
            for u in reprompted:
                if u.status != "active":
                    continue
                seen = u.validated.get((_norm(u.expanded), others_sig(u)))
                if seen is not None:
                    u.rows[-1]["verdicts"].append({"round": rnd, "verdict": "CYCLE"})  # type: ignore[union-attr]
                    u.rows[-1]["cycle_of_attempt"] = seen
                    drop(u, "cycle", rnd, cycle_of_attempt=seen)
                    actions.append({"unit": u.key, "action": "dropped", "reason": "cycle",
                                    "cycle_of_attempt": seen})
                    changed = True
        if on_round:
            on_round(rnd, row)
    else:
        stop = "max_rounds"

    # ---- per-unit outcome
    out_units: dict[str, dict[str, object]] = {}
    for u in units:
        final = u.last_verdict
        in_final = u.status == "active"
        vs1 = measure(u.expanded or "", u.first_expanded or "")
        reasons = shrink_reasons(vs1) if in_final else []
        if in_final and final == "VALID":
            final = VALID_PR
        if in_final:
            stop_reason = {VALID_PR: "success", "INCONCLUSIVE": "inconclusive"}.get(
                final or "", "blocked" if stop == "unattributed" else (stop or "max_rounds"))
        else:
            stop_reason = u.reason
        out_units[u.key] = {
            "sap": u.sap, "file": u.file, "unit": u.index, "entity": u.entity,
            "is_test": u.is_test, "attempts": u.rows, "attempts_run": u.attempts,
            "status": u.status, "in_final_state": in_final,
            "attempt1_verdict": u.round1_verdict, "final_verdict": final,
            "stop_reason": stop_reason, "stop_detail": u.reason_detail,
            "first_success_attempt": u.current_attempt if final == VALID_PR else None,
            "applied_attempt": u.current_attempt if in_final else None,
            "echo": u.echoes > 0, "echoes": u.echoes,
            "suspicious_shrink": bool(reasons), "shrink_reasons": reasons,
            "success_class": (("suspicious_shrink" if reasons else "clean")
                              if final == VALID_PR else None),
            "code_equal_to_target": code_equal(u.expanded or "", u.target_original),
            "metrics": {"vs_attempt1": vs1,
                        "vs_target_original": measure(u.expanded or "", u.target_original)},
        }
    return {
        "k": k, "rounds": rounds, "rounds_run": len(rounds), "stop_reason": stop,
        "attempt1_verdict": rounds[0]["verdict"] if rounds else None,
        "final_verdict": last.verdict if last is not None else None,
        "final_applied_units": sorted(u.key for u in units if u.status == "active"),
        "final_mech": list(mech_active), "mech": mech_state, "units": out_units,
    }
