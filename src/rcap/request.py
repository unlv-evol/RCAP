"""Adaptation Request Synthesis (RCAP ref section 10).

Renders the backend-independent Adaptation Context into the reference LLM
request: a systematically constructed prompt separating objective, source
transformation, target context, evidence-derived constraints, and expected
output. Changes representation, never information — every case-specific line
traces to the context; nothing is introduced (I2), and no ground truth can
appear because none exists upstream (I13). Identical (context, instructions)
yield an identical request and hash.
"""

from __future__ import annotations

import hashlib

from pydantic import BaseModel

from rcap.context import AdaptationContext

TEMPLATE_ID = "rcap-request-v1"
TEMPLATE_VERSION = "1"

INSTRUCTIONS = (
    "You are adapting a source-side code change into a diverged fork.\n"
    "Below is the change as a before/after pair from the source project, the\n"
    "corresponding target function from the fork, and constraints derived from\n"
    "recovered evidence. Apply the same change to the target function,\n"
    "respecting every constraint. Placeholder comments of the form\n"
    "/* RCAP_PH_... */ stand for elided code: keep each one exactly where it\n"
    "is, character for character, and do not invent code for it.\n"
    "Return ONLY the complete adapted target function in one ```{lang} code\n"
    "block, with no commentary."
)


# Re-prompt (repair) template. Used only when the context carries repair
# feedback (rcap.reprompt.RepairContext); a plain AdaptationContext never
# reaches it, so the default request stays byte-identical.
#
# v2 (round 2). v1 told the model to "use only what the fork demonstrably has"
# AND to "keep applying the source change": for a change whose new names live
# in a co-change, the only answer satisfying both is the fork's original
# function, and that is what it returned. v2 removes the contradiction (names
# from APPLIED co-changes are allowed and listed), makes the previous attempt
# the thing to fix and the original a labelled reference, gives a way to
# decline (CANNOT_ADAPT), and repeats the no-revert rule where the answer is
# specified.
REPAIR_TEMPLATE_ID = "rcap-reprompt-v2"
REPAIR_TEMPLATE_VERSION = "2"
DECLINE_TOKEN = "CANNOT_ADAPT"

REPAIR_INSTRUCTIONS = (
    "You are fixing a previous attempt at porting a source-side code change\n"
    "into a diverged fork. The port is done one function at a time; the other\n"
    "parts of the same upstream change (its co-changes) are ported separately.\n"
    "Below: the change as a before/after pair from the source project; the\n"
    "fork's ORIGINAL function from before the port (reference only); the\n"
    "co-changes and whether each one is applied in the fork for this build;\n"
    "the PREVIOUS ATTEMPT; and the diagnostics it produced when it was built\n"
    "and tested inside the fork.\n"
    "The previous attempt is the code to fix: keep the source change applied\n"
    "and make the smallest edit to the previous attempt that resolves the\n"
    "diagnostics.\n"
    "You may use whatever the fork already has (the original function, the\n"
    "fork-side context and the diagnostics show what exists) and the names\n"
    "introduced by co-changes marked APPLIED. Do not use names that exist only\n"
    "in the source project or in co-changes marked NOT APPLIED; where the fork\n"
    "names something differently, use the fork's name. Do not invent APIs.\n"
    "Do NOT resolve the diagnostics by undoing the change: an answer equal to\n"
    "the ORIGINAL function is rejected automatically, and so is one that\n"
    "deletes logic, tests or assertions.\n"
    "If no edit to this one function can work, because it depends on something\n"
    "that is neither in the fork nor in an APPLIED co-change (a declaration it\n"
    "must call, or, for a test, the behaviour it checks), do not return code and\n"
    "do not return the original. Answer with this single line instead:\n"
    "{decline}: needs <symbol or change> from <file>\n"
    "Placeholder comments of the form /* RCAP_PH_... */ stand for elided code:\n"
    "keep each one exactly where it is, character for character, and do not\n"
    "invent code for it.\n"
    "Otherwise return ONLY the complete fixed function in one ```{lang} code\n"
    "block, with no commentary."
)

_FEEDBACK_HEADINGS = {
    "build": "## Compiler diagnostics for the previous attempt",
    "test": "## Test failures with the previous attempt applied",
    "echo": "## Problem with the previous attempt",
}
MAX_CHANGE_DIFF_LINES = 40


def _change_diff(before: str, after: str) -> str:
    """The source change as a compact diff: restated in every repair round
    so the thing being ported stays in view (UTFix, OOPSLA'25)."""
    import difflib
    import re
    # Placeholder ids differ between the two sides by construction
    # (source.before.N / source.after.N); that is not part of the change.
    ph = re.compile(r"/\* RCAP_PH_source\.(?:before|after)\.(\d+) \*/")
    lines = [ln.rstrip("\n") for ln in difflib.unified_diff(
        ph.sub("/* elided */", before).splitlines(),
        ph.sub("/* elided */", after).splitlines(), lineterm="", n=1)][2:]
    if len(lines) > MAX_CHANGE_DIFF_LINES:
        omitted = len(lines) - MAX_CHANGE_DIFF_LINES
        lines = lines[:MAX_CHANGE_DIFF_LINES] + [f"[... {omitted} more diff lines omitted]"]
    return "\n".join(lines)


class AdaptationRequest(BaseModel):
    case_id: str
    # Recorded so the response is parsed with the grammar the request was
    # written in, instead of a language assumed at the call site.
    language: str = "java"
    template_id: str = TEMPLATE_ID
    template_version: str = TEMPLATE_VERSION
    prompt: str
    request_hash: str


def synthesize(context: AdaptationContext) -> AdaptationRequest:
    # Every fence is labelled with the case's own language: a Scala function
    # fenced as ```java misdescribes the input and invites a Java-shaped answer.
    lang = context.language or "java"
    # Repair feedback rides in the context (never injected into the prompt
    # afterwards), so every feedback line stays traceable (section 17).
    feedback = getattr(context, "feedback", None)
    sections = [
        "## Task\n" + (INSTRUCTIONS.format(lang=lang) if feedback is None else
                       REPAIR_INSTRUCTIONS.format(lang=lang, decline=DECLINE_TOKEN)),
        f"## Source function BEFORE the change\n```{lang}\n"
        + context.transformation["source.before"] + "```",
        f"## Source function AFTER the change\n```{lang}\n"
        + context.transformation["source.after"] + "```",
    ]
    if feedback is None:
        sections.append(f"## Target function in the fork (adapt this)\n```{lang}\n"
                        + context.transformation["target"] + "```")
    else:
        diff = _change_diff(context.transformation["source.before"],
                            context.transformation["source.after"])
        if diff:
            sections.append("## The change to port (diff of BEFORE -> AFTER)\n```diff\n"
                            + diff + "\n```")
        sections.append(
            "## ORIGINAL function in the fork, before the port (reference only; "
            f"returning it unchanged is NOT an acceptable answer)\n```{lang}\n"
            + context.transformation["target"] + "```")
    if context.constraints:
        lines = [f"- [{c.kind}] {c.detail}" + (" (BLOCKING)" if c.blocking else "")
                 for c in context.constraints]
        sections.append("## Constraints from recovered evidence\n" + "\n".join(lines))
    if context.siblings:
        # Section 13: sibling-edit signatures, so a rename or signature change
        # made in another unit of the same change stays consistent here.
        lines = []
        for s in context.siblings:
            line = f"- {s.entity}: `{s.signature_after or s.target_signature or '?'}`"
            if s.signature_before and s.signature_after \
                    and s.signature_before != s.signature_after:
                line += f" (changed from `{s.signature_before}`)"
            lines.append(line)
        sections.append(
            "## Other functions edited by this same change (signatures only)\n"
            "This change also edits the functions below; keep any shared names\n"
            "and signatures consistent with these edits.\n" + "\n".join(lines))
    helper_arts = [a for a in context.program_context
                   if a.role == "target"
                   and a.entity != context.target_localization.get("function")]
    if helper_arts:
        # Section 9(5): retained helper context (reduced, recovery maps kept in
        # the context) — supporting material, not something to adapt.
        blocks = [f"### {a.entity} (do not modify)\n```{lang}\n{a.reduced_text}```"
                  for a in helper_arts]
        sections.append("## Related fork context (reduced helpers)\n"
                        + "\n\n".join(blocks))
    if context.target_localization.get("file"):
        lines = [f"File: {context.target_localization['file']}"]
        alternatives = context.target_localization.get("alternatives") or []
        if alternatives:
            # Section 9(2): ambiguity stays visible, never silently collapsed.
            lines.append(f"NOTE: localization is ambiguous — "
                         f"{len(alternatives)} plausible target functions were "
                         f"recorded; the one above is the primary candidate.")
        sections.append("## Target location\n" + "\n".join(lines))
    if feedback is not None:
        # Co-changes: what else the same upstream change does, and whether it
        # is in the fork for this build. This is what lifts the "only what the
        # fork has" rule for names the change itself introduces.
        if feedback.cochanges:
            lines = [f"- [{c.status}] {c.path}: {c.summary}" for c in feedback.cochanges]
            if feedback.cochanges_omitted:
                lines.append(f"({feedback.cochanges_omitted} more co-change hunks not listed: "
                             "none of them is applied)")
            if feedback.allowed_symbols:
                lines.append("Names introduced by APPLIED co-changes (you may use these): "
                             + ", ".join(feedback.allowed_symbols))
            sections.append("## Co-changes of the same upstream change\n" + "\n".join(lines))
        elif feedback.cochanges_omitted:
            sections.append("## Co-changes of the same upstream change\n"
                            f"None applied ({feedback.cochanges_omitted} co-change hunks of the "
                            "upstream change are not in the fork and are not listed).")
        elif feedback.pr_context:
            sections.append("## Co-changes of the same upstream change\n"
                            "None: the upstream change touches nothing outside this "
                            "function.")
        else:
            sections.append("## Co-changes of the same upstream change\n"
                            "None applied: only this function is changed in the fork "
                            "for this build.")
        if feedback.fork_context:
            sections.append("## Fork-side context (what the fork declares; do not modify)\n"
                            + "\n".join(feedback.fork_context))
        previous = feedback.previous_candidate
        sections.append(
            f"## PREVIOUS ATTEMPT (attempt "
            f"{feedback.previous_attempt or feedback.attempt - 1}): the code to fix\n"
            f"```{lang}\n" + previous + ("" if previous.endswith("\n") else "\n")
            + "```")
        body = feedback.error_text.rstrip("\n")
        if feedback.truncated:
            body += ("\n[diagnostics truncated to fit the request limit: "
                     f"{len(feedback.error_text)} of {feedback.full_chars} chars shown]")
        for note in feedback.notes:
            body += "\n" + note
        sections.append(_FEEDBACK_HEADINGS.get(feedback.kind, _FEEDBACK_HEADINGS["build"])
                        + "\n" + body)
        sections.append(
            f"## Output\nEither one ```{lang} block containing the complete fixed function: "
            "the previous attempt with the diagnostics resolved and the source change still "
            "applied. Returning the ORIGINAL function unchanged is rejected.\n"
            "Or, if this function cannot be fixed because it depends on something that is in "
            "neither the fork nor an APPLIED co-change, exactly one line:\n"
            f"{DECLINE_TOKEN}: needs <symbol or change> from <file>\n"
            "Nothing else.")
    else:
        sections.append(f"## Output\nOne ```{lang} block containing the full adapted "
                        "target function. Nothing else.")

    prompt = "\n\n".join(sections) + "\n"
    if feedback is None:
        request_hash = hashlib.sha256(
            f"{TEMPLATE_ID}:{TEMPLATE_VERSION}\n{prompt}".encode()).hexdigest()
        return AdaptationRequest(case_id=context.case_id, language=lang, prompt=prompt,
                                 request_hash=request_hash)
    # The attempt index is part of the hash, so every attempt of a case has a
    # distinct request_hash (and hence exec_id) even if two prompts coincide.
    request_hash = hashlib.sha256(
        f"{REPAIR_TEMPLATE_ID}:{REPAIR_TEMPLATE_VERSION}:attempt={feedback.attempt}\n"
        f"{prompt}".encode()).hexdigest()
    return AdaptationRequest(case_id=context.case_id, language=lang,
                             template_id=REPAIR_TEMPLATE_ID,
                             template_version=REPAIR_TEMPLATE_VERSION,
                             prompt=prompt, request_hash=request_hash)


def untraceable_lines(prompt: str, context: AdaptationContext) -> list[str]:
    """Section 17 mapping test: every request fact must originate in the
    context. Synthesis is deterministic (R = Psi(C, I)), so the reference
    rendering of the context is the complete set of legitimate lines; any
    non-empty line of the given prompt outside it has no context origin —
    invented evidence."""
    legitimate = set(synthesize(context).prompt.splitlines())
    return [line for line in prompt.splitlines()
            if line.strip() and line not in legitimate]
