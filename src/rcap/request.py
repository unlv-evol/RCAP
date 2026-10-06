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
from rcap.unitkind import FILE, FUNCTION, REGION, WHOLE_KINDS, text_kind

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


# Whole-target units (rcap.unitkind): the target payload is the fork's whole
# file (or a whole type), not a function. They used to be sent with the function
# wording above; asked for "the complete adapted target function" next to a
# source function, the model returned that one function and the splice replaced
# the whole file with it. A unit whose target is not a whole file / type never
# reaches these templates, so function units keep the request above byte for byte.
WHOLE_TEMPLATE_ID = "rcap-request-whole-v1"
WHOLE_TEMPLATE_VERSION = "1"

_WHOLE_TAIL = (
    "Change only what the source change changes and keep everything else of\n"
    "the fork's {noun} as it is. Placeholder comments of the form\n"
    "/* RCAP_PH_... */ stand for elided code: keep each one exactly where it\n"
    "is, character for character, and do not invent code for it.\n"
    "Return ONLY the complete adapted target {noun}, all of it from its first\n"
    "line to its last, in one ```{lang} code block, with no commentary. Do not\n"
    "return a single function, an excerpt or a diff."
)
# file-level change: imports, fields, declarations (source payloads are diff excerpts)
WHOLE_INSTRUCTIONS_REGION = (
    "You are adapting a source-side code change into a diverged fork.\n"
    "This change is not inside one function: it edits file-level code (imports,\n"
    "fields, declarations, or several members). Below are the changed parts of\n"
    "the source file BEFORE and AFTER the change (excerpts: a line starting\n"
    "with @@ is a diff hunk header that says where an excerpt sits, it is not\n"
    "code), the corresponding complete {noun} from the fork, and constraints\n"
    "derived from recovered evidence. Apply the same change to the target\n"
    "{noun}, respecting every constraint.\n" + _WHOLE_TAIL
)
# one source function whose counterpart was not located in the fork
WHOLE_INSTRUCTIONS_FUNCTION = (
    "You are adapting a source-side code change into a diverged fork.\n"
    "Below is the change as a before/after pair of ONE function from the source\n"
    "project. No function of that name was located in the fork, so the target\n"
    "shown is the fork's complete {noun}, together with constraints derived\n"
    "from recovered evidence. If the fork's {noun} contains the code this\n"
    "function corresponds to (under another name or signature), apply the same\n"
    "change to it there, respecting every constraint. If the fork has no such\n"
    "code, this change has nothing to adapt here: return the {noun} unchanged\n"
    "and do not add the source function to it.\n" + _WHOLE_TAIL
)
# a function the source change ADDS (no BEFORE side)
WHOLE_INSTRUCTIONS_ADDED = (
    "You are adapting a source-side code change into a diverged fork.\n"
    "The change ADDS the function shown below to the source project (it has no\n"
    "BEFORE side). The target shown is the fork's complete {noun}, together\n"
    "with constraints derived from recovered evidence. Add the function to the\n"
    "fork's {noun} at the corresponding place, adapted to the fork's code and\n"
    "respecting every constraint.\n" + _WHOLE_TAIL
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

# v3 (context-aware). Rounds 1-3 showed that re-prompting with the compiler
# errors alone does not help: the model reverts to the fork's code, because the
# request never showed it what the missing names ARE. v3 is used only when the
# feedback carries `symbol_context` (rcap.adaptctx): for every missing name, the
# source definition when the same change adds it (to inline), or the fork's own
# members when it predates the change (to express the change with the fork's
# API). A feedback without symbol context still renders v2, unchanged.
REPAIR_V3_TEMPLATE_ID = "rcap-reprompt-v3-context"
REPAIR_V3_TEMPLATE_VERSION = "1"

REPAIR_V3_INSTRUCTIONS = (
    "You are fixing a previous attempt at porting a source-side code change\n"
    "into a diverged fork. The port is done one function at a time and only this\n"
    "function can be changed: nothing else of the source change is added to the\n"
    "fork for this build.\n"
    "Below: the change as a before/after pair from the source project; the\n"
    "fork's ORIGINAL function from BEFORE the port (reference only); the\n"
    "PREVIOUS ATTEMPT; the diagnostics it produced when it was built inside the\n"
    "fork; and, for every name the diagnostics report as missing, what that\n"
    "name is in the source project and what the fork has instead.\n"
    "The previous attempt is the code to fix. It fails because it uses things\n"
    "the fork does not have. Keep the source change applied and adapt the\n"
    "previous attempt to the fork, name by name:\n"
    "- ADDED BY THE SAME CHANGE (its source definition is shown, marked not\n"
    "  present in the fork): if it is a small helper whose body only needs what\n"
    "  the fork has, inline its logic into this function in place of the call\n"
    "  (for a constant, use its value). Do not call it and do not declare it.\n"
    "  If its logic cannot be inlined into this function, answer {decline}\n"
    "  naming it.\n"
    "- OLDER THAN THE CHANGE and absent from the fork (the fork's own members\n"
    "  are shown): express the source change with the fork's API exactly as\n"
    "  declared there (names, parameter lists, types). Keep the arguments and\n"
    "  behaviour that only the fork has and that the source change does not\n"
    "  touch.\n"
    "- RENAMED or MOVED BY THE SAME CHANGE: the rename / move is not applied in\n"
    "  the fork; keep the fork's name and package.\n"
    "Do not invent APIs: use only what the ORIGINAL function, the previous\n"
    "attempt and the fork members shown below demonstrate.\n"
    "The ORIGINAL function is the code from before the port. Do NOT resolve the\n"
    "diagnostics by undoing the change: returning the ORIGINAL function is not\n"
    "an acceptable answer, and neither is one that deletes logic, tests or\n"
    "assertions. The only exception: if, after adapting to the fork, the\n"
    "source change has nothing left to port into this function (for example it\n"
    "only switches to a renamed name the fork does not have), return the\n"
    "ORIGINAL function unchanged.\n"
    "If this function cannot be adapted because it needs something that is\n"
    "neither in the fork nor inlinable here, do not return code. Answer with\n"
    "this single line instead:\n"
    "{decline}: needs <symbol or change> from <file>\n"
    "Placeholder comments of the form /* RCAP_PH_... */ stand for elided code:\n"
    "keep each one exactly where it is, character for character, and do not\n"
    "invent code for it.\n"
    "Otherwise return ONLY the complete fixed function in one ```{lang} code\n"
    "block, with no commentary."
)

_ORIGIN_LINE = {
    "added_by_change": "ADDED BY THE SAME CHANGE",
    "renamed_by_change": "RENAMED BY THE SAME CHANGE",
    "moved_by_change": "MOVED BY THE SAME CHANGE",
    "predates_change": "OLDER THAN THE CHANGE",
    "unknown": "NOT FOUND IN THE SOURCE CHANGE",
}


def render_symbol_context(sc, lang: str, noun: str = "function") -> str:
    """One missing name as the v3 request shows it (rcap.reprompt.SymbolContext)."""
    head = f"### `{sc.symbol}`" + (f" (on `{sc.owner}`)" if sc.owner else "")
    lines = [head]
    if sc.usage:
        lines.append(f"Used by the previous attempt as: `{sc.usage}`")
    where = f" in {sc.upstream_file}" if sc.upstream_file else ""
    definition = sc.upstream_text if sc.show_body else sc.upstream_signature
    if sc.origin == "added_by_change":
        lines.append(f"{_ORIGIN_LINE[sc.origin]}{where}. It is NOT PRESENT in the fork and this "
                     "port does not add it.")
        if definition:
            what = ("Source definition" if sc.show_body and sc.upstream_complete
                    else "Source declaration (body not shown"
                         + (f": {sc.upstream_lines} lines" if sc.upstream_lines else "") + ")")
            lines.append(f"{what}:\n```{lang}\n{definition.rstrip()}\n```")
        if sc.names_checked and sc.show_body and sc.upstream_complete:
            if sc.missing_in_fork:
                lines.append("Names this definition uses that occur NOWHERE in the fork (it cannot "
                             "be inlined as it stands): " + ", ".join(f"`{n}`" for n in sc.missing_in_fork))
            else:
                lines.append("Every name this definition uses occurs in the fork.")
            if sc.qualify:
                lines.append("Types it uses that this file does not import (write them fully "
                             "qualified when inlining): " + ", ".join(sc.qualify))
        if sc.fork_members:
            lines.append((sc.fork_heading or "The fork declares") + ":")
            lines += [f"  {m}" for m in sc.fork_members]
            lines.append("If the change can be expressed with the fork's declaration above, "
                         "use that.")
        lines.append(
            f"What to do: if this is a small helper whose body only needs what the fork has, "
            f"inline its logic into this {noun} instead of using `{sc.symbol}`"
            " (for a constant, use its value); otherwise answer "
            f"{DECLINE_TOKEN}: needs {sc.symbol} from {sc.upstream_file or '<file>'}")
    elif sc.origin == "renamed_by_change":
        lines.append(f"{_ORIGIN_LINE[sc.origin]}{where}: `{sc.old_name}` becomes `{sc.symbol}`. "
                     f"The rename is not applied in the fork, which still declares `{sc.old_name}`.")
        lines.append(f"What to do: keep the fork's name `{sc.old_name}`. If the rename is all the "
                     f"source change does in this {noun}, nothing is left to port: return the "
                     f"ORIGINAL {noun} unchanged.")
    elif sc.origin == "moved_by_change":
        lines.append(f"{_ORIGIN_LINE[sc.origin]}{where}. The move is not applied in the fork, "
                     f"which still declares the type as `{sc.old_name}`.")
        lines.append(f"What to do: keep the fork's package / import for `{sc.symbol}`. If the "
                     f"move is all the source change does in this {noun}, nothing is left to "
                     f"port: return the ORIGINAL {noun} unchanged.")
    elif sc.origin == "predates_change":
        lines.append(f"{_ORIGIN_LINE[sc.origin]}: the source project already had it before this "
                     "change" + where + "; the fork does not have it in this form.")
        if sc.upstream_signature:
            lines.append(f"Source declaration: `{sc.upstream_signature}`")
    else:
        lines.append(f"{_ORIGIN_LINE['unknown']}: no declaration of it was found in the change "
                     "or located in the source project.")
    if sc.origin not in ("added_by_change", "moved_by_change"):
        if sc.fork_members:
            lines.append((sc.fork_heading or "The fork declares") + ":")
            lines += [f"  {m}" for m in sc.fork_members]
        elif sc.fork_heading:
            lines.append(sc.fork_heading)
        lines += [f"  {h}" for h in sc.hints]
        if sc.origin == "predates_change":
            lines.append(f"What to do: express the source change with the fork's API as declared "
                         f"above. Keep fork-only arguments and behaviour the change does not touch. "
                         f"If the fork offers nothing equivalent, answer {DECLINE_TOKEN}: needs "
                         f"{sc.symbol} from {sc.upstream_file or '<file>'}")
        elif sc.origin == "unknown":
            lines.append("What to do: do not use this name; use what the fork declares.")
    return "\n".join(lines)


# Placeholder repair (one retry, opt-in). The original request is kept verbatim
# and a single section is appended, so everything the first request said still
# holds and the only new information is what was wrong with the placeholders.
PLACEHOLDER_REPAIR_TEMPLATE_ID = "rcap-placeholder-repair-v1"
PLACEHOLDER_REPAIR_TEMPLATE_VERSION = "1"


def synthesize_placeholder_repair(request: "AdaptationRequest", expected: list[str],
                                  report: dict[str, list[str]]) -> "AdaptationRequest":
    """The retry request for an answer rejected for its placeholder comments.

    `expected`: the placeholder comments of the target function; they are
    listed in the order they stand in the request. `report`: `missing`,
    `duplicated`, `unknown` as rcap.generate.placeholder_report names them."""
    ordered = sorted(expected, key=lambda ph: (request.prompt.rfind(ph) < 0, request.prompt.rfind(ph)))
    lines = ["## Placeholder check FAILED on your previous answer",
             "Your previous answer to this exact request was rejected automatically, before it "
             "was built, because of its placeholder comments:"]
    if report.get("missing"):
        lines.append("- missing (must appear, but was absent): " + ", ".join(report["missing"]))
    if report.get("duplicated"):
        lines.append("- appeared more than once (must appear exactly once): "
                     + ", ".join(report["duplicated"]))
    if report.get("unknown"):
        lines.append("- not a placeholder of the target function (you must not write it): "
                     + ", ".join(report["unknown"]))
    if ordered:
        lines.append(f"The target function contains exactly {len(ordered)} placeholder comment(s), "
                     "in this order:")
        lines += [f"  {ph}" for ph in ordered]
        lines.append(
            "Answer the request again, applying the same change. Your answer must contain each of "
            "these comments exactly once, character for character, at the place where it stands in "
            "the target function, and no other /* RCAP_PH_... */ comment.")
    else:
        lines.append("The target function contains no placeholder comment. Answer the request "
                     "again, applying the same change; your answer must contain no "
                     "/* RCAP_PH_... */ comment at all.")
    lines.append(
        "A placeholder stands for code that is still there: do not delete it, do not write the "
        "code it stands for, and do not replace any other code with a placeholder.")
    lines.append(f"Return ONLY the complete adapted target function in one ```{request.language} "
                 "code block, with no commentary.")
    prompt = request.prompt + "\n" + "\n".join(lines) + "\n"
    request_hash = hashlib.sha256(
        f"{PLACEHOLDER_REPAIR_TEMPLATE_ID}:{PLACEHOLDER_REPAIR_TEMPLATE_VERSION}\n{prompt}"
        .encode()).hexdigest()
    return AdaptationRequest(case_id=request.case_id, language=request.language,
                             template_id=PLACEHOLDER_REPAIR_TEMPLATE_ID,
                             template_version=PLACEHOLDER_REPAIR_TEMPLATE_VERSION,
                             prompt=prompt, request_hash=request_hash)


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
    # What a well-formed answer is: "function" (the default contract), or
    # "file" / "type" when the target is the fork's whole file / a whole type
    # (rcap.unitkind). `generate` rejects an answer of the wrong shape.
    answer_kind: str = FUNCTION


def unit_kinds(context: AdaptationContext) -> tuple[str, str, str]:
    """(source.before kind, source.after kind, target kind) of a unit, from the
    context's own texts (rcap.unitkind.text_kind)."""
    lang = context.language or "java"
    t = context.transformation
    return (text_kind(t.get("source.before"), lang), text_kind(t.get("source.after"), lang),
            text_kind(t.get("target"), lang))


def synthesize(context: AdaptationContext) -> AdaptationRequest:
    # Every fence is labelled with the case's own language: a Scala function
    # fenced as ```java misdescribes the input and invites a Java-shaped answer.
    lang = context.language or "java"
    # Repair feedback rides in the context (never injected into the prompt
    # afterwards), so every feedback line stays traceable (section 17).
    feedback = getattr(context, "feedback", None)
    # Unit kind. `whole` is False for every unit whose target is not a whole
    # file / type, and then each string below is the function wording, verbatim.
    before_kind, after_kind, target_kind = unit_kinds(context)
    whole = target_kind in WHOLE_KINDS
    noun = "function" if not whole else ("file" if target_kind == FILE else "class")
    region = whole and REGION in (before_kind, after_kind)

    def w(text: str) -> str:
        """Function wording -> this unit's wording (identity for function units)."""
        return text if not whole else text.replace("function", noun)

    symbols = list(getattr(feedback, "symbol_context", None) or []) if feedback else []
    v3 = bool(symbols)
    if feedback is None:
        if not whole:
            task = INSTRUCTIONS.format(lang=lang)
        elif region:
            task = WHOLE_INSTRUCTIONS_REGION.format(lang=lang, noun=noun)
        elif before_kind != FUNCTION and after_kind == FUNCTION:
            task = WHOLE_INSTRUCTIONS_ADDED.format(lang=lang, noun=noun)
        else:
            task = WHOLE_INSTRUCTIONS_FUNCTION.format(lang=lang, noun=noun)
    else:
        task = w((REPAIR_V3_INSTRUCTIONS if v3 else REPAIR_INSTRUCTIONS)
                 .format(lang=lang, decline=DECLINE_TOKEN))
        if whole:
            task += (f"\nThe {noun} is the whole target: return ALL of it, from its first line to "
                     "its last, never a single function or an excerpt.")
    if region:
        before_head = "## Source file BEFORE the change (changed parts only)"
        after_head = "## Source file AFTER the change (changed parts only)"
    else:
        before_head = "## Source function BEFORE the change"
        after_head = "## Source function AFTER the change"
    sections = [
        "## Task\n" + task,
        f"{before_head}\n```{lang}\n"
        + context.transformation["source.before"] + "```",
        f"{after_head}\n```{lang}\n"
        + context.transformation["source.after"] + "```",
    ]
    if feedback is None:
        if not whole:
            sections.append(f"## Target function in the fork (adapt this)\n```{lang}\n"
                            + context.transformation["target"] + "```")
        else:
            sections.append(f"## Target {noun} in the fork (adapt this; return all of it)\n"
                            f"```{lang}\n" + context.transformation["target"] + "```")
    else:
        diff = _change_diff(context.transformation["source.before"],
                            context.transformation["source.after"])
        if diff:
            sections.append("## The change to port (diff of BEFORE -> AFTER)\n```diff\n"
                            + diff + "\n```")
        if v3:
            sections.append(w(
                "## ORIGINAL function in the fork, from BEFORE the port (reference only; not an "
                "acceptable answer unless this function has nothing to port)")
                + f"\n```{lang}\n" + context.transformation["target"] + "```")
        else:
            sections.append(w(
                "## ORIGINAL function in the fork, before the port (reference only; "
                "returning it unchanged is NOT an acceptable answer)")
                + f"\n```{lang}\n" + context.transformation["target"] + "```")
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
        narrowing = context.target_localization.get("narrowing")
        if whole and isinstance(narrowing, dict) and narrowing.get("found") is False:
            # Only on whole-target units whose source is one function: say why
            # the whole file is shown, from the recorded decision.
            lines.append(f"NOTE: the source function was not located in this file: "
                         f"{narrowing.get('reason')}.")
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
        elif v3:
            pass    # v3 states it in the task: nothing else of the change is in the fork
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
                            + w("None applied: only this function is changed in the fork "
                                "for this build."))
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
        if v3:
            blocks = [render_symbol_context(sc, lang, noun) for sc in symbols]
            omitted = getattr(feedback, "symbol_context_omitted", 0)
            if omitted:
                blocks.append(f"({omitted} more missing name(s) not described: cut to fit the "
                              "request limit)")
            sections.append(
                "## The names the previous attempt uses that the fork does not have\n"
                + "\n\n".join(blocks))
            sections.append(w(
                f"## Output\nEither one ```{lang} block containing the complete fixed function: "
                "the previous attempt adapted to the fork as described above, with the source "
                "change still applied. The ORIGINAL function unchanged is acceptable only if "
                "nothing of the change is left to port into this function.\n"
                "Or, if this function cannot be adapted because it needs something that is "
                "neither in the fork nor inlinable here, exactly one line:\n"
                f"{DECLINE_TOKEN}: needs <symbol or change> from <file>\n"
                "Nothing else."))
        else:
            sections.append(w(
                f"## Output\nEither one ```{lang} block containing the complete fixed function: "
                "the previous attempt with the diagnostics resolved and the source change still "
                "applied. Returning the ORIGINAL function unchanged is rejected.\n"
                "Or, if this function cannot be fixed because it depends on something that is in "
                "neither the fork nor an APPLIED co-change, exactly one line:\n"
                f"{DECLINE_TOKEN}: needs <symbol or change> from <file>\n"
                "Nothing else."))
    elif not whole:
        sections.append(f"## Output\nOne ```{lang} block containing the full adapted "
                        "target function. Nothing else.")
    else:
        sections.append(f"## Output\nOne ```{lang} block containing the complete adapted target "
                        f"{noun}: every line of it, from the first to the last. Nothing else.")

    prompt = "\n\n".join(sections) + "\n"
    answer_kind = target_kind if whole else FUNCTION
    if feedback is None:
        template_id, version = ((TEMPLATE_ID, TEMPLATE_VERSION) if not whole
                                else (WHOLE_TEMPLATE_ID, WHOLE_TEMPLATE_VERSION))
        request_hash = hashlib.sha256(
            f"{template_id}:{version}\n{prompt}".encode()).hexdigest()
        return AdaptationRequest(case_id=context.case_id, language=lang,
                                 template_id=template_id, template_version=version,
                                 prompt=prompt, request_hash=request_hash,
                                 answer_kind=answer_kind)
    template_id, version = ((REPAIR_V3_TEMPLATE_ID, REPAIR_V3_TEMPLATE_VERSION) if v3
                            else (REPAIR_TEMPLATE_ID, REPAIR_TEMPLATE_VERSION))
    # The attempt index is part of the hash, so every attempt of a case has a
    # distinct request_hash (and hence exec_id) even if two prompts coincide.
    request_hash = hashlib.sha256(
        f"{template_id}:{version}:attempt={feedback.attempt}\n"
        f"{prompt}".encode()).hexdigest()
    return AdaptationRequest(case_id=context.case_id, language=lang,
                             template_id=template_id,
                             template_version=version,
                             prompt=prompt, request_hash=request_hash,
                             answer_kind=answer_kind)


def untraceable_lines(prompt: str, context: AdaptationContext) -> list[str]:
    """Section 17 mapping test: every request fact must originate in the
    context. Synthesis is deterministic (R = Psi(C, I)), so the reference
    rendering of the context is the complete set of legitimate lines; any
    non-empty line of the given prompt outside it has no context origin —
    invented evidence."""
    legitimate = set(synthesize(context).prompt.splitlines())
    return [line for line in prompt.splitlines()
            if line.strip() and line not in legitimate]
