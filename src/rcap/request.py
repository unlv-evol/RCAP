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
    "Return ONLY the complete adapted target function in one ```java code\n"
    "block, with no commentary."
)


class AdaptationRequest(BaseModel):
    case_id: str
    template_id: str = TEMPLATE_ID
    template_version: str = TEMPLATE_VERSION
    prompt: str
    request_hash: str


def synthesize(context: AdaptationContext) -> AdaptationRequest:
    sections = [
        "## Task\n" + INSTRUCTIONS,
        "## Source function BEFORE the change\n```java\n"
        + context.transformation["source.before"] + "```",
        "## Source function AFTER the change\n```java\n"
        + context.transformation["source.after"] + "```",
        "## Target function in the fork (adapt this)\n```java\n"
        + context.transformation["target"] + "```",
    ]
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
    if context.target_localization.get("file"):
        sections.append("## Target location\n"
                        f"File: {context.target_localization['file']}")
    sections.append("## Output\nOne ```java block containing the full adapted "
                    "target function. Nothing else.")

    prompt = "\n\n".join(sections) + "\n"
    request_hash = hashlib.sha256(
        f"{TEMPLATE_ID}:{TEMPLATE_VERSION}\n{prompt}".encode()).hexdigest()
    return AdaptationRequest(case_id=context.case_id, prompt=prompt,
                             request_hash=request_hash)
