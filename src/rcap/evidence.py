"""Edit evidence: what else the same upstream change edits, for a function unit's prompt.

Off by default (ExecutionConfig.edit_evidence); off, every context and prompt is
byte-identical to what RCAP wrote before. On, a function unit's context carries

* `other_edits`: the change's edits OUTSIDE functions in the same SAP (imports,
  fields, class or file-level code): for each such unit, the lines its
  source.before -> source.after removes and adds, from the materialized
  payloads (never re-read from disk). These units leave the "signatures only"
  list, where they could only be shown as `?`.
* `pr_functions`: the signatures (never bodies) of the functions the same pull
  request adds, removes or re-signs in its OTHER files (SAPs), read from those
  SAPs' function payloads. Functions whose body alone changes are left out:
  calls to them stay valid, and on large PRs they are most of the list.

Both are SAP evidence only: nothing from the fork, nothing invented. Rendering is
deterministic and capped (`render_sections`), with an explicit truncation note.
"""

from __future__ import annotations

import difflib
import functools
import json
import re
from pathlib import Path

from pydantic import BaseModel

from rcap.unitkind import FILE, FUNCTION, REGION, TYPE, text_kind

_HUNK_HEADER = re.compile(r"^\s*@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@")
_COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*", re.DOTALL)

# Section caps, in characters of the rendered section body (about 3.5 characters per
# token on Java, so together under 3k tokens). Measured uncapped on the 2,604 Java function
# units of study 2026-11-candidates-32k: other edits p95 4.7k / max 23k chars (61 prompts cut
# at 6k); re-signed functions of other files p90 3.6k / max 27k chars (210 prompts cut at 4k).
OTHER_EDITS_CHARS = 6000
PR_FUNCTIONS_CHARS = 4000


class OtherEdit(BaseModel):
    """One non-function unit of the same SAP and the lines its change removes / adds."""

    entity: str
    kind: str            # imports | type | file | region | other
    lines: list[str]     # "- old line" / "+ new line", in diff order, stripped


class PRFunction(BaseModel):
    """A function the same PR edits in another file (SAP): its signature only."""

    file: str
    entity: str
    signature_before: str | None = None
    signature_after: str | None = None

    @property
    def status(self) -> str:
        if self.signature_before is None and self.signature_after is not None:
            return "added"
        if self.signature_after is None and self.signature_before is not None:
            return "removed"
        if self.signature_before != self.signature_after:
            return "signature changed"
        return "body changed"


def is_function_payload(before: str | None, after: str | None, language: str) -> bool:
    """A unit is a function unit when each side is exactly one function or empty
    (an added or removed function), and at least one side is a function."""
    kb, ka = text_kind(before, language), text_kind(after, language)
    return FUNCTION in (kb, ka) and (kb == FUNCTION or not (before or "").strip()) \
        and (ka == FUNCTION or not (after or "").strip())


def change_lines(before: str, after: str) -> list[str]:
    """The lines a before/after pair removes ("- ") and adds ("+ "), in diff
    order, whitespace-stripped; blank lines and diff-excerpt headers ignored."""
    a, b = before.splitlines(), after.splitlines()
    out: list[str] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        out.extend("- " + x.strip() for x in a[i1:i2] if x.strip() and not _HUNK_HEADER.match(x))
        out.extend("+ " + x.strip() for x in b[j1:j2] if x.strip() and not _HUNK_HEADER.match(x))
    return out


def _kind_label(entity: str, before: str, after: str, language: str) -> str:
    if entity.endswith("_imports"):
        return "imports"
    kinds = {text_kind(before, language), text_kind(after, language)}
    for k in (TYPE, FILE, REGION):
        if k in kinds:
            return {TYPE: "class/type level", FILE: "file level", REGION: "file excerpt"}[k]
    return "outside functions"


def other_edits(sibling_entities, materialized, language: str) -> tuple[list[OtherEdit], set[str]]:
    """(the non-function units among `sibling_entities` with their changed
    lines, the set of those entities). Order is the siblings' order."""
    edits: list[OtherEdit] = []
    moved: set[str] = set()
    for entity in sibling_entities:
        roles = materialized.by_role(entity)
        before = roles.get("source.before")
        after = roles.get("source.after")
        if before is None or after is None:
            continue
        if is_function_payload(before.content, after.content, language):
            continue
        moved.add(entity)
        lines = change_lines(before.content, after.content)
        if lines:
            edits.append(OtherEdit(entity=entity.split("/", 1)[-1],
                                   kind=_kind_label(entity, before.content, after.content, language),
                                   lines=lines))
    return edits, moved


def declaration_of(text: str) -> str | None:
    """A function's declaration: its text up to the body's `{` (or the `;` of a
    bodyless declaration) at parenthesis depth 0, so a brace in an annotation
    argument does not end it; comments dropped (a moved or reworded javadoc is
    not a signature change), whitespace collapsed. None without a `(`."""
    text = _COMMENT.sub(" ", text)
    depth, quote, prev = 0, None, ""
    for i, ch in enumerate(text):
        if quote:
            if ch == quote and prev != "\\":
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch in "{;" and depth <= 0:
            head = re.sub(r"\s+", " ", text[:i]).strip()
            return head if "(" in head else None
        prev = ch
    return None


@functools.lru_cache(maxsize=4096)
def _sap_functions(sap_dir: str) -> tuple[PRFunction, ...]:
    """The function units of one SAP with their before/after signatures."""
    d = Path(sap_dir)
    try:
        man = json.loads((d / "sap.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ()
    src = man.get("source_file") or man.get("target_file") or ""
    language = src.rsplit(".", 1)[-1].lower() if "." in src.rsplit("/", 1)[-1] else "java"
    out = []
    for fn in man.get("functions") or []:
        fdir = d / "functions" / fn
        texts = {}
        for role in ("source.before", "source.after"):
            f = next((p for p in sorted(fdir.glob(role + ".*")) if p.is_file()), None)
            try:
                texts[role] = f.read_bytes().decode("utf-8") if f else ""
            except (OSError, UnicodeDecodeError):
                texts[role] = ""
        b, a = texts["source.before"], texts["source.after"]
        if b == a or not is_function_payload(b, a, language):
            continue
        sb = declaration_of(b) if b.strip() else None
        sa = declaration_of(a) if a.strip() else None
        if sb is None and sa is None:
            continue
        out.append(PRFunction(file=src.rsplit("/", 1)[-1], entity=fn,
                              signature_before=sb, signature_after=sa))
    return tuple(out)


def pr_functions(sap_dir, pr_manifest: dict | None) -> list[PRFunction]:
    """Function units of the PR's OTHER SAPs whose signature the change alters:
    signature changed, then removed, then added; each group in pr.json order,
    then each SAP's function order."""
    if not pr_manifest:
        return []
    here = Path(sap_dir).resolve()
    pr_dir = here.parent
    found: list[PRFunction] = []
    for s in pr_manifest.get("saps") or []:
        rel = str(s.get("path") or "").rstrip("/")
        if not rel:
            continue
        other = (pr_dir / rel).resolve()
        if other == here or not other.is_dir():
            continue
        found.extend(_sap_functions(str(other)))
    order = ("signature changed", "removed", "added")
    return [f for status in order for f in found if f.status == status]


def _capped(lines: list[str], cap: int, what: str) -> list[str]:
    out, used = [], 0
    for n, line in enumerate(lines):
        if used + len(line) + 1 > cap:
            out.append(f"[... {len(lines) - n} more {what} not shown: cut at {cap} characters]")
            return out
        out.append(line)
        used += len(line) + 1
    return out


def render_sections(context, lang: str) -> list[str]:
    """The edit-evidence prompt sections of a context (empty when it has none)."""
    sections: list[str] = []
    edits = getattr(context, "other_edits", None) or []
    if edits:
        body: list[str] = []
        for e in edits:
            body.append(f"# {e.entity} ({e.kind})")
            body.extend(e.lines)
        body = _capped(body, OTHER_EDITS_CHARS, "lines")
        sections.append(
            "## Other edits made by this same change (outside functions)\n"
            "The same upstream change also makes these edits in this file, outside any function\n"
            "(lines removed `-` and added `+`). They are ported separately: do not copy them into\n"
            "your answer, but the function may use what they introduce (imports, fields, types).\n"
            "```diff\n" + "\n".join(body) + "\n```")
    funcs = getattr(context, "pr_functions", None) or []
    if funcs:
        lines = []
        for f in funcs:
            if f.status == "signature changed":
                lines.append(f"- {f.file}: `{f.signature_after}` (changed from `{f.signature_before}`)")
            elif f.status == "removed":
                lines.append(f"- {f.file}: `{f.signature_before}` (removed)")
            else:
                lines.append(f"- {f.file}: `{f.signature_after}`" + (" (added)" if f.status == "added" else ""))
        lines = _capped(lines, PR_FUNCTIONS_CHARS, "functions")
        sections.append(
            "## Functions this same change adds, removes or re-signs in other files (signatures only)\n"
            "Calls to these must match their signatures after the change.\n"
            + "\n".join(lines))
    return sections
