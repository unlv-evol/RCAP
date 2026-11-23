"""gcc diagnostics -> the feedback of a repair request, and the names it misses.

The counterpart for C of rcap.feedback (javac / scalac): pure text processing,
no compiler is run here. Line numbers are rewritten relative to the function
that was spliced in, because that function is all the model sees.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

MAX_ERRORS = 15          # as rcap.feedback
MAX_CHARS = 4000
MAX_BLOCK_LINES = 8

_HEAD = re.compile(r"^(?P<file>[^:\s][^:\n]*):(?P<line>\d+):(?:(?P<col>\d+):)? "
                   r"(?P<sev>fatal error|error|warning|note): (?P<msg>.*)$")
_CONTEXT = re.compile(r"^(?:In file included from |\s+from |[^:\n]+: (?:In function|At top level))")
_QUOTE = "'`‘’"

_MISSING = (
    (re.compile(rf"implicit declaration of function [{_QUOTE}](?P<name>\w+)[{_QUOTE}]"), "function"),
    (re.compile(rf"[{_QUOTE}](?P<name>\w+)[{_QUOTE}] undeclared"), "variable"),
    (re.compile(rf"unknown type name [{_QUOTE}](?P<name>\w+)[{_QUOTE}]"), "type"),
    (re.compile(rf"[{_QUOTE}](?P<owner>[^{_QUOTE}]+)[{_QUOTE}](?: \{{aka [{_QUOTE}][^{_QUOTE}]+[{_QUOTE}]\}})? "
                rf"has no member named [{_QUOTE}](?P<name>\w+)[{_QUOTE}]"), "member"),
    (re.compile(rf"too (?:few|many) arguments to function [{_QUOTE}](?P<name>\w+)[{_QUOTE}]"), "signature"),
    (re.compile(rf"incompatible type for argument \d+ of [{_QUOTE}](?P<name>\w+)[{_QUOTE}]"), "signature"),
    (re.compile(rf"called object [{_QUOTE}](?P<name>\w+)[{_QUOTE}] is not a function"), "signature"),
    (re.compile(rf"conflicting types for [{_QUOTE}](?P<name>\w+)[{_QUOTE}]"), "signature"),
    (re.compile(rf"storage size of [{_QUOTE}](?P<name>\w+)[{_QUOTE}] isn[{_QUOTE}]t known"), "type"),
)


@dataclass
class Diag:
    path: str
    line: int
    severity: str
    message: str
    block: list[str] = field(default_factory=list)   # the source excerpt gcc prints under it
    notes: list[str] = field(default_factory=list)   # attached `note:` lines

    @property
    def code_line(self) -> str:
        for ln in self.block:
            m = re.match(r"^\s*\d+ \| (.*)$", ln)
            if m:
                return m.group(1).strip()
        return ""


@dataclass
class Missing:
    name: str
    kind: str                 # function | variable | type | member | signature
    owner: str | None = None
    usage: str = ""
    line: int | None = None


def parse_gcc(log: str) -> list[Diag]:
    """Errors (with their notes) of a gcc run; warnings are dropped."""
    out: list[Diag] = []
    cur: Diag | None = None
    keep = False
    for raw in (log or "").splitlines():
        m = _HEAD.match(raw)
        if m:
            sev = m.group("sev")
            if sev == "note":
                if cur is not None and keep:
                    cur.notes.append(m.group("msg"))
                continue
            keep = sev in ("error", "fatal error")
            cur = Diag(m.group("file"), int(m.group("line")), sev, m.group("msg"))
            if keep:
                out.append(cur)
            continue
        if _CONTEXT.match(raw) or raw.startswith("cc1:") or raw.startswith("compilation terminated"):
            continue
        if cur is not None and keep and not cur.notes and len(cur.block) < MAX_BLOCK_LINES and raw.strip():
            cur.block.append(raw.rstrip())
    return out


def missing_names(errors: list[Diag]) -> list[Missing]:
    """The names the errors report as absent or used differently, once each."""
    out: list[Missing] = []
    seen: set[tuple[str, str | None]] = set()
    for e in errors:
        for rx, kind in _MISSING:
            m = rx.search(e.message)
            if not m:
                continue
            owner = m.groupdict().get("owner")
            key = (m.group("name"), owner)
            if key not in seen:
                seen.add(key)
                out.append(Missing(m.group("name"), kind, owner, e.code_line, e.line))
            break
    return out


def render(errors: list[Diag], *, first_line: int, last_line: int, rel_path: str,
           max_errors: int = MAX_ERRORS, max_chars: int = MAX_CHARS) -> tuple[str, dict[str, int]]:
    """Feedback text for the repair request. `first_line`..`last_line`: the
    lines of the compiled file the candidate function occupies; an error in
    there is shown with its line WITHIN the function, any other with the file
    line and a note that it lies outside the function."""
    blocks: list[str] = []
    seen: set[tuple[int, str]] = set()
    inside = outside = 0
    for e in errors:
        if (e.line, e.message) in seen:
            continue
        seen.add((e.line, e.message))
        if e.path.endswith(rel_path) and first_line <= e.line <= last_line:
            inside += 1
            head = f"line {e.line - first_line + 1} of the function: error: {e.message}"
        else:
            outside += 1
            where = e.path.split("/src/", 1)[-1] if "/src/" in e.path else e.path
            head = f"{where}:{e.line} (outside the function): error: {e.message}"
        lines = [head]
        code = e.code_line
        if code:
            lines.append(f"    {code}")
        lines += [f"    note: {n}" for n in e.notes[:2]]
        blocks.append("\n".join(lines))
    shown = blocks[:max_errors]
    text = "\n".join(shown)
    while len(text) > max_chars and len(shown) > 1:
        shown = shown[:-1]
        text = "\n".join(shown)
    total = len(blocks)
    head = f"{total} compile error(s) (gcc, the fork's own build flags)" + (
        f"; the first {len(shown)} are shown" if len(shown) < total else "") + ":\n"
    return head + text, {"errors": total, "shown": len(shown), "inside_function": inside,
                         "outside_function": outside}
