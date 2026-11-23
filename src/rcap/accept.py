"""Text checks every candidate must pass before it is accepted or applied.

Two defects make a candidate that parses, builds and passes its tests wrong
all the same, and neither can be seen by a build:

PLACEHOLDER LEAK   a placeholder marker of ANY spelling (`RCAP_PH\\w*`) in text
    that is about to be written into the fork. The first check was the
    substring `"RCAP_PH_"`; a model also writes `/* RCAP_PHname */`, which that
    test let through (PR-15476 ListOffsetsIntegrationTest).

DUPLICATED BLOCK   the candidate holds one more copy of a block of code than
    it should: the model wrote the changed version of a loop IN FRONT OF the
    loop it was meant to change (PR-13212, PR-14026), or kept a placeholder
    comment AND wrote the elided block out again, so that expansion puts it in
    twice. The marker check is blind to both.

`duplicated_blocks` is deliberately narrow (precision over recall): a window of
`min_lines` consecutive code lines (comments and blank lines dropped, layout
ignored) of which at least two are substantive must
  * occur in the fork original,
  * occur MORE often in the candidate than in the fork original,
  * that increase must exceed the increase upstream itself makes (before ->
    after), so a block upstream really does repeat is not reported, and
  * the surplus copy must stand in the method that already had the block (same
    method name; or outside any method in both texts). A new test method that
    repeats the set-up lines of its neighbour, or a second constructor, is
    ordinary code; the same statements once more in the body that had them is
    what a build cannot see and what the audit found.
Counting increases rather than copies keeps it usable when the upstream texts
are diff excerpts of a file and the candidate is the whole file.

Text processing, plus one tree-sitter parse of the candidate of the fork text and of the candidate to find
their method bodies. A text that does not parse is not reported (it does not build).
"""

from __future__ import annotations

import re

PLACEHOLDER_MARK = re.compile(r"RCAP_PH\w*")

_STRIP = re.compile(r'''
    (?P<string>"""(?:.|\n)*?"""|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])')
  | (?P<comment>//[^\n]*|/\*.*?(?:\*/|\Z))
''', re.DOTALL | re.VERBOSE)
_WS = re.compile(r"\s+")
_OPEN_BRACE = re.compile(r"\s*\{$")
_WORD = re.compile(r"[A-Za-z_$][\w$]*|\d[\w.]*")
_HUNK_HEADER = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@[^\n]*$", re.MULTILINE)
# words that do not make a line substantive on their own (`} else {`, `try {`, `return;`)
_FILLER = frozenset({"else", "try", "finally", "return", "break", "continue", "do", "case", "default"})
# a boundary between excerpts: no window may run across it
_BREAK = "\0"

MIN_BLOCK_LINES = 3


def placeholder_marks(text: str | None) -> list[str]:
    """Every placeholder marker in `text`, whatever its spelling."""
    return PLACEHOLDER_MARK.findall(text or "")


def has_placeholder_leak(text: str | None) -> bool:
    """A placeholder marker in text that is about to be written into the fork
    (i.e. AFTER expansion) is never acceptable."""
    return bool(text) and PLACEHOLDER_MARK.search(text) is not None


def strip_comments(code: str) -> str:
    """`code` without comments; string literals are left alone."""
    return _STRIP.sub(lambda m: m.group(0) if m.lastgroup == "string" else " ", code or "")


def code_lines(code: str, numbers: list[int] | None = None) -> list[str]:
    """Code lines with comments, blank lines and layout removed. Excerpt
    boundaries (`@@ ... @@` headers) become a break no window may cross.
    `numbers`, when given, receives the 0-based source line of every entry."""
    out: list[str] = []
    code = code or ""
    # comments are blanked line by line so that line numbers survive
    stripped = _STRIP.sub(lambda m: m.group(0) if m.lastgroup == "string"
                          else re.sub(r"[^\n]", " ", m.group(0)), code)
    raw = code.splitlines()
    for n, line in enumerate(stripped.splitlines()):
        if n < len(raw) and _HUNK_HEADER.match(raw[n]):
            out.append(_BREAK)
            if numbers is not None:
                numbers.append(n)
            continue
        # an opening brace at the end of a line is layout: `for (..) {` and a
        # brace-less `for (..)` are the same line (PR-15910 KafkaBasedLog)
        line = _OPEN_BRACE.sub("", _WS.sub(" ", line).strip()).strip()
        if line:
            out.append(line)
            if numbers is not None:
                numbers.append(n)
    return out


def _method_of_line(text: str, language: str | None) -> list[str] | None:
    """For every source line of `text`: the name of the outermost method it
    lies in, "" outside any method. None when the text cannot be parsed."""
    from rcap.grammars import function_name_node, grammar_for, is_classless, parse

    grammar = grammar_for((language or "java").lstrip("."))
    if grammar is None:
        return None
    lines = text.count("\n") + 1
    # C: a function stands at the top of the file, there is no class to wrap it in
    for head in (("",) if is_classless(grammar) else ("", "class __RcapProbe {\n")):
        src = head + text + ("\n}" if head else "")
        tree = parse(src, grammar)
        if tree is None or tree.root_node.has_error:
            continue
        raw = src.encode()
        shift = 1 if head else 0
        owner = [""] * (lines + 2)
        stack = [tree.root_node]
        while stack:
            node = stack.pop()
            if node.type in grammar.method_like:
                ident = function_name_node(node)
                name = raw[ident.start_byte:ident.end_byte].decode() if ident is not None else "<init>"
                for ln in range(node.start_point[0] - shift, node.end_point[0] - shift + 1):
                    if 0 <= ln < len(owner):
                        owner[ln] = name
                continue                      # outermost only
            stack.extend(node.children)
        return owner
    return None


def _substantive(line: str) -> bool:
    words = [w for w in _WORD.findall(line) if w not in _FILLER]
    return len(words) >= 2


def _windows(lines: list[str], n: int) -> dict[tuple[str, ...], list[int]]:
    out: dict[tuple[str, ...], list[int]] = {}
    for i in range(len(lines) - n + 1):
        w = tuple(lines[i:i + n])
        if _BREAK in w or sum(1 for x in w if _substantive(x)) < 2:
            continue
        out.setdefault(w, []).append(i)
    return out


def duplicated_blocks(fork_original: str, candidate: str, upstream_before: str = "",
                      upstream_after: str = "", *, language: str | None = None,
                      min_lines: int = MIN_BLOCK_LINES) -> list[dict[str, object]]:
    """Blocks the candidate holds one more copy of than the fork and upstream
    account for (see the module docstring). One entry per maximal block:
    {"lines": [...], "copies_fork", "copies_candidate", "copies_upstream_before",
    "copies_upstream_after"}; empty when there is none."""
    numbers: list[int] = []
    cl = code_lines(candidate, numbers)
    fork_numbers: list[int] = []
    c, f = _windows(cl, min_lines), _windows(code_lines(fork_original, fork_numbers), min_lines)
    if not c or not f:
        return []
    b, a = _windows(code_lines(upstream_before), min_lines), _windows(code_lines(upstream_after), min_lines)
    bad: dict[int, tuple[str, ...]] = {}
    owner_c: list[str] | None = None
    owner_f: list[str] | None = None
    for w, where in c.items():
        at_fork = f.get(w, ())
        if not at_fork or len(where) <= len(at_fork):
            continue
        if len(where) - len(at_fork) <= max(0, len(a.get(w, ())) - len(b.get(w, ()))):
            continue
        if owner_c is None:
            owner_c = _method_of_line(candidate, language) or []
            owner_f = _method_of_line(fork_original, language) or []
        if not owner_c or not owner_f:
            return []
        in_c = [owner_c[numbers[i]] for i in where]
        in_f = [owner_f[fork_numbers[i]] for i in at_fork]
        # the method that had the block has it more often now
        grown = {m for m in in_f if in_c.count(m) > in_f.count(m)}
        for i, m in zip(where, in_c):
            if m in grown:
                bad[i] = w
    if not bad:
        return []
    # join overlapping windows of the FIRST surplus copy into maximal blocks
    out: list[dict[str, object]] = []
    seen: set[tuple[str, ...]] = set()
    starts = sorted(bad)
    i = 0
    while i < len(starts):
        j = i
        while j + 1 < len(starts) and starts[j + 1] == starts[j] + 1:
            j += 1
        block = tuple(cl[starts[i]:starts[j] + min_lines])
        w = bad[starts[i]]
        if block not in seen and not any(set(block) <= set(s) for s in seen):
            seen.add(block)
            out.append({"lines": list(block), "copies_fork": len(f.get(w, ())),
                        "copies_candidate": len(c[w]),
                        "copies_upstream_before": len(b.get(w, ())),
                        "copies_upstream_after": len(a.get(w, ()))})
        i = j + 1
    return out


def describe_duplicates(blocks: list[dict[str, object]]) -> list[str]:
    return [(f"a block of {len(x['lines'])} code lines stands {x['copies_candidate']} times in the candidate "
             f"({x['copies_fork']} in the fork, upstream {x['copies_upstream_before']} -> "
             f"{x['copies_upstream_after']}): `{' / '.join(x['lines'][:3])[:160]}`") for x in blocks]  # type: ignore[index,arg-type]


def rejection(candidate_expanded: str | None, fork_original: str | None = None,
              upstream_before: str = "", upstream_after: str = "",
              *, language: str | None = None) -> tuple[str, list[str]] | None:
    """(reason, evidence) when text about to be written into the fork must not
    be: `placeholder_leak` or `duplicated_block`; None when it may."""
    marks = placeholder_marks(candidate_expanded)
    if marks:
        return "placeholder_leak", [f"placeholder marker(s) left in the code: {', '.join(dict.fromkeys(marks))[:200]}"]
    if candidate_expanded and fork_original:
        dup = duplicated_blocks(fork_original, candidate_expanded, upstream_before, upstream_after,
                                language=language)
        if dup:
            return "duplicated_block", describe_duplicates(dup)
    return None
