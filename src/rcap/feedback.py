"""Build/test diagnostics -> compact feedback for the re-prompt loop.

Pure text processing (no build system knowledge beyond the javac/scalac
message shapes and the JUnit XML report format), so it is unit-testable and
reusable by any driver. The goal is the opposite of a log tail: the most
relevant error blocks, complete (message, offending source line, javac's
`symbol:` / `location:` lines), the edited file first, de-duplicated and
capped. Line numbers are mapped onto the processing unit they fall in.
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

MAX_ERRORS = 15
MAX_CHARS = 4000
MAX_BLOCK_LINES = 8
MAX_TESTS = 8
MAX_FRAMES = 5

OTHER_FILE_NOTE = (
    "Note: errors in other files come from existing code that uses this function. "
    "Those files cannot be changed; only this function can, so it must stay "
    "compatible with them.")

# javac:   /abs/File.java:123: error: cannot find symbol
# scalac:  [Error] /abs/File.scala:12:5: not found: value x      (zinc via Gradle)
_HEAD = re.compile(
    r"^(?:\[(?P<lvl>Error|Warn|error|warn)\]\s+)?(?P<path>/[^:\n]+\.(?:java|scala|sc))"
    r":(?P<line>\d+)(?::(?P<col>\d+))?:\s*(?P<msg>.*)$")
_STOP = re.compile(r"^(FAILURE:|> Task|\* What went wrong|\d+ errors?$|\d+ warnings?$|Note: |"
                   r"\[(?:Error|Warn|error|warn)\]\s|(?:one|two|three|four|\d+) errors? found)")
# Stack frames that say nothing about the code under adaptation.
_NOISE_FRAME = re.compile(
    r"^\s*at (?:org\.junit\.|org\.gradle\.|java\.base/|jdk\.|sun\.|com\.sun\.|"
    r"java\.lang\.reflect\.|java\.util\.|org\.mockito\.|org\.easymock\.|org\.powermock\.|"
    r"org\.opentest4j\.|worker\.org\.gradle\.|scala\.)")


@dataclass
class CompileError:
    path: str            # as printed by the compiler
    line: int
    message: str
    col: int | None = None
    body: list[str] = field(default_factory=list)  # source line, caret, symbol:, location:
    in_edited_file: bool = False
    unit: int | None = None        # index of the unit whose span holds the line
    unit_line: int | None = None   # 1-based line inside that unit's candidate
    also_at: list[int] = field(default_factory=list)  # same error at other lines


@dataclass
class Feedback:
    kind: str                      # build | test
    text: str                      # rendered, capped
    summary: str                   # one line, for records
    n_total: int = 0               # distinct errors / failing tests found
    n_shown: int = 0
    per_unit: dict[int, str] = field(default_factory=dict)  # unit index -> rendered text
    units_with_errors: list[int] = field(default_factory=list)
    mappable: bool = False         # at least one error fell inside a unit span


def parse_compile_errors(log: str) -> list[CompileError]:
    """Every compiler ERROR block in a full build log (warnings are skipped)."""
    errors: list[CompileError] = []
    current: CompileError | None = None
    for raw in log.splitlines():
        line = raw.rstrip()
        m = _HEAD.match(line)
        if m:
            current = None
            msg = m.group("msg")
            lvl = (m.group("lvl") or "").lower()
            if lvl == "warn" or msg.startswith("warning:"):
                continue
            if msg.startswith("error:"):
                msg = msg[len("error:"):].strip()
            current = CompileError(path=m.group("path"), line=int(m.group("line")),
                                   message=msg,
                                   col=int(m.group("col")) if m.group("col") else None)
            errors.append(current)
            continue
        if current is None:
            continue
        if not line.strip() or _STOP.match(line):
            current = None
            continue
        if len(current.body) < MAX_BLOCK_LINES:
            current.body.append(line)
    return errors


def _dedup(errors: Iterable[CompileError]) -> list[CompileError]:
    """Collapse repeats: the same (file, message, detail) at another line is
    listed on the first occurrence rather than rendered again."""
    seen: dict[tuple, CompileError] = {}
    out: list[CompileError] = []
    for e in errors:
        detail = tuple(b.strip() for b in e.body if b.strip() != "^")
        key = (e.path, e.message, detail)
        first = seen.get(key)
        if first is None:
            seen[key] = e
            out.append(e)
        elif e.line != first.line and e.line not in first.also_at:
            first.also_at.append(e.line)
    return out


def _render_error(e: CompileError, root: str | None) -> str:
    path = e.path
    if root and path.startswith(root.rstrip("/") + "/"):
        path = path[len(root.rstrip("/")) + 1:]
    where = f"{path}:{e.line}"
    if e.unit_line is not None:
        where += f" [line {e.unit_line} of your function]"
    elif e.unit is not None:
        where += " [inside your function]"
    elif e.in_edited_file:
        where += " [in the edited file, outside your function]"
    else:
        where += " [in another file that uses the edited one]"
    out = [f"- {where}: {e.message}"]
    out.extend("    " + b for b in e.body)
    if e.also_at:
        out.append(f"    (same error also at lines {', '.join(map(str, e.also_at[:10]))})")
    return "\n".join(out)


def _cap(blocks: Sequence[str], total: int, noun: str,
         max_items: int, max_chars: int) -> tuple[str, int]:
    shown: list[str] = []
    used = 0
    for b in blocks[:max_items]:
        if shown and used + len(b) + 1 > max_chars:
            break
        if not shown and len(b) > max_chars:
            b = b[:max_chars] + "\n    [...]"
        shown.append(b)
        used += len(b) + 1
    text = "\n".join(shown)
    if total > len(shown):
        text += f"\n({total - len(shown)} more {noun} omitted)"
    return text, len(shown)


def _fallback(log: str, max_chars: int) -> str:
    """No file:line error block: quote Gradle's own failure description."""
    lines = log.splitlines()
    for i, line in enumerate(lines):
        if line.startswith("* What went wrong"):
            return "\n".join(lines[i + 1:i + 25]).strip()[:max_chars]
    return "\n".join(lines[-25:]).strip()[:max_chars]


def extract_build_feedback(
    log: str, edited_file: str, unit_spans: Sequence[tuple[int, int]] = (),
    *, root: str | None = None, max_errors: int = MAX_ERRORS, max_chars: int = MAX_CHARS,
    unit_line: Callable[[int, int], int | None] | None = None,
    edited_source: str | None = None,
) -> Feedback:
    """Feedback from a full compiler log.

    `edited_file` is the repo-relative path of the spliced file; `unit_spans`
    are the 1-based inclusive (first, last) file lines each unit's candidate
    occupies after splicing, in unit order. Errors in the edited file come
    first; an error inside a span is attributed to that unit and its line is
    also given relative to the candidate the model wrote. `unit_line(unit,
    line_in_spliced_candidate)` translates that line when the spliced text is
    not the text the model wrote (expanded placeholders); None suppresses it.
    `edited_source` is the spliced file's text: some compilers (javac driven by
    zinc in a mixed Scala/Java project) print only `file:line:col: error: msg`,
    and the offending line is then quoted from it."""
    errors = parse_compile_errors(log)
    source_lines = edited_source.splitlines() if edited_source is not None else []
    for e in errors:
        e.in_edited_file = e.path.endswith("/" + edited_file) or e.path == edited_file
        if e.in_edited_file and not e.body and 0 < e.line <= len(source_lines):
            e.body = [source_lines[e.line - 1]]
            if e.col:
                e.body.append(" " * (e.col - 1) + "^")
        if e.in_edited_file:
            for idx, (first, last) in enumerate(unit_spans):
                if first <= e.line <= last:
                    e.unit, e.unit_line = idx, e.line - first + 1
                    if unit_line is not None:
                        e.unit_line = unit_line(idx, e.unit_line)
                    break
    errors = _dedup(errors)
    # edited file first (errors inside a unit before the rest of the file), stable
    errors.sort(key=lambda e: (0 if e.unit is not None else 1 if e.in_edited_file else 2))
    if not errors:
        text = _fallback(log, max_chars)
        return Feedback(kind="build", text=text, n_total=0, n_shown=0,
                        summary="build failed; no file:line compiler error found")
    text, shown = _cap([_render_error(e, root) for e in errors], len(errors), "errors",
                       max_errors, max_chars)
    # Errors in files the model cannot edit are almost always callers that no
    # longer match: say so, since "fix the caller" is not an available move.
    note = ("\n" + OTHER_FILE_NOTE) if any(not e.in_edited_file for e in errors) else ""
    text += note
    units = sorted({e.unit for e in errors if e.unit is not None})
    per_unit: dict[int, str] = {}
    for u in units:
        # this unit's own errors, then everything not attributable to a sibling
        mine = [e for e in errors if e.unit == u or e.unit is None]
        per_unit[u], _ = _cap([_render_error(e, root) for e in mine], len(mine), "errors",
                              max_errors, max_chars)
        if any(not e.in_edited_file for e in mine):
            per_unit[u] += "\n" + OTHER_FILE_NOTE
    n_file = sum(e.in_edited_file for e in errors)
    first = errors[0]
    summary = (f"{len(errors)} compile error(s), {n_file} in the edited file; first: "
               f"{Path(first.path).name}:{first.line}: {first.message}"
               + (f" / {first.body[-1].strip()}" if first.body
                  and first.body[-1].strip().startswith(("symbol", "location")) else ""))
    return Feedback(kind="build", text=text, summary=summary[:400], n_total=len(errors),
                    n_shown=shown, per_unit=per_unit, units_with_errors=units,
                    mappable=bool(units))


@dataclass
class TestFailure:
    name: str            # classname.testname, as the baseline cache records it
    kind: str            # exception type
    message: str
    frames: list[str]


def parse_junit_failures(xml_paths: Iterable[str | Path]) -> tuple[int, list[TestFailure]]:
    """(tests run, failures) across JUnit XML reports."""
    ran, failures = 0, []
    for path in xml_paths:
        try:
            root = ET.parse(path).getroot()
        except (ET.ParseError, OSError):
            continue
        for tc in root.iter("testcase"):
            ran += 1
            node = tc.find("failure")
            if node is None:
                node = tc.find("error")
            if node is None:
                continue
            trace = (node.text or "").splitlines()
            frames = [ln.strip() for ln in trace if ln.strip().startswith("at ")
                      and not _NOISE_FRAME.match(ln)][:MAX_FRAMES]
            message = (node.get("message") or (trace[0] if trace else "")).strip()
            failures.append(TestFailure(
                name=f"{tc.get('classname')}.{tc.get('name')}",
                kind=node.get("type") or "", message=message, frames=frames))
    return ran, failures


def extract_test_feedback(
    failures: Sequence[TestFailure], baseline_failed: Iterable[str] = (),
    *, ran: int | None = None, gradle_log: str = "",
    max_tests: int = MAX_TESTS, max_chars: int = MAX_CHARS,
) -> Feedback:
    """Feedback from JUnit results: NEW failures only (not in the baseline of
    the unmodified fork), with assertion message and the top non-framework
    frames. Failures sharing message and frames (parameterized tests) are
    grouped."""
    base = set(baseline_failed)
    new = [f for f in failures if f.name not in base]
    if not new:
        text = ("The code compiled, but the test class did not run to completion "
                "with this change"
                + (f" ({ran} tests ran)" if ran is not None else "") + ".\n"
                + _fallback(gradle_log, max_chars - 200))
        return Feedback(kind="test", text=text, summary="tests did not run with the candidate")
    groups: dict[tuple, list[TestFailure]] = {}
    for f in new:
        groups.setdefault((f.kind, f.message, tuple(f.frames)), []).append(f)
    blocks = []
    for (kind, message, frames), members in groups.items():
        names = [m.name for m in members]
        head = f"- FAILED {names[0]}"
        if len(names) > 1:
            head += (f" (and {len(names) - 1} more with the same failure: "
                     + ", ".join(n.rsplit(".", 1)[-1] for n in names[1:4])
                     + (" ..." if len(names) > 4 else "") + ")")
        msg = message if len(message) <= 600 else message[:600] + " [...]"
        lines = [head, f"    {kind + ': ' if kind and kind not in msg else ''}{msg}"]
        lines.extend("      " + fr for fr in frames)
        blocks.append("\n".join(lines))
    text, shown = _cap(blocks, len(blocks), "distinct failures", max_tests, max_chars)
    text = ("The code compiled. These tests pass on the unmodified fork and FAIL "
            f"with the previous attempt applied ({len(new)} new failing test(s)):\n" + text)
    summary = f"{len(new)} new failing test(s); first: {new[0].name}: {new[0].message[:160]}"
    return Feedback(kind="test", text=text, summary=summary[:400], n_total=len(new),
                    n_shown=shown)
