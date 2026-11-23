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
    # PR-level attribution (attribute_errors): repo-relative path, the unit /
    # mechanical-hunk keys the error is charged to, and how that was decided
    # (span | name | file | owner), None when it could not be attributed.
    rel: str | None = None
    keys: list[str] = field(default_factory=list)
    how: str | None = None


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


def failing_tests_intro(n: int, *, unit_is_test: bool = False,
                        pr_context: bool = False) -> str:
    """The sentence that introduces test failures in a repair request.

    It must not cue a revert. "These tests pass on the unmodified fork and
    fail with your attempt" does, and it is simply wrong when the function
    under repair IS the test: a ported test fails because the behaviour it now
    checks is missing or the port is off, not because the old test was right."""
    applied = ("the previous attempt applied together with the applied co-changes"
               if pr_context else "the previous attempt applied")
    if unit_is_test:
        return (
            f"The code compiled. The function being ported is TEST code, and with {applied} "
            f"{n} test(s) of its class fail that did not fail before the port. The ported "
            "test checks the behaviour the source change introduces, so a failure means "
            "either the test was ported incorrectly for this fork's API, or the production "
            "change it checks is not applied in the fork (then answer CANNOT_ADAPT). "
            "Restoring the old assertions is not a fix:")
    return (
        f"The code compiled. With {applied}, {n} test(s) fail that did not fail before the "
        "port. A new failure means the ported change is incomplete or wrong in this "
        "function; it is not a reason to undo the change:")


@dataclass
class TestFailure:
    __test__ = False     # not a pytest test class, despite the name

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
    unit_is_test: bool = False, pr_context: bool = False,
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
    text = failing_tests_intro(len(new), unit_is_test=unit_is_test,
                               pr_context=pr_context) + "\n" + text
    summary = f"{len(new)} new failing test(s); first: {new[0].name}: {new[0].message[:160]}"
    return Feedback(kind="test", text=text, summary=summary[:400], n_total=len(new),
                    n_shown=shown)


# ---------------------------------------------------------------------------
# PR-level validation: several edited files, several units per file, sibling
# changes applied together. Everything below is pure text processing too.
# ---------------------------------------------------------------------------

_IDENT = re.compile(r"[A-Za-z_$][\w$]*")
_JAVAC_SYMBOL = re.compile(r"^\s*symbol:\s+(?P<kind>\w+)\s+(?P<name>[A-Za-z_$][\w$]*)")
_JAVAC_LOCATION = re.compile(
    r"^\s*location:\s+(?:(?:variable|field|parameter)\s+\S+\s+of type\s+|"
    r"(?:class|interface|enum|package)\s+)(?P<owner>[\w.$]+)")
_MISSING_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"^(?:method|constructor) (?P<name>[\w$]+) in (?:class|interface|enum) "
                r"(?P<owner>[\w.$]+?)(?:<.*>)? cannot be applied"), "signature"),
    (re.compile(r"^no suitable (?:method|constructor) found for (?P<name>[\w$]+)\("), "signature"),
    (re.compile(r"does not override abstract method (?P<name>[\w$]+)\(.*\) in (?P<owner>[\w.$]+)"),
     "abstract"),
    (re.compile(r"^value (?P<name>[\w$]+) is not a member of (?P<owner>[\w.$\[\], ]+)"), "member"),
    (re.compile(r"^not found: (?:value|type|object) (?P<name>[\w$]+)"), "symbol"),
    (re.compile(r"^(?:too many|not enough) arguments (?:\(.*?\) )?for "
                r"(?:method|constructor) (?P<name>[\w$]+)"), "signature"),
    (re.compile(r"^overloaded method (?:value )?(?P<name>[\w$]+) with alternatives"), "signature"),
    (re.compile(r"^(?:method|value) (?P<name>[\w$]+) overrides nothing"), "override"),
)


@dataclass
class MissingSymbol:
    """A name a diagnostic says the build could not resolve (or could not
    apply): the unit of evidence for the co-change pre-check."""

    name: str
    kind: str                 # method | variable | class | signature | member | symbol | ...
    owner: str | None = None  # receiver / enclosing type (`Outer.Inner`), when printed
    path: str | None = None
    line: int | None = None


def _simple(name: str | None) -> str | None:
    """`a.b.Outer.Inner<T>` -> `Outer.Inner`: the type names, package dropped."""
    if not name:
        return None
    name = re.sub(r"<.*", "", name).strip().rstrip("[]").replace("$", ".")
    types = [p for p in name.split(".") if p[:1].isupper()]
    return ".".join(types) or name.rsplit(".", 1)[-1] or None


def owner_types(owner: str | None) -> list[str]:
    """Type names of an owner, innermost first (`Outer.Inner` -> Inner, Outer)."""
    return list(reversed((owner or "").split("."))) if owner else []


def missing_symbols(errors: Iterable[CompileError]) -> list[MissingSymbol]:
    """Names the compiler could not resolve, de-duplicated by (name, owner).

    javac: `cannot find symbol` + `symbol:` / `location:` lines, signature
    mismatches, missing overrides. scalac: `value x is not a member of T`,
    `not found: value x`, arity mismatches. A bare `cannot find symbol` (zinc
    prints no symbol line) falls back to the identifier under the caret."""
    out: list[MissingSymbol] = []
    seen: set[tuple[str, str | None]] = set()

    def add(name: str, kind: str, owner: str | None, e: CompileError) -> None:
        owner = _simple(owner)
        if (name, owner) not in seen:
            seen.add((name, owner))
            out.append(MissingSymbol(name=name, kind=kind, owner=owner,
                                     path=e.rel or e.path, line=e.line))

    for e in errors:
        msg = e.message.strip()
        owner = next((m.group("owner") for b in e.body
                      if (m := _JAVAC_LOCATION.match(b))), None)
        if msg.startswith("cannot find symbol"):
            sym = next((m for b in e.body if (m := _JAVAC_SYMBOL.match(b))), None)
            if sym:
                add(sym.group("name"), sym.group("kind"), owner, e)
            elif e.col and e.body:
                # javac's column is the member-select dot for `a.name`, else the name
                m = re.match(r"\.?\s*([A-Za-z_$][\w$]*)", e.body[0][e.col - 1:]) \
                    if len(e.body[0]) >= e.col else None
                if m:
                    add(m.group(1), "symbol", owner, e)
            continue
        for pattern, kind in _MISSING_PATTERNS:
            m = pattern.search(msg)
            if m:
                add(m.group("name"), kind, m.groupdict().get("owner") or owner, e)
                break
        else:
            if "does not override or implement a method from a supertype" in msg:
                # the offending declaration follows the @Override the caret is on
                for b in e.body[:3]:
                    m = re.search(r"([A-Za-z_$][\w$]*)\s*\(", b)
                    if m:
                        add(m.group(1), "override", owner, e)
                        break
    return out


@dataclass
class Span:
    """Lines of one applied change inside a spliced file."""

    key: str
    first: int
    last: int
    kind: str = "unit"            # unit | mech
    names: tuple[str, ...] = ()   # names this change declares (by-name attribution)


def attribute_errors(
    errors: list[CompileError], spans_by_file: dict[str, list[Span]],
    *, root: str | None = None, sources: dict[str, str] | None = None,
    unit_line: Callable[[str, int], int | None] | None = None,
) -> list[CompileError]:
    """Charge each compile error to the applied change(s) responsible.

    In order: the span that contains the error line (`span`); spans anywhere
    that declare a name the error could not resolve or apply (`name`, a caller
    that no longer matches a ported signature); every span of the file the
    error is in (`file`); every span of the file named after the error's
    `location:` type (`owner`). Anything else stays unattributed and is
    reported as such, never guessed. Returns the de-duplicated errors."""
    prefix = (root.rstrip("/") + "/") if root else None
    for e in errors:
        e.rel = e.path[len(prefix):] if prefix and e.path.startswith(prefix) else e.path
        spans = spans_by_file.get(e.rel, [])
        e.in_edited_file = e.rel in spans_by_file
        src = (sources or {}).get(e.rel)
        if src is not None and not e.body:
            lines = src.splitlines()
            if 0 < e.line <= len(lines):
                e.body = [lines[e.line - 1]]
                if e.col:
                    e.body.append(" " * (e.col - 1) + "^")
        hit = next((s for s in spans if s.first <= e.line <= s.last), None)
        if hit is not None:
            e.keys, e.how = [hit.key], "span"
            if hit.kind == "unit":
                n = e.line - hit.first + 1
                e.unit_line = unit_line(hit.key, n) if unit_line else n
    errors = _dedup(errors)
    for e in errors:
        if e.keys:
            continue
        missing = missing_symbols([e])
        names = {m.name for m in missing}
        by_name = [s.key for ss in spans_by_file.values() for s in ss if names & set(s.names)]
        if by_name:
            e.keys, e.how = sorted(set(by_name)), "name"
            continue
        if e.rel in spans_by_file and spans_by_file[e.rel]:
            # No evidence favours any one change of this file. Mechanically
            # applied hunks are charged before model-written units: nobody
            # adapted them, and charging a unit means asking the model to
            # "fix" a function that may be fine.
            here = spans_by_file[e.rel]
            mech = [s for s in here if s.kind == "mech"]
            e.keys, e.how = [s.key for s in (mech or here)], "file"
            continue
        owners = {t for m in missing for t in owner_types(m.owner)}
        by_owner = [s.key for f, ss in spans_by_file.items() for s in ss
                    if Path(f).stem in owners]
        if by_owner:
            e.keys, e.how = sorted(set(by_owner)), "owner"
    return errors


def render_errors_for(key: str, errors: Sequence[CompileError], file_of_key: str,
                      *, max_errors: int = MAX_ERRORS, max_chars: int = MAX_CHARS) -> str:
    """The diagnostics charged to one unit, as that unit's request shows them:
    its own lines first, located relative to the function the model wrote."""
    mine = [e for e in errors if key in e.keys]
    mine.sort(key=lambda e: 0 if e.how == "span" else 1 if e.rel == file_of_key else 2)
    blocks = []
    for e in mine:
        where = f"{e.rel}:{e.line}"
        if e.how == "span":
            where += (f" [line {e.unit_line} of your function]" if e.unit_line is not None
                      else " [inside your function]")
        elif e.rel == file_of_key:
            where += " [in the edited file, outside your function]"
        else:
            where += " [in another file that uses the edited one]"
        out = [f"- {where}: {e.message}"]
        out.extend("    " + b for b in e.body)
        if e.also_at:
            out.append(f"    (same error also at lines {', '.join(map(str, e.also_at[:10]))})")
        blocks.append("\n".join(out))
    text, _ = _cap(blocks, len(blocks), "errors", max_errors, max_chars)
    if any(e.how != "span" for e in mine):
        text += "\n" + OUTSIDE_NOTE
    return text


OUTSIDE_NOTE = (
    "Note: errors outside your function are in code that uses it and that this request "
    "cannot change. Keep the function compatible with that code, or, if that is "
    "impossible without changing it, answer CANNOT_ADAPT.")

_FRAME = re.compile(r"at ([\w.$]+)\.([\w$<>]+)\(([\w$]+\.(?:java|scala)):(\d+)\)")


def render_test_failures(failures: Sequence[TestFailure], *, unit_is_test: bool = False,
                         pr_context: bool = False, max_tests: int = MAX_TESTS,
                         max_chars: int = MAX_CHARS) -> str:
    """Rendered NEW test failures (the caller has already removed baseline
    ones), grouped as in `extract_test_feedback`."""
    return extract_test_feedback(failures, (), max_tests=max_tests, max_chars=max_chars,
                                 unit_is_test=unit_is_test, pr_context=pr_context).text


def attribute_test_failures(
    failures: Sequence[TestFailure], spans_by_file: dict[str, list[Span]],
    *, test_file_of_class: dict[str, str], mapped_files: dict[str, list[str]],
    sources: dict[str, str] | None = None,
) -> dict[str, list[tuple[TestFailure, str]]]:
    """Charge each NEW test failure to applied changes: key -> [(failure, how)].

    `frame`: a stack frame lies inside the change's span. `own_test`: the
    change contains the failing test method itself. `mapped`: the failing
    class is the test class of the file the change is in — used only when
    neither of the first two located the failure.
    `test_file_of_class`: FQCN -> edited test file; `mapped_files`: FQCN ->
    edited files whose test class it is."""
    out: dict[str, list[tuple[TestFailure, str]]] = {}
    by_base: dict[str, list[str]] = {}
    for f in spans_by_file:
        by_base.setdefault(Path(f).name, []).append(f)

    def charge(key: str, failure: TestFailure, how: str) -> None:
        rows = out.setdefault(key, [])
        if all(r[0] is not failure for r in rows):
            rows.append((failure, how))

    for failure in failures:
        cls = failure.name.split(".", 1)[0] if "." not in failure.name else None
        cls = next((c for c in sorted(set(test_file_of_class) | set(mapped_files),
                                      key=len, reverse=True)
                    if failure.name.startswith(c + ".")), cls)
        narrowed: set[str] = set()
        for frame in failure.frames:
            m = _FRAME.search(frame)
            if not m:
                continue
            files = by_base.get(m.group(3), [])
            if len(files) != 1:
                continue
            line = int(m.group(4))
            for s in spans_by_file[files[0]]:
                if s.first <= line <= s.last:
                    charge(s.key, failure, "frame")
                    narrowed.add(files[0])
        test_file = test_file_of_class.get(cls or "")
        if test_file and test_file not in narrowed:
            method = failure.name[len(cls) + 1:] if cls else failure.name
            method = re.sub(r"\(.*$", "", method).strip()
            src = (sources or {}).get(test_file, "")
            if re.fullmatch(r"[A-Za-z_$][\w$]*", method or "") and src:
                for m in re.finditer(rf"\b{re.escape(method)}\s*\(", src):
                    line = src.count("\n", 0, m.start()) + 1
                    hit = next((s for s in spans_by_file.get(test_file, [])
                                if s.first <= line <= s.last), None)
                    if hit is not None:
                        charge(hit.key, failure, "own_test")
                        narrowed.add(test_file)
                        break
        if narrowed:
            # The failure is already located (a frame, or the failing test
            # method is itself a ported unit): do not also charge every change
            # of the production file, which may be perfectly fine.
            continue
        for f in mapped_files.get(cls or "", []):
            for s in spans_by_file.get(f, []):
                charge(s.key, failure, "mapped")
    return out


# --- fork-side context: what the receiver type actually offers ---------------

_JAVA_MEMBER = re.compile(
    r"^\s*(?:@\w+(?:\([^)]*\))?\s+)*"
    r"(?:(?:public|protected|private|static|final|abstract|synchronized|default|native)\s+)*"
    r"(?:<[^>]+>\s+)?[\w$.<>\[\], ?&]+?\s+(?P<name>[A-Za-z_$][\w$]*)\s*\([^;{]*\)?[^;{]*[{;]?\s*$")
_JAVA_FIELD = re.compile(
    r"^\s*(?:public|protected|private|static)\s+(?:(?:static|final|volatile|transient)\s+)*"
    r"[\w$.<>\[\], ?]+?\s+(?P<name>[A-Za-z_$][\w$]*)\s*(?:=.*)?;\s*$")
_SCALA_MEMBER = re.compile(
    r"^\s*(?:@\w+(?:\([^)]*\))?\s+)*(?:(?:override|private(?:\[\w+\])?|protected(?:\[\w+\])?|"
    r"final|implicit|lazy|abstract)\s+)*(?:def|val|var)\s+(?P<name>[A-Za-z_$][\w$]*)")
_NOT_MEMBER = re.compile(r"^\s*(?:if|for|while|switch|return|new|else|try|catch|throw|do|"
                         r"synchronized|assert|case)\b")


def member_signatures(source: str, language: str = "java") -> list[tuple[str, str]]:
    """(name, declaration line) of the members a source file declares. A
    line-level heuristic: enough to show which names exist, not a parser."""
    out: list[tuple[str, str]] = []
    for raw in source.splitlines():
        if _NOT_MEMBER.match(raw) or raw.lstrip().startswith(("//", "*", "/*")):
            continue
        if language in ("scala", "sc"):
            m = _SCALA_MEMBER.match(raw)
        else:
            m = None
            head = raw.split("(", 1)[0]
            tokens = head.split()
            if "(" in raw and "=" not in head and len(tokens) >= 2 and "." not in tokens[-1]:
                m = _JAVA_MEMBER.match(raw)
            if m is None:
                m = _JAVA_FIELD.match(raw)
        if m:
            sig = _WS_RUN.sub(" ", raw).strip().rstrip("{").strip()
            out.append((m.group("name"), sig[:200]))
    return out


_WS_RUN = re.compile(r"\s+")
_ACCESSOR = re.compile(r"^(?:get|set|is|with|has)(?=[A-Z_])")


def rank_members(name: str, members: Sequence[tuple[str, str]], limit: int = 8,
                 *, floor: float = 0.5) -> list[str]:
    """Declarations whose name is closest to a missing `name` (a renamed
    accessor, a differently spelled method), best first."""
    import difflib

    def core(n: str) -> str:
        return _ACCESSOR.sub("", n).lower()

    scored = []
    for member, sig in members:
        a, b = core(name), core(member)
        score = difflib.SequenceMatcher(a=a, b=b).ratio()
        if a and b and (a in b or b in a):
            score = max(score, 0.9)
        if member == name:
            score = 1.0
        if score >= floor:
            scored.append((score, sig))
    scored.sort(key=lambda x: -x[0])
    out: list[str] = []
    for _, sig in scored:
        if sig not in out:
            out.append(sig)
    return out[:limit]
