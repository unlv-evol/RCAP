"""Co-changes: what the SAME upstream change does outside one processing unit.

A unit is one function. The upstream change usually touches more: a renamed
declaration in another file, a new field next to the function, a schema
entry, a caller. A candidate that is a faithful port of its own function can
still fail to build because a co-change is not in the fork, and then no edit
to that one function is a fix. This module reads the upstream change as a
unified diff and answers three questions, all deterministically:

* which diff hunks does a set of units cover, and which are left over
  (`classify_hunks`);
* does a name the build could not resolve appear on an added line of the
  change OUTSIDE the unit (`find_evidence`) — the co-change pre-check;
* what is the short, bounded list of co-changes a repair request shows, with
  each one marked applied or not (`cochange_notes`).

Pure text processing: no git, no build system. The driver supplies the diff
text, where each unit sits in the upstream files, and what is applied.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")
_IDENT = re.compile(r"[A-Za-z_$][\w$]*")
_DECL_KEYWORDS = frozenset({
    "if", "for", "while", "switch", "return", "new", "else", "try", "catch", "throw",
    "do", "synchronized", "assert", "case", "super", "this"})
_ACCESSOR = re.compile(r"^(?:get|set|is|has|with)(?=[A-Z_])")
MAX_NOTES = 10
MAX_NOTE_CHARS = 180
MAX_ALLOWED_SYMBOLS = 24


@dataclass
class DiffHunk:
    """One hunk of the upstream change."""

    hunk_id: str                 # "<new path or old path>@@<index>"
    old_path: str | None
    new_path: str | None
    status: str                  # A | M | D | R
    old_start: int
    new_start: int
    lines: list[str]             # body lines, with their ' ', '+', '-' prefix
    section: str = ""            # the text after the second @@
    covered_by: list[str] = field(default_factory=list)   # unit keys
    coverage: str = "none"       # full | partial | none
    members: list[str] = field(default_factory=list)      # ids merged into this hunk

    @property
    def path(self) -> str:
        return self.new_path or self.old_path or "?"

    @property
    def added(self) -> list[tuple[int, str]]:
        out, n = [], self.new_start
        for ln in self.lines:
            if ln.startswith("+"):
                out.append((n, ln[1:]))
                n += 1
            elif not ln.startswith("-") and not ln.startswith("\\"):
                n += 1
        return out

    @property
    def removed(self) -> list[tuple[int, str]]:
        out, n = [], self.old_start
        for ln in self.lines:
            if ln.startswith("-"):
                out.append((n, ln[1:]))
                n += 1
            elif not ln.startswith("+") and not ln.startswith("\\"):
                n += 1
        return out

    def patch(self, target_path: str) -> str:
        """A one-hunk patch against `target_path` (a new file keeps /dev/null)."""
        old_len = sum(1 for ln in self.lines if ln[:1] in (" ", "-"))
        new_len = sum(1 for ln in self.lines if ln[:1] in (" ", "+"))
        old = "/dev/null" if self.status == "A" else f"a/{target_path}"
        head = f"diff --git a/{target_path} b/{target_path}\n"
        if self.status == "A":
            head += "new file mode 100644\n"
        return (head + f"--- {old}\n+++ b/{target_path}\n"
                f"@@ -{self.old_start},{old_len} +{self.new_start},{new_len} @@\n"
                + "\n".join(self.lines) + "\n")


def _old_range(h: DiffHunk) -> tuple[int, int]:
    """0-based [start, end) of the old lines a zero-context hunk replaces; an
    insertion is the empty range at its insertion point."""
    removed = sum(1 for ln in h.lines if ln.startswith("-"))
    start = h.old_start - 1 if removed else h.old_start
    return start, start + removed


def with_context(groups: Sequence[DiffHunk], old_lines: Sequence[str],
                 n: int = 3) -> list[DiffHunk]:
    """Zero-context hunks of ONE file (`git diff -U0`: one hunk per contiguous
    change group, in order) -> hunks that can each be applied on their own, by
    context instead of by line number, with `n` lines of the OLD file around.

    Change groups are the granularity at which coverage is decided: a `-U3`
    hunk routinely holds a unit's edit AND a declaration next to it that no
    unit covers. Groups closer than `n` old lines are merged into one hunk
    (`members` lists them): applied separately, the first would change the
    context the second needs."""
    out: list[DiffHunk] = []
    cluster: list[DiffHunk] = []

    def flush() -> None:
        if not cluster:
            return
        first = cluster[0]
        if first.status == "A":
            out.extend(cluster)
            cluster.clear()
            return
        start, _ = _old_range(first)
        before = list(old_lines[max(0, start - n):start])
        lines = [" " + ln for ln in before]
        cursor = start
        for g in cluster:
            gs, ge = _old_range(g)
            lines += [" " + ln for ln in old_lines[cursor:gs]]
            lines += [ln for ln in g.lines if ln.startswith("-")]
            lines += [ln for ln in g.lines if ln.startswith("+")]
            cursor = ge
        lines += [" " + ln for ln in old_lines[cursor:cursor + n]]
        out.append(DiffHunk(
            hunk_id=first.hunk_id, old_path=first.old_path, new_path=first.new_path,
            status=first.status, old_start=start - len(before) + 1,
            new_start=max(1, first.new_start - len(before)), lines=lines,
            section=first.section, covered_by=[], coverage="none",
            members=[g.hunk_id for g in cluster]))
        cluster.clear()

    for g in groups:
        if cluster and _old_range(g)[0] - _old_range(cluster[-1])[1] >= n:
            flush()
        cluster.append(g)
    flush()
    return out


@dataclass
class FileDiff:
    old_path: str | None
    new_path: str | None
    status: str
    hunks: list[DiffHunk] = field(default_factory=list)
    binary: bool = False

    @property
    def path(self) -> str:
        return self.new_path or self.old_path or "?"


def parse_unified_diff(text: str) -> list[FileDiff]:
    """`git diff` output -> files and hunks. Renames keep both paths."""
    files: list[FileDiff] = []
    current: FileDiff | None = None
    hunk: DiffHunk | None = None
    for raw in text.splitlines():
        if raw.startswith("diff --git "):
            m = re.match(r"diff --git a/(.+?) b/(.+)$", raw)
            current = FileDiff(old_path=m.group(1) if m else None,
                               new_path=m.group(2) if m else None, status="M")
            files.append(current)
            hunk = None
            continue
        if current is None:
            continue
        if hunk is None:
            if raw.startswith("new file mode"):
                current.status, current.old_path = "A", None
            elif raw.startswith("deleted file mode"):
                current.status = "D"
            elif raw.startswith("rename from "):
                current.status, current.old_path = "R", raw[len("rename from "):]
            elif raw.startswith("rename to "):
                current.new_path = raw[len("rename to "):]
            elif raw.startswith(("Binary files ", "GIT binary patch")):
                current.binary = True
            elif raw.startswith("+++ /dev/null"):
                current.new_path = None
        m = _HUNK.match(raw)
        if m:
            hunk = DiffHunk(
                hunk_id=f"{current.path}@@{len(current.hunks) + 1}",
                old_path=current.old_path, new_path=current.new_path, status=current.status,
                old_start=int(m.group(1)), new_start=int(m.group(3)), lines=[],
                section=m.group(5).strip())
            current.hunks.append(hunk)
            continue
        if hunk is not None and raw[:1] in (" ", "+", "-", "\\"):
            hunk.lines.append(raw)
    return files


@dataclass
class UnitRegion:
    """Where one unit sits in the upstream files (1-based inclusive lines).
    A missing span falls back to line-text membership in the unit's text."""

    key: str
    source_file: str                              # upstream path after the change
    after_span: tuple[int, int] | None = None     # in the file after the change
    before_span: tuple[int, int] | None = None    # in the file before the change
    after_text: str = ""
    before_text: str = ""
    old_source_file: str | None = None

    def holds_added(self, path: str | None, line: int, text: str) -> bool:
        if path != self.source_file:
            return False
        if self.after_span is not None:
            return self.after_span[0] <= line <= self.after_span[1]
        return bool(text.strip()) and text.strip() in _stripped(self.after_text)

    def holds_removed(self, path: str | None, line: int, text: str) -> bool:
        if path not in (self.old_source_file or self.source_file, self.source_file):
            return False
        if self.before_span is not None:
            return self.before_span[0] <= line <= self.before_span[1]
        return bool(text.strip()) and text.strip() in _stripped(self.before_text)


def _stripped(text: str) -> set[str]:
    return {ln.strip() for ln in text.splitlines() if ln.strip()}


def locate(needle: str, haystack: str) -> tuple[int, int] | None:
    """1-based inclusive line span of `needle` in `haystack` when it occurs
    exactly once (leading/trailing blank space ignored), else None."""
    needle = needle.strip("\n")
    if not needle.strip():
        return None
    core = needle.strip()
    if haystack.count(core) != 1:
        return None
    pos = haystack.find(core)
    first = haystack.count("\n", 0, pos) + 1
    return first, first + core.count("\n")


def classify_hunks(files: Sequence[FileDiff], regions: Sequence[UnitRegion]) -> None:
    """Mark every hunk with the units that cover it: `full` when every changed
    (non-blank) line lies in a unit, `partial` when some do, else `none`."""
    for f in files:
        for h in f.hunks:
            changed = [(True, n, t) for n, t in h.added if t.strip()] \
                + [(False, n, t) for n, t in h.removed if t.strip()]
            keys: list[str] = []
            inside = 0
            for is_add, n, t in changed:
                owner = next((r.key for r in regions
                              if (r.holds_added(h.new_path, n, t) if is_add
                                  else r.holds_removed(h.old_path, n, t))), None)
                if owner is not None:
                    inside += 1
                    if owner not in keys:
                        keys.append(owner)
            h.covered_by = keys
            h.coverage = ("none" if not inside else
                          "full" if inside == len(changed) else "partial")


def is_declaration(line: str, name: str) -> bool:
    """Does `line` look like it DECLARES `name` (method, constructor, field,
    type, Scala def/val, or a JSON schema field), rather than using it?"""
    s = line.strip()
    if re.search(rf'"name"\s*:\s*"{re.escape(name)}"', s):
        return True
    if re.search(rf"\b(?:class|interface|enum|trait|object|record|def|val|var|type)\s+"
                 rf"{re.escape(name)}\b", s):
        return True
    m = re.search(rf"(?P<before>.*?)\b{re.escape(name)}\s*(?P<after>[(=;,)]|$)", s)
    if not m:
        return False
    before = m.group("before").rstrip()
    if not before:
        # `Name(args) {` : a constructor declaration; `name(args);` is a call
        return m.group("after") == "(" and s.rstrip().endswith("{")
    if before.endswith((".", "(", ",", "=", "!", "&", "|", "+", "-", "?", ":", "return",
                        "new", "->", "::")):
        return False
    last = before.split()[-1]
    if last in _DECL_KEYWORDS or not re.search(r"[\w>\]]$", last):
        return False
    if m.group("after") == "(":
        # a declaration ends in `{`, `;` (abstract/interface) or an open
        # parameter list; a call nested in a statement was excluded above
        return True
    return m.group("after") in ("=", ";") and bool(re.match(r"^[\w$.<>\[\], ?@()\"]+$", before))


def _aliases(name: str) -> list[str]:
    """Names under which a generated accessor's source may be declared:
    `setErrorCode` comes from a schema field `ErrorCode`."""
    out = [name]
    core = _ACCESSOR.sub("", name)
    if core != name:
        out.append(core)
        out.append(core[:1].lower() + core[1:])
    elif name[:1].islower():
        out.append(name[:1].upper() + name[1:])
    return out


@dataclass
class Evidence:
    """A missing name found on an added line of the change outside the unit."""

    symbol: str
    file: str
    line: int
    text: str
    declaration: bool
    hunk_id: str
    covered_by: list[str]
    matched_as: str
    owner: str | None = None   # the unit whose region holds this very line


def find_evidence(names: Iterable[str], files: Sequence[FileDiff],
                  region: UnitRegion | None, *, limit: int = 3,
                  all_regions: Sequence[UnitRegion] = ()) -> dict[str, list[Evidence]]:
    """For each missing name: added lines of the upstream change, outside the
    unit, that mention it. Declarations first; when at least one declaration
    is found, usages are dropped (a usage is not what the unit is missing)."""
    out: dict[str, list[Evidence]] = {}
    for name in dict.fromkeys(names):
        hits: list[Evidence] = []
        for f in files:
            if f.binary:
                continue
            is_schema = (f.path or "").endswith(".json")
            for h in f.hunks:
                for n, text in h.added:
                    if region is not None and region.holds_added(h.new_path, n, text):
                        continue
                    for alias in _aliases(name):
                        if alias != name and not is_schema:
                            continue
                        if not re.search(rf"(?<![\w$]){re.escape(alias)}(?![\w$])", text):
                            continue
                        hits.append(Evidence(
                            symbol=name, file=f.path, line=n, text=text.strip()[:200],
                            declaration=is_declaration(text, alias), hunk_id=h.hunk_id,
                            covered_by=list(h.covered_by), matched_as=alias,
                            owner=next((r.key for r in all_regions
                                        if r.holds_added(h.new_path, n, text)), None)))
                        break
        decls = [e for e in hits if e.declaration]
        chosen = decls or hits
        if chosen:
            out[name] = chosen[:limit]
    return out


def declared_names(lines: Iterable[str]) -> list[str]:
    """Names declared on the given source lines (best effort, ordered)."""
    out: list[str] = []
    for line in lines:
        m = re.search(r'"name"\s*:\s*"([A-Za-z_$][\w$]*)"', line)
        candidates = [m.group(1)] if m else []
        for ident in _IDENT.findall(line):
            if ident not in candidates and ident not in _DECL_KEYWORDS \
                    and is_declaration(line, ident):
                candidates.append(ident)
        for c in candidates:
            if c not in out:
                out.append(c)
    return out


def summarize_hunk(h: DiffHunk) -> str:
    """One line that says what a hunk does: its first declaration-like added
    line, else its first added line, else its first removed line."""
    added = [t.strip() for _, t in h.added if t.strip() and not t.strip().startswith(("//", "*", "/*"))]
    removed = [t.strip() for _, t in h.removed if t.strip()]
    pick = next((t for t in added if declared_names([t])), None) or (added[0] if added else None)
    if pick is not None:
        text = "adds `" + pick[:MAX_NOTE_CHARS] + "`"
    elif removed:
        text = "removes `" + removed[0][:MAX_NOTE_CHARS] + "`"
    else:
        text = "whitespace/comment change"
    if h.status == "A":
        text = "new file; " + text
    return text


@dataclass
class CoChangeNote:
    status: str          # APPLIED | NOT APPLIED
    path: str
    summary: str
    symbols: list[str] = field(default_factory=list)
    hunk_id: str = ""
    why: str = ""        # recorded, not rendered


def _related(a: str, b: str) -> bool:
    """A class and its test (or two files of one type family) by file stem."""
    sa, sb = a.rsplit("/", 1)[-1].split(".")[0], b.rsplit("/", 1)[-1].split(".")[0]
    return bool(sa and sb) and (sa.startswith(sb) or sb.startswith(sa))


def cochange_notes(
    files: Sequence[FileDiff], region: UnitRegion | None,
    applied: Callable[[DiffHunk], tuple[bool, str]],
    *, priority_hunks: Iterable[str] = (), max_notes: int = MAX_NOTES,
    in_fork: Callable[[FileDiff], bool] | None = None,
) -> tuple[list[CoChangeNote], int]:
    """The co-changes of the upstream change as one unit's request lists them:
    hunks with a change outside the unit, marked APPLIED / NOT APPLIED by the
    driver's `applied(hunk)`.

    Minimal and localised rather than complete: hunks that explain a missing
    name come first, then the unit's own file, then files related by name (a
    class and its test), each hunk on its own line. Every other file gets ONE
    line per status. Files that do not exist in the fork (`in_fork`) are not
    listed at all, only counted. Bounded by `max_notes`. Returns (notes, number
    of hunks not listed)."""
    first = list(priority_hunks)
    ranked: list[tuple[tuple[int, int], CoChangeNote]] = []
    grouped: dict[tuple[str, str], list[tuple[DiffHunk, list[str], str]]] = {}
    order = hidden = 0
    for f in files:
        if f.binary:
            continue
        for h in f.hunks:
            outside_added = [t for n, t in h.added if t.strip()
                             and not (region and region.holds_added(h.new_path, n, t))]
            outside_removed = [t for n, t in h.removed if t.strip()
                               and not (region and region.holds_removed(h.old_path, n, t))]
            if not outside_added and not outside_removed:
                continue
            ok, why = applied(h)
            status = "APPLIED" if ok else "NOT APPLIED"
            rank = (0 if h.hunk_id in first else
                    1 if region is not None and h.path == region.source_file else
                    2 if region is not None and _related(h.path, region.source_file) else 3)
            order += 1
            if rank < 3:
                ranked.append(((rank, order), CoChangeNote(
                    status=status, path=h.path, summary=summarize_hunk(h),
                    symbols=declared_names(outside_added)[:6], hunk_id=h.hunk_id, why=why)))
            elif not ok and in_fork is not None and not in_fork(f):
                hidden += 1          # nothing the fork has, nothing applied: noise
            else:
                grouped.setdefault((h.path, status), []).append((h, outside_added, why))
    for (path, status), items in grouped.items():
        symbols: list[str] = []
        for _, added, _ in items:
            for sym in declared_names(added):
                if sym not in symbols:
                    symbols.append(sym)
        summary = summarize_hunk(items[0][0])
        if len(items) > 1:
            summary = (f"{len(items)} hunks; declares " + ", ".join(symbols[:6]) if symbols
                       else f"{len(items)} hunks; first {summary}")
        order += 1
        ranked.append(((3 if status == "APPLIED" else 4, order), CoChangeNote(
            status=status, path=path, summary=summary, symbols=symbols[:6],
            hunk_id=items[0][0].hunk_id, why=items[0][2])))
    ranked.sort(key=lambda x: x[0])
    kept = [n for _, n in ranked[:max_notes]]
    return kept, len(ranked) - len(kept) + hidden


def allowed_symbols(notes: Sequence[CoChangeNote], limit: int = MAX_ALLOWED_SYMBOLS) -> list[str]:
    out: list[str] = []
    for n in notes:
        if n.status == "APPLIED":
            for s in n.symbols:
                if s not in out:
                    out.append(s)
    return out[:limit]
