"""Merging the candidates of several units into one file.

RCAP may send several units that cover the same region of one file: a method
unit inside a class-level unit, or a whole-file transformation next to its
functions. Splicing them one after the other by searching for each unit's
original text fails for every unit after the first, because the first
replacement has already changed the text the later ones look for ("original
text found 0 times"), and those candidates were dropped.

Here every unit is an edit against the ORIGINAL file:

1. each unit's original text is located once, in the original file, which
   gives it a fixed character span;
2. each candidate is diffed (by line) against its own original; every changed
   block becomes a hunk with offsets in the original file;
3. hunks of all units are merged. Identical hunks (an outer unit that also
   ports the inner unit's change) count once. Two different hunks that touch
   the same original text are a true conflict: the hunk of the more specific
   unit (the smaller span; the inner one of a nested pair) wins, the other is
   left out and recorded;
4. the surviving hunks are applied in one pass.

Nothing is dropped silently: every unit ends `merged`, `conflict` (with each
lost hunk and the unit it lost to) or `unspliceable` (with the reason).
Pure text processing; no parser, no build system.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher

MERGED = "merged"
CONFLICT = "conflict"
UNSPLICEABLE = "unspliceable"


@dataclass(frozen=True)
class UnitEdit:
    """One unit's candidate: `original` is its text in the unmodified file."""

    key: str
    original: str
    candidate: str
    # Higher wins a conflict regardless of span size. The PR-level driver gives
    # mechanically applied upstream hunks -1: a unit the model ported
    # deliberately is never overridden by text nobody adapted.
    priority: int = 0


@dataclass
class Hunk:
    """One changed block of one unit, in offsets of the original file."""

    unit: str
    start: int
    end: int
    text: str                       # replacement for original[start:end]
    owners: list[str] = field(default_factory=list)   # every unit with this very hunk

    def describe(self, original: str) -> dict[str, object]:
        return {"line": original.count("\n", 0, self.start) + 1,
                "removed": original[self.start:self.end][:200],
                "added": self.text[:200]}


@dataclass
class UnitMerge:
    key: str
    status: str                     # merged | conflict | unspliceable
    reason: str = ""
    span: tuple[int, int] | None = None        # offsets in the original file
    new_span: tuple[int, int] | None = None    # offsets in the merged file
    hunks: int = 0                  # changed blocks in this unit's candidate
    applied: int = 0                # of them in the merged file (own or identical)
    shared: int = 0                 # of the applied ones, identical to another unit's hunk
    lost: list[dict[str, object]] = field(default_factory=list)
    won_over: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {"status": self.status, "reason": self.reason, "hunks": self.hunks,
                "applied": self.applied, "shared": self.shared, "lost": self.lost,
                "won_over": self.won_over}


@dataclass
class MergeResult:
    text: str
    units: dict[str, UnitMerge]
    applied: list[Hunk]             # in file order
    original: str

    def lines_of(self, key: str) -> tuple[int, int] | None:
        """1-based inclusive line span of the unit in the merged text."""
        u = self.units.get(key)
        if u is None or u.new_span is None:
            return None
        s, e = u.new_span
        first = self.text.count("\n", 0, s) + 1
        return first, first + self.text[s:e].rstrip("\n").count("\n")

    def hunk_lines(self) -> list[tuple[int, int, list[str]]]:
        """(first, last, owners) of every applied hunk in the merged text
        (1-based inclusive; a pure deletion is the line it was removed at)."""
        out, shift = [], 0
        for h in self.applied:
            s = h.start + shift
            first = self.text.count("\n", 0, s) + 1
            out.append((first, first + max(0, h.text.rstrip("\n").count("\n")), list(h.owners)))
            shift += len(h.text) - (h.end - h.start)
        return out

    @property
    def changed(self) -> bool:
        return self.text != self.original


def _align_trailing_newline(original: str, candidate: str) -> str:
    """A candidate always ends in a newline; its original may not. Without this
    the last line of every such unit would be a spurious hunk."""
    if original.endswith("\n") or not candidate.endswith("\n"):
        return candidate
    return candidate[:-1]


def unit_hunks(key: str, original: str, candidate: str, offset: int) -> list[Hunk]:
    """The changed line blocks of original -> candidate, at file offsets."""
    candidate = _align_trailing_newline(original, candidate)
    a, b = original.splitlines(keepends=True), candidate.splitlines(keepends=True)
    starts = [0]
    for line in a:
        starts.append(starts[-1] + len(line))
    out: list[Hunk] = []
    for op, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        out.append(Hunk(key, offset + starts[i1], offset + starts[i2], "".join(b[j1:j2]), [key]))
    return out


def _overlap(a: Hunk, b: Hunk) -> bool:
    a_ins, b_ins = a.start == a.end, b.start == b.end
    if a_ins and b_ins:
        return a.start == b.start
    if a_ins:
        return b.start < a.start < b.end
    if b_ins:
        return a.start < b.start < a.end
    return a.start < b.end and b.start < a.end


def merge_units(original: str, edits: Sequence[UnitEdit]) -> MergeResult:
    """Apply every unit's candidate to `original` (see the module docstring).

    Units are given in application order; that order only breaks ties between
    two units with the same span (the earlier one wins)."""
    units: dict[str, UnitMerge] = {}
    order: dict[str, int] = {}
    hunks: dict[str, list[Hunk]] = {}
    for n, e in enumerate(edits):
        order[e.key] = n
        found = original.count(e.original) if e.original else 0
        if found != 1:
            why = ("empty original text" if not e.original else
                   f"original text found {found} times in the unmodified file")
            units[e.key] = UnitMerge(e.key, UNSPLICEABLE, why)
            continue
        pos = original.find(e.original)
        um = UnitMerge(e.key, MERGED, span=(pos, pos + len(e.original)))
        hunks[e.key] = unit_hunks(e.key, e.original, e.candidate, pos)
        um.hunks = len(hunks[e.key])
        if not um.hunks:
            um.reason = "candidate equals the original text: nothing to apply"
        units[e.key] = um

    # most specific unit first: its hunks are accepted before an enclosing unit's
    prio = {e.key: e.priority for e in edits}
    ranked = sorted(hunks, key=lambda k: (-prio[k], units[k].span[1] - units[k].span[0], order[k]))
    accepted: list[Hunk] = []
    for key in ranked:
        um = units[key]
        for h in hunks[key]:
            same = next((x for x in accepted
                         if (x.start, x.end, x.text) == (h.start, h.end, h.text)), None)
            if same is not None:
                same.owners.append(key)
                um.applied += 1
                um.shared += 1
                continue
            clash = [x for x in accepted if x.unit != key and _overlap(x, h)]
            if clash:
                winners = sorted({o for x in clash for o in x.owners})
                um.lost.append({**h.describe(original), "lost_to": winners,
                                "kept": [x.text[:200] for x in clash][:3]})
                for w in winners:
                    if key not in units[w].won_over:
                        units[w].won_over.append(key)
                continue
            accepted.append(h)
            um.applied += 1
        if um.lost:
            um.status = CONFLICT
            um.reason = (f"{len(um.lost)} of {um.hunks} hunk(s) left out: they change text that "
                         f"{', '.join(sorted({w for x in um.lost for w in x['lost_to']}))} "
                         "(a more specific or higher-priority change) changes differently")

    accepted.sort(key=lambda h: (h.start, h.end))
    parts: list[str] = []
    cursor = 0
    for h in accepted:
        parts.append(original[cursor:h.start])
        parts.append(h.text)
        cursor = h.end
    parts.append(original[cursor:])
    text = "".join(parts)

    for um in units.values():
        if um.span is None:
            continue
        s, e = um.span
        ds = sum(len(h.text) - (h.end - h.start) for h in accepted if h.end <= s and h.start < s)
        de = sum(len(h.text) - (h.end - h.start) for h in accepted
                 if h.start < e or (h.start == e and key_in(h, um.key)))
        um.new_span = (s + ds, e + de)
    return MergeResult(text=text, units=units, applied=accepted, original=original)


def key_in(h: Hunk, key: str) -> bool:
    return key in h.owners
