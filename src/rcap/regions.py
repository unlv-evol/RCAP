"""Reducing a whole-file unit to the regions its change touches.

A unit whose target payload is the fork's whole file was answered by the
whole file (rcap-request-whole-v1): the model had to write every line back to
change two of them. That costs one output token per token of the file, so a
file of ~25k characters already needs half the context window for its answer,
files past ~45k characters overflow the window while the answer is being
written (the runtime shifts the context and keeps generating until the client
times out), and what does come back is thousands of lines the model merely
copied, any of which it may have altered.

Most such units change a few lines: the source payloads are diff excerpts
(one `@@` hunk with its context lines), or one function. This module maps
each source hunk to the span of the fork's file it corresponds to, so the
request can show, and ask back, only that span:

1. `source_hunks`: the unit's change as hunks (before lines / after lines).
   Diff excerpts are split at their `@@` headers; a before/after pair of one
   function is diffed with three lines of context;
2. `locate`: a hunk is anchored in the fork's text by the lines of its BEFORE
   side (context and removed lines) that occur exactly once in the fork's
   text and once in the hunk, kept as an order-preserving chain. A line
   counts only when it can identify a place (two identifiers, or one long
   one: never a brace or a blank). No anchor, or anchors scattered over the
   file, means "not located": never guessed;
3. `plan_regions`: every hunk located -> regions (the anchored span plus
   `context` lines on both sides, whole lines, overlapping ones joined), each
   with an exact character span whose text occurs once in the target. One
   hunk not located, too many regions, or regions that cover most of the file
   anyway -> no plan, and the unit keeps the whole-file request.

Answers are merged back with rcap.splice (`merge_regions`). Pure text.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field

from rcap.splice import MERGED, UnitEdit, merge_units

_HEADER = re.compile(r"^\s*@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@")
_WORD = re.compile(r"[A-Za-z_$][\w$]*")

CONTEXT_LINES = 6        # fork lines shown (and returned) around the anchored span
MAX_REGIONS = 8          # more than this is not "a few regions"
MAX_FRACTION = 0.6       # regions covering more of the file than this: ask for the file
MAX_SPREAD = 40          # fork lines the anchors of one hunk may span beyond the hunk's own length


@dataclass
class SourceHunk:
    header: str                  # the `@@ ... @@` line of a diff excerpt, "" for a function diff
    before: list[str]
    after: list[str]

    def before_text(self) -> str:
        return (self.header + "\n" if self.header else "") + "".join(x + "\n" for x in self.before)

    def after_text(self) -> str:
        return (self.header + "\n" if self.header else "") + "".join(x + "\n" for x in self.after)


@dataclass
class Region:
    start: int                   # character span in the target text (whole lines)
    end: int
    first_line: int              # 1-based, inclusive, in the target text
    last_line: int
    text: str
    hunks: list[int] = field(default_factory=list)     # indexes into the unit's source hunks
    anchors: int = 0

    def as_dict(self) -> dict[str, object]:
        return {"start": self.start, "end": self.end, "lines": [self.first_line, self.last_line],
                "hunks": self.hunks, "anchors": self.anchors}


@dataclass
class RegionPlan:
    regions: list[Region]
    hunks: list[SourceHunk]
    reason: str = ""             # why there is no plan (regions empty)

    @property
    def located(self) -> bool:
        return bool(self.regions)

    def as_dict(self) -> dict[str, object]:
        return {"located": self.located, "reason": self.reason, "hunks": len(self.hunks),
                "regions": [r.as_dict() for r in self.regions]}


def _split_excerpts(text: str) -> list[tuple[str, list[str]]] | None:
    out: list[tuple[str, list[str]]] = []
    for line in text.splitlines():
        if _HEADER.match(line):
            out.append((line.strip(), []))
        elif out:
            out[-1][1].append(line)
        elif line.strip():
            return None              # text before the first header: not diff excerpts
    return out or None


def source_hunks(before: str | None, after: str | None) -> list[SourceHunk] | None:
    """The unit's change as hunks, or None when the payloads are neither diff
    excerpts with matching headers nor a before/after pair of text."""
    before, after = before or "", after or ""
    if _HEADER.match(before) or _HEADER.match(after):
        b, a = _split_excerpts(before), _split_excerpts(after)
        if not b or not a or [h for h, _ in b] != [h for h, _ in a]:
            return None
        return [SourceHunk(hb, lb, la) for (hb, lb), (_, la) in zip(b, a)]
    if not before.strip() or not after.strip():
        return None                  # a function the change adds or removes: nothing to anchor a diff on
    bl, al = before.splitlines(), after.splitlines()
    groups = list(difflib.SequenceMatcher(None, bl, al, autojunk=False).get_grouped_opcodes(3))
    return [SourceHunk("", bl[g[0][1]:g[-1][2]], al[g[0][3]:g[-1][4]]) for g in groups] or None


def _key(line: str) -> str:
    return " ".join(line.split())


def _substantive(key: str) -> bool:
    """A line that can identify a place: two identifiers at least, or one long one."""
    words = _WORD.findall(key)
    return len(words) >= 2 or (len(words) == 1 and len(words[0]) >= 12)


def _chain(pairs: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Longest chain increasing in both coordinates (pairs sorted by the first)."""
    best: list[list[tuple[int, int]]] = []
    for p in pairs:
        cand = max((c for c in best if c[-1][1] < p[1] and c[-1][0] < p[0]), key=len, default=[])
        best.append(cand + [p])
    return max(best, key=len, default=[])


def locate(hunk: SourceHunk, target_lines: list[str]) -> tuple[int, int, int] | None:
    """(first line, last line, anchors) of the hunk's counterpart in the target
    (0-based, inclusive), or None when it cannot be anchored."""
    tpos: dict[str, list[int]] = {}
    for n, line in enumerate(target_lines):
        tpos.setdefault(_key(line), []).append(n)
    bkeys = [_key(x) for x in hunk.before]
    pairs = [(i, tpos[k][0]) for i, k in enumerate(bkeys)
             if _substantive(k) and bkeys.count(k) == 1 and len(tpos.get(k, [])) == 1]
    chain = _chain(pairs)
    if not chain:
        return None
    (i0, j0), (i1, j1) = chain[0], chain[-1]
    if j1 - j0 > (i1 - i0) + MAX_SPREAD:
        return None
    first = max(0, j0 - i0)
    last = min(len(target_lines) - 1, j1 + (len(hunk.before) - 1 - i1))
    return first, max(first, last), len(chain)


def plan_regions(before: str | None, after: str | None, target: str, *,
                 context: int = CONTEXT_LINES, max_regions: int = MAX_REGIONS,
                 max_fraction: float = MAX_FRACTION) -> RegionPlan:
    """The regions of `target` a whole-file unit can be reduced to (see module doc)."""
    hunks = source_hunks(before, after)
    if not hunks:
        return RegionPlan([], [], "the source payloads are not a change that can be split into hunks")
    lines = target.splitlines(keepends=True)
    plain = [x.rstrip("\n") for x in lines]
    if len(plain) < 2 * context + 4:
        return RegionPlan([], hunks, "the target is too short to be worth reducing")
    spans: list[tuple[int, int, list[int], int]] = []
    for n, h in enumerate(hunks):
        if not any(k for k in map(_key, h.before)) and not any(k for k in map(_key, h.after)):
            continue
        hit = locate(h, plain)
        if hit is None:
            return RegionPlan([], hunks, f"hunk {n + 1} of {len(hunks)} could not be anchored in the "
                                         "fork's file (no line of its BEFORE side occurs exactly once there)")
        first, last, anchors = hit
        spans.append((max(0, first - context), min(len(plain) - 1, last + context), [n], anchors))
    if not spans:
        return RegionPlan([], hunks, "the change has no line to anchor")
    spans.sort()
    joined: list[tuple[int, int, list[int], int]] = []
    for s in spans:
        if joined and s[0] <= joined[-1][1] + 1:
            p = joined[-1]
            joined[-1] = (p[0], max(p[1], s[1]), p[2] + s[2], p[3] + s[3])
        else:
            joined.append(s)
    offsets = [0]
    for x in lines:
        offsets.append(offsets[-1] + len(x))
    regions: list[Region] = []
    for first, last, hs, anchors in joined:
        # the exact-text splice needs the region's text to occur once: widen until it does
        while target.count("".join(lines[first:last + 1])) != 1 and (first > 0 or last < len(lines) - 1):
            first, last = max(0, first - 1), min(len(lines) - 1, last + 1)
        text = "".join(lines[first:last + 1])
        if not text.endswith("\n"):
            return RegionPlan([], hunks, "a region would end at a last line without a newline")
        regions.append(Region(offsets[first], offsets[last + 1], first + 1, last + 1, text, hs, anchors))
    for a, b in zip(regions, regions[1:]):
        if b.start < a.end:
            return RegionPlan([], hunks, "regions overlap after widening to a unique text")
    if len(regions) > max_regions:
        return RegionPlan([], hunks, f"{len(regions)} regions: more than {max_regions}")
    covered = sum(len(r.text) for r in regions)
    if covered > max_fraction * len(target):
        return RegionPlan([], hunks, f"the regions cover {covered} of {len(target)} characters: "
                                     "the whole file is asked for instead")
    return RegionPlan(regions, hunks)


def region_shape_error(region_text: str, answer: str) -> str | None:
    """Why `answer` is not the adapted region: it must still begin and end with
    the region's own first and last line (they are context, never the change),
    so a truncated answer, one statement, or the whole file is refused."""
    want = [x for x in region_text.splitlines() if x.strip()]
    got = [x for x in answer.splitlines() if x.strip()]
    if not got:
        return "the answer is empty"
    if not want:
        return None
    if _key(got[0]) != _key(want[0]):
        return f"the answer does not begin with the part's first line (`{want[0].strip()[:80]}`)"
    if _key(got[-1]) != _key(want[-1]):
        return f"the answer does not end with the part's last line (`{want[-1].strip()[:80]}`)"
    return None


def merge_regions(target: str, regions: list[Region], answers: list[str]) -> tuple[str | None, str]:
    """Splice the answers over their regions (rcap.splice). -> (text, "") or (None, why)."""
    edits = [UnitEdit(f"region-{n + 1}", r.text, a) for n, (r, a) in enumerate(zip(regions, answers))]
    mr = merge_units(target, edits)
    bad = [f"{k}: {u.status} ({u.reason})" for k, u in mr.units.items() if u.status != MERGED]
    if bad or mr.violations:
        return None, "; ".join(bad + list(mr.violations))[:300]
    return mr.text, ""
