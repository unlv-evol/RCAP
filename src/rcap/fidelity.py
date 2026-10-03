"""Upstream fidelity of a ported unit: is the candidate the upstream change, or something else?

A unit that passes the build and its tests is not necessarily a port. Round 2 found a
test whose candidate asserts the FORK's current behaviour (`TimeoutException`) where
upstream asserts the new one (`RejectedExecutionException`): it compiles, passes, is no
echo and no shrink. This module compares, deterministically and at token level,

* the candidate's change (fork original -> candidate) with
* the upstream change (source before -> source after) of the same function,

and returns one label with its evidence:

FAITHFUL   the upstream change is reproduced (recall >= RECALL_FAITHFUL) and the
           candidate adds little that is in neither upstream's after-side nor the
           fork original (novel share <= EXTRA_FAITHFUL; the plain extra share, not
           in upstream's after-side, is reported too), and no expectation was weakened;
PARTIAL    something is missing or extra; `reasons` says what;
WEAKENED   an expectation (assertion / expected exception / literal inside an
           assertion) differs from upstream's after-side AND equals the old value
           (fork original / upstream before) or a value present in neither; or an
           assertion that upstream keeps was removed;
NO_CHANGE  the candidate is the fork's code (comments and layout ignored).

Pure text processing: comments are ignored, string literals are single tokens,
punctuation does not count in the token shares. Mechanically applied hunks are
upstream text verbatim and are not judged here; the caller records them separately.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher

FAITHFUL = "FAITHFUL"
PARTIAL = "PARTIAL"
WEAKENED = "WEAKENED"
NO_CHANGE = "NO_CHANGE"

RECALL_FAITHFUL = 0.9
EXTRA_FAITHFUL = 0.2

_TOKEN = re.compile(r'''
    (?P<comment>//[^\n]*|/\*.*?(?:\*/|\Z))
  | (?P<string>"""(?:.|\n)*?"""|"(?:\\.|[^"\\\n])*"|'(?:\\.|[^'\\\n])')
  | (?P<number>\b\d[\w.]*(?:[eE][+-]?\d+)?[fFdDlL]?)
  | (?P<ident>[A-Za-z_$][\w$]*)
  | (?P<op>->|=>|::|<<=|>>>=|>>=|<=|>=|==|!=|&&|\|\||\+\+|--|[-+*/%&|^]=|<-|.)
''', re.DOTALL | re.VERBOSE)
_PUNCT = frozenset("(){}[];,.@")
# SAP unit texts made of diff excerpts separate them with hunk headers
_HUNK_HEADER = re.compile(r"^@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@[^\n]*$", re.MULTILINE)
_ASSERT_NAME = re.compile(r"^(?:assert\w*|expect\w*|fail|intercept|verify)$")
_CONST = re.compile(r"^[A-Z][A-Z0-9_]*[A-Z0-9]$")
_LITERAL_WORDS = frozenset({"true", "false", "null"})


def tokens(code: str) -> list[str]:
    """Code tokens, comments dropped, `X . class` folded into `X.class`."""
    out: list[str] = []
    for seg in _HUNK_HEADER.split(code or ""):
        out += _segment_tokens(seg)
    return out


def _segment_tokens(code: str) -> list[str]:
    out: list[str] = []
    # an excerpt may start inside a doc comment (its region begins below the `/**`) and end
    # inside one (an unterminated `/*` runs to the end: see _TOKEN)
    close, open_ = code.find("*/"), code.find("/*")
    if close >= 0 and (open_ < 0 or close < open_):
        code = code[close + 2:]
    for m in _TOKEN.finditer(code):
        kind = m.lastgroup
        if kind == "comment":
            continue
        t = m.group(0)
        if t.isspace():
            continue
        if t == "class" and len(out) >= 2 and out[-1] == "." and re.match(r"[A-Za-z_$]", out[-2]):
            out[-2:] = [out[-2] + ".class"]
            continue
        out.append(t)
    return out


def _significant(ts: Sequence[str]) -> list[str]:
    return [t for t in ts if t not in _PUNCT]


def _delta(a: Sequence[str], b: Sequence[str]) -> tuple[list[str], list[str]]:
    """(added, removed) tokens of a -> b."""
    added: list[str] = []
    removed: list[str] = []
    for op, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op in ("replace", "delete"):
            removed += a[i1:i2]
        if op in ("replace", "insert"):
            added += b[j1:j2]
    return added, removed


def _is_value(t: str) -> bool:
    return (t[:1] in "\"'" or t[:1].isdigit() or t in _LITERAL_WORDS
            or t.endswith(".class") or bool(_CONST.match(t)))


@dataclass(frozen=True)
class Assertion:
    """One outermost assertion-like call: its shape and the values it pins."""

    text: str                      # tokens joined, for evidence
    skeleton: tuple[str, ...]      # tokens with the method name and values abstracted
    values: tuple[str, ...]        # method name first, then literals / X.class / CONSTANTS

    @property
    def key(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        return self.skeleton, self.values


def assertions(code: str) -> list[Assertion]:
    """Outermost calls assert*/expect*/fail/intercept/verify(...) and @Test(expected = ...)."""
    ts = tokens(code)
    out: list[Assertion] = []
    i = 0
    while i < len(ts):
        t = ts[i]
        name = None
        if _ASSERT_NAME.match(t) and i + 1 < len(ts) and ts[i + 1] in ("(", "["):
            name = t
        elif t == "@" and ts[i + 1:i + 3] == ["Test", "("] and "expected" in ts[i + 3:i + 12]:
            name, i = "@Test", i + 1
        if name is None:
            i += 1
            continue
        j = i + 1
        if ts[j] == "[":                       # Scala: intercept[X](...) / assertThrows[X](...)
            while j < len(ts) and ts[j] != "]":
                j += 1
            j += 1
        depth, k = 0, j
        while k < len(ts):
            if ts[k] == "(":
                depth += 1
            elif ts[k] == ")":
                depth -= 1
                if depth == 0:
                    break
            k += 1
        body = ts[i:k + 1]
        vals = [name] + [x for x in body[1:] if _is_value(x)]
        skel = ["<M>"] + ["<V>" if _is_value(x) else x for x in body[1:]]
        out.append(Assertion(" ".join(body), tuple(skel), tuple(vals)))
        i = k + 1
    return out


def _sim(a: Assertion, b: Assertion) -> float:
    return SequenceMatcher(None, a.skeleton, b.skeleton, autojunk=False).ratio()


def _best(target: Assertion, pool: Sequence[Assertion], used: set[int], floor: float = 0.6) -> int | None:
    best, score = None, floor
    for n, c in enumerate(pool):
        if n in used:
            continue
        s = 1.0 + (c.values[0] == target.values[0]) if c.skeleton == target.skeleton else _sim(c, target)
        if s >= score:
            best, score = n, s
    return best


@dataclass
class Fidelity:
    label: str
    recall: float | None = None            # share of the upstream change reproduced
    extra_share: float | None = None       # share of candidate-added tokens not on upstream's after-side
    novel_share: float | None = None       # ... and not in the fork original either (the label uses this)
    upstream_added: int = 0
    upstream_removed: int = 0
    candidate_added: int = 0
    candidate_removed: int = 0
    extra_tokens: list[str] = field(default_factory=list)
    missing_tokens: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)   # assertion evidence lines

    def as_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


def _assertion_check(fork: str, cand: str, before: str, after: str) -> tuple[list[str], list[str]]:
    """-> (weakened evidence, partial notes) for the expectations of one unit."""
    F, C, B, A = assertions(fork), assertions(cand), assertions(before), assertions(after)
    weak: list[str] = []
    notes: list[str] = []
    if not (F or C or B or A):
        return weak, notes
    # what upstream changed: replaced/deleted before-items, replaced/inserted after-items, paired
    sm = SequenceMatcher(None, [x.key for x in B], [x.key for x in A], autojunk=False)
    changed: list[tuple[Assertion | None, Assertion]] = []
    b_changed: list[Assertion] = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            continue
        bs, as_ = B[i1:i2], A[j1:j2]
        b_changed += bs
        used_b: set[int] = set()
        for a in as_:
            n = _best(a, bs, used_b, floor=0.3)
            if n is not None:
                used_b.add(n)
            changed.append((bs[n] if n is not None else None, a))
    a_vals = {v for x in A for v in x.values}
    b_vals = {v for x in B for v in x.values}
    f_vals = {v for x in F for v in x.values}
    c_keys = Counter(x.key for x in C)
    f_keys = Counter(x.key for x in F)
    a_keys = Counter(x.key for x in A)
    used_c: set[int] = set()
    # exact reproductions first, so they are not re-used as a weakened partner
    pending: list[tuple[Assertion | None, Assertion]] = []
    for b, a in changed:
        hit = next((n for n, c in enumerate(C) if n not in used_c and c.key == a.key), None)
        if hit is not None:
            used_c.add(hit)
        else:
            pending.append((b, a))
    for b, a in pending:
        if a.key in f_keys:                    # the fork already has upstream's expectation
            continue
        n = _best(a, C, used_c)
        if n is None and b is not None:
            n = _best(b, C, used_c)
        if n is None:
            notes.append(f"upstream expectation not ported: {a.text}")
            continue
        used_c.add(n)
        c = C[n]
        if b is not None and c.key == b.key:
            weak.append(f"kept the old expectation `{c.text}` where upstream changed it to `{a.text}`")
            continue
        for v in c.values:
            if v in a.values:
                continue
            if b is not None and v in b.values and v not in a_vals:
                weak.append(f"`{v}` in `{c.text}` is upstream's BEFORE value; upstream after: `{a.text}`")
            elif v not in a_vals and v not in b_vals and v not in f_vals:
                weak.append(f"`{v}` in `{c.text}` is in neither upstream (after: `{a.text}`) nor the fork")
            else:
                notes.append(f"expectation value `{v}` differs from upstream after `{a.text}`")
    # candidate-introduced expectations that match nothing upstream
    for n, c in enumerate(C):
        if n in used_c or c.key in f_keys or c.key in a_keys:
            continue
        for v in c.values:
            if v not in a_vals and v not in f_vals:
                where = "upstream's BEFORE value" if v in b_vals else "in neither upstream nor the fork"
                weak.append(f"new expectation `{c.text}`: `{v}` is {where}")
    # expectations the fork has, upstream keeps, and the candidate dropped
    b_changed_keys = {x.key for x in b_changed}
    for key, cnt in f_keys.items():
        if key in b_changed_keys or key not in a_keys or c_keys.get(key, 0) >= min(cnt, a_keys[key]):
            continue
        x = next(y for y in F if y.key == key)
        # an assertion whose call around the values was adapted still pins the same values
        if any(c.values == x.values and _sim(c, x) >= 0.6 for c in C):
            continue
        weak.append(f"removed or altered the expectation `{x.text}`, which upstream keeps")
    return list(dict.fromkeys(weak)), list(dict.fromkeys(notes))


def judge(fork_original: str, candidate: str, upstream_before: str, upstream_after: str) -> Fidelity:
    """Fidelity of `candidate` (a port of upstream_before -> upstream_after onto fork_original)."""
    tf, tc = tokens(fork_original), tokens(candidate)
    tb, ta = tokens(upstream_before), tokens(upstream_after)
    if tf == tc:
        if tb == ta:
            # upstream changes only comments / layout here, and so does (or not) the candidate
            return Fidelity(NO_CHANGE, reasons=["comment/layout-only: upstream changes no code token in this unit either"])
        return Fidelity(NO_CHANGE, recall=0.0, reasons=["candidate is the fork's code"])
    up_add, up_rem = (_significant(x) for x in _delta(tb, ta))
    c_add, c_rem = (_significant(x) for x in _delta(tf, tc))
    # upstream-added tokens the fork already has (beyond upstream's before) need not be added;
    # upstream-removed tokens the fork never had cannot be removed
    pre = Counter(_significant(tf)) - Counter(_significant(tb))
    absent = Counter(_significant(tb)) - Counter(_significant(tf))
    miss_add = Counter(up_add) - Counter(c_add)
    miss_add_net = miss_add - pre
    miss_rem = Counter(up_rem) - Counter(c_rem)
    miss_rem_net = miss_rem - absent
    den = (len(up_add) - sum((miss_add - miss_add_net).values())
           + len(up_rem) - sum((miss_rem - miss_rem_net).values()))
    got = den - sum(miss_add_net.values()) - sum(miss_rem_net.values())
    recall = round(got / den, 3) if den > 0 else None
    extra = Counter(c_add) - Counter(_significant(ta))
    extra_share = round(sum(extra.values()) / len(c_add), 3) if c_add else 0.0
    # tokens the fork already uses (its drifted API: a different type, an extra argument) are an
    # adaptation, not an invention; only tokens in neither text count against FAITHFUL
    novel = extra - Counter(_significant(tf))
    novel_share = round(sum(novel.values()) / len(c_add), 3) if c_add else 0.0
    weak, notes = _assertion_check(fork_original, candidate, upstream_before, upstream_after)
    f = Fidelity(PARTIAL, recall=recall, extra_share=extra_share, novel_share=novel_share,
                 upstream_added=len(up_add), upstream_removed=len(up_rem),
                 candidate_added=len(c_add), candidate_removed=len(c_rem),
                 extra_tokens=sorted(extra.elements())[:40],
                 missing_tokens=sorted((miss_add_net + miss_rem_net).elements())[:40],
                 evidence=weak + notes)
    if weak:
        f.label = WEAKENED
        f.reasons.append(f"{len(weak)} weakened expectation(s)")
        return f
    if recall is None:
        f.reasons.append("upstream changes no code token in this unit (comment/layout only, or already in the fork)")
    elif recall == 0 and c_add and extra_share == 1.0:
        f.reasons.append("UNRELATED: the candidate's change shares no token with the upstream change "
                         f"(adds {' '.join(f.extra_tokens[:8])}; upstream needs {' '.join(f.missing_tokens[:8])})")
    elif recall < RECALL_FAITHFUL:
        f.reasons.append(f"recall {recall} < {RECALL_FAITHFUL}: missing {' '.join(f.missing_tokens[:12])}")
    if novel_share > EXTRA_FAITHFUL:
        f.reasons.append(f"novel share {novel_share} > {EXTRA_FAITHFUL} (added tokens in neither upstream-after "
                         f"nor the fork): {' '.join(sorted(novel.elements())[:12])}")
    if notes:
        f.reasons += notes
    if not f.reasons:
        f.label = FAITHFUL
    return f
