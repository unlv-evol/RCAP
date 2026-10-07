r"""Upstream fidelity of a ported unit: is the candidate the upstream change, or something else?

A unit that passes the build and its tests is not necessarily a port. Round 2 found a
test whose candidate asserts the FORK's current behaviour (`TimeoutException`) where
upstream asserts the new one (`RejectedExecutionException`): it compiles, passes, is no
echo and no shrink. This module compares, deterministically and at token level,

* the candidate's change (fork original -> candidate) with
* the upstream change (source before -> source after) of the same function,

and returns one label with its evidence:

FAITHFUL_CODE  the upstream change is reproduced (recall >= RECALL_FAITHFUL) and the
           candidate adds little that is in neither upstream's after-side nor the
           fork original (novel share <= EXTRA_FAITHFUL; the plain extra share, not
           in upstream's after-side, is reported too), no expectation was weakened,
           AND the candidate changes code where upstream changes code. Only this
           label is a code port. (`FAITHFUL` is kept as an alias of it.)
IMPORT_ONLY  the candidate changes import statements and nothing else. Never
           reported as a code port, however complete: of 40 units the first guard
           called FAITHFUL, 16 were this or COMMENT_ONLY. `flags` says when the
           import change is incomplete or upstream also changes code here. An
           import unit whose candidate ALSO edits code is PARTIAL with the flag
           `unrequested_code_edit`, unless the upstream file shows the edit is
           upstream's (see `upstream_file`).
COMMENT_ONLY  code tokens equal the fork's; the candidate changes comments the way
           upstream does in this unit (javadoc, typo in a comment).
PARTIAL    something is missing or extra; `reasons` says what;
DUPLICATED_BLOCK  the candidate holds a block of the fork's code one more time than
           the fork and upstream account for (rcap.accept): the changed loop written
           in front of the loop it was to change. Builds, may pass, is wrong.
WEAKENED   an expectation (assertion / expected exception / literal inside an
           assertion) differs from upstream's after-side AND equals the old value
           (fork original / upstream before) or a value present in neither; or an
           assertion that upstream keeps was removed;
NO_CHANGE  the candidate is the fork's code (comments and layout ignored).
PLACEHOLDER_LEAK  the candidate carries a placeholder marker of any spelling
           (`RCAP_PH\w*`, also `/* RCAP_PHname */`) into the fork. Checked
           first: comments are otherwise ignored, so a candidate that only adds a stray
           placeholder comment used to look like NO_CHANGE.
FORK_REGRESSION  a fork-only token (in the fork's original unit, in neither side of the
           upstream change) was removed or replaced although upstream does not change
           the code it sits in: the port displaced fork code. Token overlap cannot see
           this (every token may exist upstream or in the fork), so it is positional.
           Only fork STATE counts: a name, a literal, a call on a fork object. A
           respelled static API (`TimeUnit.NANOSECONDS.convert(1, HOURS)` ->
           `HOURS.toNanos(1)`) or a dropped argument label (`f(name = v)` -> `f(v)`)
           is not fork state. Also given when the candidate writes lines that
           upstream REMOVES (the fork is moved to upstream's BEFORE text).
NOTHING_TO_PORT / NOTHING_TO_PORT_IN_ISOLATION  the upstream change of this unit is a
           pure rename of names declared outside the unit, and the candidate keeps the
           fork's names. Validated alone (or when the rename cannot be completed across
           the PR) the unchanged function is the correct answer, not an echo. Only
           given when the caller says so (`rename_context`). Also given, with
           `no_counterpart=True`, to an unchanged file whose unit is a source function
           the fork does not have. Neither a model failure nor a success.

Pure text processing: comments are ignored, string literals are single tokens,
punctuation does not count in the token shares. Mechanically applied hunks are
upstream text verbatim and are not judged here; the caller records them separately.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from rcap.accept import code_lines, describe_duplicates, duplicated_blocks, placeholder_marks

FAITHFUL_CODE = "FAITHFUL_CODE"
FAITHFUL = FAITHFUL_CODE          # the old name; the old label string was "FAITHFUL"
IMPORT_ONLY = "IMPORT_ONLY"
COMMENT_ONLY = "COMMENT_ONLY"
DUPLICATED_BLOCK = "DUPLICATED_BLOCK"
PARTIAL = "PARTIAL"
WEAKENED = "WEAKENED"
NO_CHANGE = "NO_CHANGE"
PLACEHOLDER_LEAK = "PLACEHOLDER_LEAK"
FORK_REGRESSION = "FORK_REGRESSION"
NOTHING_TO_PORT = "NOTHING_TO_PORT"
NOTHING_TO_PORT_IN_ISOLATION = "NOTHING_TO_PORT_IN_ISOLATION"

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
    out: list[Assertion] = []
    # one excerpt at a time: a call cut off by the end of a diff excerpt must not swallow
    # the next excerpt (PR-15154 ErrorHandlingIntegrationTest: a false "not ported")
    for seg in _HUNK_HEADER.split(code or ""):
        out += _segment_assertions(_segment_tokens(seg))
    return out


def _segment_assertions(ts: list[str]) -> list[Assertion]:
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
        if j >= len(ts):
            break
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
        if k >= len(ts):                       # cut off by the end of the excerpt: not judged
            i += 1
            continue
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
    kind: str | None = None                # what the candidate changes: code | import | comment | none
    upstream_kind: str | None = None       # what upstream changes in this unit: code | import | comment | none
    flags: list[str] = field(default_factory=list)
    adapted: list[str] = field(default_factory=list)    # upstream token -> fork-side spelling credited

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


_IDENT_TOKEN = re.compile(r"^[A-Za-z_$][\w$]*$")
_NEAR = 3              # code tokens on each side within which an upstream rewrite exempts a fork-only token
_MOVE_WINDOW = 12      # tokens around a rewritten block in which a fork-only token counts as kept


def pure_rename(upstream_before: str, upstream_after: str) -> list[tuple[str, str]] | None:
    """[(old, new)] when the upstream change of a unit is nothing but a consistent
    renaming of identifiers (same token sequence otherwise), else None."""
    tb, ta = tokens(upstream_before), tokens(upstream_after)
    if len(tb) != len(ta) or tb == ta:
        return None
    fwd: dict[str, str] = {}
    for b, a in zip(tb, ta):
        if b == a:
            continue
        if not (_IDENT_TOKEN.match(b) and _IDENT_TOKEN.match(a)) or fwd.setdefault(b, a) != a:
            return None
    if len(set(fwd.values())) != len(fwd):
        return None
    # consistent: an old name that is renamed is renamed everywhere
    if any(b in fwd and a == b for b, a in zip(tb, ta)):
        return None
    return sorted(fwd.items())


def declared_locally(code: str, name: str) -> bool:
    """Is `name` a local variable, parameter or lambda parameter of this unit
    (as opposed to the unit's own name or a name declared elsewhere)?"""
    ts = tokens(code)
    for i, t in enumerate(ts):
        if t != name:
            continue
        nxt = ts[i + 1] if i + 1 < len(ts) else ""
        prev = ts[i - 1] if i else ""
        if nxt in ("->", "=>"):
            return True
        if prev in ("val", "var") or (nxt == ":" and prev in ("(", ",")):
            return True                        # Scala local / parameter
        if nxt in ("=", ";", ",", ")", ":") and (
                _IDENT_TOKEN.match(prev) or prev in (">", "]")) \
                and prev not in ("return", "new", "throw", "case", "else"):
            return True                        # Java `Type name` (local or parameter)
    return False


def _is_fork_state(ts: Sequence[str], i: int) -> bool:
    """Does token i denote something of the fork's own (a name, a literal, a
    member of a fork object), as opposed to a spelling?

    Not fork state: an argument LABEL (`f(name = value)`: the value is what
    counts), and a member of a chain rooted in a type (`TimeUnit.NANOSECONDS
    .convert(..)`: a static API, which can be respelled without changing the
    value). Everything else is, including a call on an object
    (`client.setFollowRedirects(false)`) and a chain this cannot resolve."""
    t = ts[i]
    if not _IDENT_TOKEN.match(t):
        return True
    prev = ts[i - 1] if i else ""
    nxt = ts[i + 1] if i + 1 < len(ts) else ""
    if nxt == "=" and prev in ("(", ","):
        return False
    if prev == ".":
        k = i
        while k >= 2 and ts[k - 1] == "." and _IDENT_TOKEN.match(ts[k - 2]):
            k -= 2
        if k >= 1 and ts[k - 1] == ".":
            return True                        # `x().y`: receiver is not a plain name
        return not ts[k][:1].isupper()
    return True


def reverts_to_before(fork_original: str, candidate: str, upstream_before: str,
                      upstream_after: str) -> list[str]:
    """Code lines the candidate ADDS to the fork that upstream REMOVES (they are
    in upstream's before-side and not in its after-side): the candidate moves
    the fork to the text upstream moved away from (PR-16170). Two or more
    substantive lines, so a re-added brace or a one-liner does not count."""
    from rcap.accept import _substantive

    fork, after = set(code_lines(fork_original)), set(code_lines(upstream_after))
    removed = {ln for ln in code_lines(upstream_before) if ln not in after and _substantive(ln)}
    added = [ln for ln in dict.fromkeys(code_lines(candidate)) if ln not in fork and ln in removed]
    return added if len(added) >= 2 else []


_IMPORT_LINE = re.compile(r"^[ \t]*(?:import|package)\b[^\n]*$", re.MULTILINE)
_COMMENT_TEXT = re.compile(r'''"(?:\\.|[^"\\\n])*"|(//[^\n]*|/\*.*?(?:\*/|\Z))''', re.DOTALL)
_WORDS = re.compile(r"[A-Za-z0-9_@<>{}#.]+")


def without_imports(code: str) -> str:
    """`code` without its import and package statements."""
    return _IMPORT_LINE.sub("", code or "")


def comment_words(code: str) -> Counter:
    """Words of the comments of `code` (excerpt texts may start inside one)."""
    out: Counter = Counter()
    for seg in _HUNK_HEADER.split(code or ""):
        close, open_ = seg.find("*/"), seg.find("/*")
        if close >= 0 and (open_ < 0 or close < open_):
            out.update(_WORDS.findall(seg[:close]))
            seg = seg[close + 2:]
        for m in _COMMENT_TEXT.finditer(seg):
            if m.group(1):
                out.update(_WORDS.findall(m.group(1).lstrip("/*")))
    return out


def change_kind(a: str, b: str) -> str:
    """What differs between two texts: code | import | comment | none."""
    ta, tb = tokens(a), tokens(b)
    if ta == tb:
        return "comment" if comment_words(a) != comment_words(b) else "none"
    return "import" if tokens(without_imports(a)) == tokens(without_imports(b)) else "code"


def _char_edit(a: str, b: str) -> tuple[tuple[str, str, int], ...]:
    """The character edit a -> b: (deleted, inserted, characters after it). Quotes
    of a literal are not part of its value. The position from the end is part of
    the edit: `0 -> 1` in the last place is not `0 -> 1` in the middle."""
    a, b = a.strip("\"'"), b.strip("\"'")
    return tuple((a[i1:i2], b[j1:j2], len(a) - i2) for op, i1, i2, j1, j2
                 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if op != "equal")


def _replacements(a: Sequence[str], b: Sequence[str]) -> list[tuple[str, str]]:
    """(old, new) token pairs of the equal-length replace blocks of a -> b."""
    out: list[tuple[str, str]] = []
    for op, i1, i2, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op == "replace" and i2 - i1 == j2 - j1:
            out += [(x, y) for x, y in zip(a[i1:i2], b[j1:j2]) if x != y]
    return out


def adaptations(tf: Sequence[str], tc: Sequence[str], tb: Sequence[str], ta: Sequence[str],
                up_add: Sequence[str], c_add: Sequence[str]) -> dict[str, str]:
    """Upstream-added tokens the candidate wrote in the fork's spelling: {upstream
    token: candidate token}. Two rules, both about a token upstream adds that
    the candidate lacks while it adds one upstream does not have:

    * a consistent NAME substitution: the candidate's added tokens are
      upstream's with identifier u always written c (one-to-one), where the
      fork does not have u (PR-18611: the fork has no `d3`, the port says `d2`);
    * the SAME CHARACTER EDIT: upstream turns X0 into X1, the candidate turns
      the fork's Y0 into Y1, and X0->X1 and Y0->Y1 are the same edit of
      characters (PR-12250: `IBP_3_0_IV0`->`IBP_3_0_IV1` upstream, the fork's
      `"3.0-IV0"`->`"3.0-IV1"`)."""
    out: dict[str, str] = {}
    fork_vocab, after_vocab = set(tf), set(ta)
    fwd: dict[str, str] = {}
    bad: set[str] = set()
    for u, c in _replacements(list(up_add), list(c_add)):
        if not (_IDENT_TOKEN.match(u) and _IDENT_TOKEN.match(c)) or u in fork_vocab or c in after_vocab:
            continue
        if fwd.setdefault(u, c) != c:
            bad.add(u)
    for u, c in fwd.items():
        if u not in bad and list(fwd.values()).count(c) == 1:
            out[u] = c
    up_edits = {(x, y): _char_edit(x, y) for x, y in _replacements(tb, ta)}
    for (y0, y1) in _replacements(tf, tc):
        edit = _char_edit(y0, y1)
        for (x0, x1), e in up_edits.items():
            if e and e == edit and x1 not in out and y1 not in after_vocab and x1 != y1:
                out[x1] = y1
    return out


def fork_regressions(fork_original: str, candidate: str, upstream_before: str,
                     upstream_after: str) -> list[str]:
    """Evidence lines for fork-only code the candidate displaced.

    A token is fork-only when it is in the fork's original unit and on neither
    side of the upstream change. It must stay where it is (or at least inside
    the block that replaces its surroundings) unless upstream itself deletes or
    rewrites the code it sits in: the before-tokens the fork-only run replaces,
    or one of the `_NEAR` code tokens before or after it. An upstream INSERTION next to
    it is not an exemption: the port has to keep both. The exemption is
    deliberately wide (precision over recall): a displaced fork token right next
    to a token upstream rewrites is not reported."""
    tf, tc = tokens(fork_original), tokens(candidate)
    tb, ta = tokens(upstream_before), tokens(upstream_after)
    if not tb or tf == tc:
        return []
    vocab = set(tb) | set(ta)
    fork_only = {i for i, t in enumerate(tf) if t not in vocab and t not in _PUNCT
                 and (_IDENT_TOKEN.match(t) or t[:1] in "\"'" or t[:1].isdigit())
                 and _is_fork_state(tf, i)}
    if not fork_only:
        return []
    changed_b = [False] * (len(tb) + 1)
    for op, i1, i2, _, _ in SequenceMatcher(None, tb, ta, autojunk=False).get_opcodes():
        if op in ("replace", "delete"):
            for i in range(i1, i2):
                changed_b[i] = True
    exempt: set[int] = set()
    for op, i1, i2, j1, j2 in SequenceMatcher(None, tf, tb, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        inner = any(changed_b[j] for j in range(j1, j2))
        # nearest code tokens around it (punctuation aligns too freely to mean anything)
        left = [j for j in range(j1 - 1, -1, -1) if tb[j] not in _PUNCT][:_NEAR]
        right = [j for j in range(j2, len(tb)) if tb[j] not in _PUNCT][:_NEAR]
        around = any(changed_b[j] for j in left + right)
        if inner or around:
            exempt.update(range(i1, i2))
    out: list[str] = []
    for op, i1, i2, j1, j2 in SequenceMatcher(None, tf, tc, autojunk=False).get_opcodes():
        if op == "equal":
            continue
        # "still there" = anywhere in the replacing block or right around it (a moved argument)
        block = tc[max(0, j1 - _MOVE_WINDOW):j2 + _MOVE_WINDOW]
        lost = [tf[i] for i in range(i1, i2) if i in fork_only and i not in exempt and tf[i] not in block]
        if not lost:
            continue
        out.append(
            f"fork-only `{'`, `'.join(dict.fromkeys(lost))}` (on neither side of the upstream change) "
            f"removed: fork `{' '.join(tf[max(0, i1 - 4):i2 + 2])}` -> candidate "
            f"`{' '.join(tc[max(0, j1 - 4):j2 + 2])}`; upstream does not change the code around it")
    return out


def judge(fork_original: str, candidate: str, upstream_before: str, upstream_after: str,
          *, rename_context: str | None = None,
          fork_has: Callable[[str], bool] | None = None,
          language: str | None = None, no_counterpart: bool = False,
          upstream_file: tuple[str, str] | None = None) -> Fidelity:
    """Fidelity of `candidate` (a port of upstream_before -> upstream_after onto fork_original).

    `rename_context`: "isolation" (the unit was validated alone) or "pr" (the PR's
    rename could not be completed); only then may an unchanged unit whose upstream
    change is a pure rename be labelled NOTHING_TO_PORT(_IN_ISOLATION). `fork_has(name)`
    lets the caller veto that when the fork already declares the new name.
    `language` is needed for the duplicated-block check (it finds method bodies);
    `no_counterpart`: the caller established that the unit is a source function the
    fork does not have (rcap.scoring), so an unchanged fork file is NOTHING_TO_PORT.
    `upstream_file`: (before, after) of the WHOLE upstream file. SALP may slice the
    import of a change into one unit and its code into a sibling; a whole-file
    candidate of the import unit then carries the sibling's code too. With the file
    texts that code is recognised as upstream's (flag `code_of_sibling_unit`,
    FAITHFUL_CODE); without them such a unit is PARTIAL + `unrequested_code_edit`."""
    marks = placeholder_marks(candidate)
    if marks:
        lines = [ln.strip()[:200] for ln in candidate.splitlines() if "RCAP_PH" in ln]
        return Fidelity(PLACEHOLDER_LEAK, reasons=[f"{len(lines)} line(s) carry a placeholder marker into the fork"],
                        evidence=lines[:5])
    tf, tc = tokens(fork_original), tokens(candidate)
    tb, ta = tokens(upstream_before), tokens(upstream_after)
    kind, up_kind = change_kind(fork_original, candidate), change_kind(upstream_before, upstream_after)
    if tf == tc:
        if kind == "comment":
            cw = comment_words(candidate) - comment_words(fork_original)
            cw += comment_words(fork_original) - comment_words(candidate)
            uw = comment_words(upstream_after) - comment_words(upstream_before)
            uw += comment_words(upstream_before) - comment_words(upstream_after)
            got = sum((cw & uw).values())
            if uw and got / sum(uw.values()) >= 0.5:
                return Fidelity(COMMENT_ONLY, recall=round(got / sum(uw.values()), 3), kind=kind, upstream_kind=up_kind,
                                reasons=["only comments change, as upstream's comments do in this unit"]
                                + (["upstream also changes code here, which is not ported"] if tb != ta else []),
                                flags=["code_not_ported"] if tb != ta else [])
            return Fidelity(NO_CHANGE, recall=0.0 if tb != ta else None, kind=kind, upstream_kind=up_kind,
                            reasons=["the candidate edits comments only, and not the way upstream does"])
        if tb == ta:
            # upstream changes only comments / layout here, and so does (or not) the candidate
            return Fidelity(NO_CHANGE, kind=kind, upstream_kind=up_kind,
                            reasons=["comment/layout-only: upstream changes no code token in this unit either"])
        pairs = pure_rename(upstream_before, upstream_after) if rename_context else None
        if pairs and not any(declared_locally(upstream_after, new) for _, new in pairs) \
                and not (fork_has and any(fork_has(new) for _, new in pairs)):
            label = NOTHING_TO_PORT_IN_ISOLATION if rename_context == "isolation" else NOTHING_TO_PORT
            shown = ", ".join(f"{o} -> {n}" for o, n in pairs)
            return Fidelity(label, recall=0.0, kind=kind, upstream_kind=up_kind, reasons=[
                f"upstream only renames {shown} (declared outside this unit); keeping the fork's "
                "name is the correct result " + ("when the unit is validated alone" if rename_context == "isolation"
                                                 else "because the rename is not completed in the fork")],
                evidence=[f"rename {o} -> {n}" for o, n in pairs])
        if no_counterpart:
            return Fidelity(NOTHING_TO_PORT, recall=0.0, kind=kind, upstream_kind=up_kind, flags=["no_counterpart"],
                            reasons=["the unit is a source function the fork does not have; the fork's file is unchanged"])
        return Fidelity(NO_CHANGE, recall=0.0, kind=kind, upstream_kind=up_kind, reasons=["candidate is the fork's code"])
    dup = duplicated_blocks(fork_original, candidate, upstream_before, upstream_after, language=language) \
        if language is not None else []
    if dup:
        return Fidelity(DUPLICATED_BLOCK, kind=kind, upstream_kind=up_kind,
                        reasons=[f"{len(dup)} block(s) of the fork's code written once more than the fork and upstream account for"],
                        evidence=describe_duplicates(dup)[:5])
    up_add, up_rem = (_significant(x) for x in _delta(tb, ta))
    c_add, c_rem = (_significant(x) for x in _delta(tf, tc))
    # upstream tokens the candidate wrote in the fork's spelling count as written
    adapted = adaptations(tf, tc, tb, ta, up_add, c_add)
    if adapted:
        back = {c: u for u, c in adapted.items()}
        c_add = [back.get(t, t) for t in c_add]
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
    ref_before, ref_after = upstream_before, upstream_after
    if upstream_file is not None and up_kind != "code" and kind == "code":
        # The unit's own upstream text holds no code change (an import unit), but the candidate
        # edits code: that code is judged against the upstream FILE, where its change is. What
        # the file changes elsewhere and this unit does not port is not held against it.
        ref_before, ref_after = upstream_file
        weak, notes = _assertion_check(fork_original, candidate, ref_before, ref_after)
        weak = [w for w in weak if not w.startswith("kept the old expectation")]
        notes = []
    else:
        weak, notes = _assertion_check(fork_original, candidate, upstream_before, upstream_after)
    if not adapted and Counter(c_add) == Counter(up_add) and Counter(c_rem) == Counter(up_rem):
        # the candidate's token change IS upstream's: a note that an expectation "differs" can
        # only come from pairing the wrong two assertions (PR-13885: two message typos)
        notes = []
    f = Fidelity(PARTIAL, recall=recall, extra_share=extra_share, novel_share=novel_share,
                 upstream_added=len(up_add), upstream_removed=len(up_rem),
                 candidate_added=len(c_add), candidate_removed=len(c_rem),
                 extra_tokens=sorted(extra.elements())[:40],
                 missing_tokens=sorted((miss_add_net + miss_rem_net).elements())[:40],
                 evidence=weak + notes, kind=kind, upstream_kind=up_kind,
                 adapted=[f"{u} -> {c}" for u, c in sorted(adapted.items())])
    if weak:
        f.label = WEAKENED
        f.reasons.append(f"{len(weak)} weakened expectation(s)")
        return f
    regress = fork_regressions(fork_original, candidate, ref_before, ref_after)
    if regress:
        f.label = FORK_REGRESSION
        f.reasons.append(f"{len(regress)} fork-only expression(s) displaced")
        f.evidence = regress + f.evidence
        return f
    back_lines = reverts_to_before(fork_original, candidate, upstream_before, upstream_after)
    if back_lines:
        f.label = FORK_REGRESSION
        f.flags.append("reverts_to_upstream_before")
        f.reasons.append(f"REVERTS_TO_UPSTREAM_BEFORE: the candidate writes {len(back_lines)} line(s) that upstream removes")
        f.evidence = [f"added although upstream removes it: `{ln[:160]}`" for ln in back_lines[:5]] + f.evidence
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
    # ---- what kind of change is it? An import (or comment) change is never a code port.
    if kind == "import" or (up_kind in ("import", "comment", "none") and kind == "code"):
        imp_up = Counter(_significant(_delta(tokens(_imports_only(upstream_before)), tokens(_imports_only(upstream_after)))[0]))
        imp_c = Counter(_significant(_delta(tokens(_imports_only(fork_original)), tokens(_imports_only(candidate)))[0]))
        rem_up = Counter(_significant(_delta(tokens(_imports_only(upstream_before)), tokens(_imports_only(upstream_after)))[1]))
        rem_c = Counter(_significant(_delta(tokens(_imports_only(fork_original)), tokens(_imports_only(candidate)))[1]))
        shares = bool((imp_up & imp_c) - Counter({"import": 10 ** 6, "static": 10 ** 6})) or bool(
            (rem_up & rem_c) - Counter({"import": 10 ** 6, "static": 10 ** 6}))
        if kind == "code" and upstream_file is not None:
            # token bags of the whole files (imports aside): layout and line breaks do not matter
            bag = lambda text: Counter(_significant(tokens(without_imports(text))))  # noqa: E731
            fork_bag, cand_bag = bag(fork_original), bag(candidate)
            before_bag, after_bag = bag(upstream_file[0]), bag(upstream_file[1])
            mine, gone = cand_bag - fork_bag, fork_bag - cand_bag
            theirs, theirs_gone = after_bag - before_bag, before_bag - after_bag
            # a token upstream removes and the fork never had cannot be removed here
            n = sum(mine.values()) + sum(gone.values())
            hit = sum((mine & theirs).values()) + sum((gone & theirs_gone).values())
            # ... and it must ADD something upstream adds: dropping two tokens upstream also
            # drops somewhere in the file (PR-14628: an argument label) proves nothing
            if n and hit / n >= RECALL_FAITHFUL and (mine & theirs):
                f.label = FAITHFUL_CODE
                f.flags.append("code_of_sibling_unit")
                f.reasons = [f"this unit's own upstream change is {up_kind}-only; the candidate's code edit "
                             f"({hit} of {n} changed tokens) is one upstream makes elsewhere in the same file"]
                return f
        if kind == "code":
            # not "import only": the row would hide an edit nobody asked for behind a harmless name
            f.flags.append("unrequested_code_edit")
            f.reasons.append("upstream changes only imports/comments in this unit; the candidate also edits code"
                             + ("" if upstream_file is not None else " (upstream file not given: it may be a sibling unit's change)"))
            return f
        if kind == "import" or shares:
            if f.reasons:
                f.flags.append("import_change_incomplete_or_extra")
            if up_kind == "code":
                f.flags.append("code_not_ported")
                f.reasons.append("upstream also changes code in this unit; the candidate changes imports only")
            f.label = IMPORT_ONLY
            return f
    if not f.reasons:
        f.label = FAITHFUL_CODE
    return f


def _imports_only(code: str) -> str:
    return "\n".join(m.group(0) for m in _IMPORT_LINE.finditer(code or ""))
