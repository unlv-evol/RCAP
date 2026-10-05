"""Dependency-ordered application of one upstream PR's change groups.

Rounds 2-3 applied every candidate of a PR at once and then dropped failing
units until the build settled. That is order-dependent and it cascades: one
bad sibling masked every good unit, and a half-applied rename took the whole
rename out. Here the PR's change groups are ordered before anything is built:

* a GROUP is one SAP unit with a candidate, or one mechanically applicable
  upstream hunk that no unit covers (declarations and fields outside units,
  JSON message schemas, ...);
* group A DEPENDS on group B when A's added lines use a name that B introduces
  (a HARD dependency: without B, A cannot compile), or when A is the test of
  B's file (a SOFT dependency: it orders the two, and explains a failure of A
  when B is missing, but A is still tried);
* a RENAME (declaration + every use) is ONE atomic group: it lands completely
  or not at all;
* groups are applied in topological layers, providers first. When a provider
  fails or is missing, its hard dependents are BLOCKED_BY(provider) and are
  not built; independent groups go on.

Pure functions over text and names: no git, no build system. The driver
supplies the groups and runs the builds.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from rcap import cochange as cc
from rcap.fidelity import tokens

_IDENT = re.compile(r"^[A-Za-z_$][\w$]*$")
_TEST_SUFFIX = re.compile(r"(?:IntegrationTest|UnitTest|Tests?|Spec|IT)$")


def added_lines(original: str, candidate: str) -> list[str]:
    """Lines the candidate adds or rewrites relative to its original."""
    a, b = original.splitlines(), candidate.splitlines()
    out: list[str] = []
    for op, _, _, j1, j2 in SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if op in ("replace", "insert"):
            out += b[j1:j2]
    return out


def identifiers(lines: Iterable[str]) -> set[str]:
    """Identifier tokens of the given lines (comments and string contents dropped)."""
    return {t for t in tokens("\n".join(lines)) if _IDENT.match(t)}


def introduced(added: Sequence[str], removed: Sequence[str], file_text: str | None) -> set[str]:
    """Names DECLARED on added lines that are new: not declared on the removed
    lines and not a word of the fork file the change lands in."""
    new = set(cc.declared_names(added)) - set(cc.declared_names(removed))
    if file_text:
        words = identifiers(file_text.splitlines())
        new -= words
    return new


@dataclass
class Group:
    gid: str
    kind: str                       # unit | mech
    file: str                       # fork-relative target file
    provides: set[str] = field(default_factory=set)
    uses: set[str] = field(default_factory=set)
    is_test: bool = False
    members: list[str] = field(default_factory=list)     # an atomic group's member ids
    atomic: str | None = None       # why the members are one group ("rename a -> b")

    @property
    def ids(self) -> list[str]:
        return self.members or [self.gid]


def is_test_of(test_file: str, prod_file: str) -> bool:
    """`FooTest` / `FooIntegrationTest` (a test source) is a test of `Foo` (a non-test source)."""
    if "/src/test/" not in test_file or "/src/test/" in prod_file:
        return False
    t, p = test_file.rsplit("/", 1)[-1].split(".")[0], prod_file.rsplit("/", 1)[-1].split(".")[0]
    return bool(p) and _TEST_SUFFIX.sub("", t) == p and t != p


def _matches(used: set[str], provider: Group) -> set[str]:
    hit = used & provider.provides
    if provider.file.endswith(".json"):
        # a schema field `ErrorCode` provides the generated accessors `errorCode` / `setErrorCode`
        hit |= {u for u in used if any(al in provider.provides for al in cc._aliases(u))}
    return hit


@dataclass
class Graph:
    hard: dict[str, dict[str, list[str]]]       # group -> provider -> names used
    soft: dict[str, list[str]]                  # group -> providers it is a test of

    def deps(self, gid: str) -> set[str]:
        return set(self.hard.get(gid, {})) | set(self.soft.get(gid, []))

    def hard_dependents(self, gid: str) -> set[str]:
        """Everything that transitively needs `gid` to compile."""
        out: set[str] = set()
        todo = [gid]
        while todo:
            cur = todo.pop()
            for g, provs in self.hard.items():
                if cur in provs and g not in out and g != gid:
                    out.add(g)
                    todo.append(g)
        return out

    def hard_providers(self, gid: str) -> set[str]:
        out: set[str] = set()
        todo = [gid]
        while todo:
            for p in self.hard.get(todo.pop(), {}):
                if p not in out and p != gid:
                    out.add(p)
                    todo.append(p)
        return out


def build_graph(groups: Sequence[Group], *,
                in_fork: Callable[[str], bool] | None = None) -> Graph:
    """`in_fork(name)`: the name already exists somewhere in the fork; using it
    is then not evidence of a dependency on the group that (also) declares it."""
    hard: dict[str, dict[str, list[str]]] = {}
    soft: dict[str, list[str]] = {}
    known: dict[str, bool] = {}

    def exists(name: str) -> bool:
        if in_fork is None:
            return False
        if name not in known:
            known[name] = in_fork(name)
        return known[name]

    for a in groups:
        for b in groups:
            if a is b:
                continue
            names = sorted(n for n in _matches(a.uses, b)
                           if n not in a.provides and not exists(n))
            if names:
                hard.setdefault(a.gid, {})[b.gid] = names
            elif a.is_test and not b.is_test and is_test_of(a.file, b.file):
                soft.setdefault(a.gid, []).append(b.gid)
    return Graph(hard, soft)


def layers(groups: Sequence[Group], graph: Graph) -> list[list[str]]:
    """Topological layers, providers first. Groups in a dependency cycle land
    in the same layer (they can only be judged together); the order inside a
    layer is the given order, so the result is deterministic."""
    order = [g.gid for g in groups]
    index = {g: n for n, g in enumerate(order)}
    deps = {g: {d for d in graph.deps(g) if d in index} for g in order}
    # strongly connected components by mutual reachability (a PR has few groups)
    reach: dict[str, set[str]] = {}
    for g in order:
        seen: set[str] = set()
        todo = list(deps[g])
        while todo:
            d = todo.pop()
            if d not in seen:
                seen.add(d)
                todo.extend(deps[d])
        reach[g] = seen
    comp: dict[str, str] = {}
    for g in order:
        comp[g] = next(h for h in order if h == g or (h in reach[g] and g in reach[h]))
    depth: dict[str, int] = {}

    def comp_depth(c: str) -> int:
        if c not in depth:
            below = {comp[d] for g in order if comp[g] == c for d in deps[g]} - {c}
            depth[c] = 1 + max((comp_depth(b) for b in below), default=-1)
        return depth[c]

    out: dict[int, list[str]] = {}
    for g in order:
        out.setdefault(comp_depth(comp[g]), []).append(g)
    return [out[k] for k in sorted(out)]


# ---------------------------------------------------------------------- renames
@dataclass
class Rename:
    old: str
    new: str
    decl_hunk: str                  # hunk id of the declaration
    use_hunks: list[str] = field(default_factory=list)   # other hunks that swap old -> new

    @property
    def label(self) -> str:
        return f"rename {self.old} -> {self.new}"

    @property
    def hunks(self) -> list[str]:
        return [self.decl_hunk, *self.use_hunks]


def _word(name: str) -> re.Pattern[str]:
    return re.compile(rf"(?<![\w$]){re.escape(name)}(?![\w$])")


def swap_word(text: str, old: str, new: str) -> str:
    return _word(old).sub(new, text)


def _swapped_pairs(h: cc.DiffHunk, old: str, new: str) -> bool:
    """Does the hunk hold a removed line that becomes an added line by old -> new?"""
    added = {" ".join(t.split()) for _, t in h.added if _word(new).search(t)}
    return any(_word(old).search(t) and " ".join(swap_word(t, old, new).split()) in added
               for _, t in h.removed)


def brace_depth(text: str, line_no: int) -> int:
    """Nesting depth at the start of 1-based line `line_no` (comments and string
    contents ignored): 0 at top level, 1 for a member of a top-level type."""
    head = "\n".join(text.splitlines()[:max(0, line_no - 1)])
    ts = tokens(head)
    return ts.count("{") - ts.count("}")


def find_renames(files: Sequence[cc.FileDiff],
                 old_text: Callable[[str], str | None] | None = None,
                 max_depth: int = 1) -> list[Rename]:
    """Renames in the upstream change: a hunk in which a DECLARATION of `old`
    turns into the same declaration of `new` (the line is otherwise equal),
    plus every other hunk in which a line turns into itself with old -> new.

    Only MEMBER renames matter across groups. With `old_text(path)` (the
    upstream file before the change) a declaration nested deeper than
    `max_depth` braces is a local variable and is ignored: a local rename is
    an ordinary edit of its one unit."""
    out: list[Rename] = []
    seen: set[tuple[str, str]] = set()
    hunks = [h for f in files if not f.binary for h in f.hunks]
    for h in hunks:
        removed = [t for _, t in h.removed]
        added = [t for _, t in h.added]
        gone = [n for n in cc.declared_names(removed) if n not in cc.declared_names(added)]
        came = [n for n in cc.declared_names(added) if n not in cc.declared_names(removed)]
        for old in gone:
            for new in came:
                if (old, new) in seen or not _swapped_pairs(h, old, new):
                    continue
                decl = any(cc.is_declaration(t, old) and " ".join(swap_word(t, old, new).split())
                           in {" ".join(x.split()) for x in added} for t in removed)
                if not decl:
                    continue
                if old_text is not None:
                    before = old_text(h.old_path or h.new_path or "")
                    line = next((n for n, t in h.removed if cc.is_declaration(t, old)), None)
                    if before and line and brace_depth(before, line) > max_depth:
                        continue
                seen.add((old, new))
                uses = [x.hunk_id for x in hunks if x is not h and _swapped_pairs(x, old, new)]
                out.append(Rename(old, new, h.hunk_id, uses))
    return out


def revert_rename(text: str, original: str, renames: Sequence[Rename]) -> tuple[str, list[str]]:
    """Keep the fork's names: put `old` back wherever the candidate wrote `new`,
    for names the original text does not already use. -> (text, labels applied)."""
    done: list[str] = []
    for r in renames:
        if _word(r.new).search(original) or not _word(r.new).search(text):
            continue
        text = swap_word(text, r.new, r.old)
        done.append(r.label)
    return text, done
