"""Charging the compile errors of one dependency layer to change groups.

The PR-level driver applies a layer of change groups on top of the groups it
has accepted and builds. When the build fails it has to decide which groups
of the layer to take out. It used to charge the group whose SPAN holds the
error line, and a whole-file unit's span is its whole file: any error in
`RestServer.java` was charged to a unit that only added an import there,
although the error was a call of a constructor that ANOTHER group of the
layer had changed (PR-15149), and a test that passes a (correct) annotation
argument was charged for an argument the neighbouring unit had removed from
the annotation's declaration (PR-12250). The correct ports were dropped, the
group that broke the build stayed in until a later round, and everything
depending on the victims was blocked.

Here an error is charged by what the groups actually CHANGED:

* `changed-line`: the error line is a line some group of the layer wrote, and
  no other group of the layer (un)declares a name the error is about: that
  group is charged;
* `declaration`: the error line is fork code nobody touched, and exactly one
  group of the layer changes the declaration of the name the compiler could
  not resolve or apply (constructor / method `cannot be applied`, `cannot find
  symbol`, unknown annotation argument, missing override, ...): the group
  that changed the declaration is charged, not the file showing the error;
* anything else is AMBIGUOUS (the line is one group's and its symbol's
  declaration is another's; several groups change the declaration; only the
  source line names something a group declares; nothing matches at all) and
  is never guessed: it is isolated by building subsets of the layer.

Certain charges are applied first and the layer is re-built, so an ambiguous
error that was a consequence of a certain one simply disappears, and a group
that is both a victim and a culprit shows its own error in the next round.
When a round has no certain charge the suspects are tested by subset builds
(`isolate`): each half of the suspect set on top of the rest, down to single
groups; a set whose members build one by one but not together is reduced to a
minimal failing set. Finally every group that was taken out is tried once
more on top of what was kept (`reinstated`): a group is only lost when the
kept state really does not build with it.

Every decision iterates groups in sorted order and every build is asked for a
sorted state, so the outcome does not depend on the order of the groups
inside the layer. Pure: the caller supplies the build function.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from rcap.feedback import CompileError, _dedup, missing_symbols

_IDENT = re.compile(r"[A-Za-z_$][\w$]*")
_DETAIL = re.compile(r"^\s*(symbol|location|required|found|reason)\s*:")
# names the compiler could not resolve / apply that `missing_symbols` does not cover
_EXTRA_STRONG: tuple[re.Pattern[str], ...] = (
    re.compile(r"unknown annotation argument name: (?P<name>[\w$]+)"),
    re.compile(r"^(?P<name>[\w$]+)\(.*\) (?:has|is not) (?:private|protected|public)"),
    re.compile(r"^(?P<name>[\w$]+) has (?:private|protected) access in"),
    re.compile(r"^(?:variable|method|constructor|class) (?P<name>[\w$]+) is already defined"),
    re.compile(r"^(?:missing|unspecified value) parameters? .*?(?:method|constructor) (?P<name>[\w$]+)"),
    re.compile(r"^missing argument(?:s)? (?:list )?for (?:method|constructor) (?P<name>[\w$]+)"),
)
_STOPWORDS = frozenset("""a an and annotation applied are argument arguments be cannot class
constructor does error expected for found from given has in incompatible interface is it length
method name no not of or package required symbol the to type types unknown value variable with
abstract final import new null private protected public return static this void int long boolean
String Object true false""".split())


@dataclass
class Applied:
    """What a built state looks like to the charging rule (line numbers are
    those of the written files, 1-based inclusive)."""

    # file -> (first, last, owners) of every changed hunk; owners are group ids
    changed: dict[str, list[tuple[int, int, tuple[str, ...]]]] = field(default_factory=dict)
    # file -> (first, last, group id): the extent of each applied unit
    spans: dict[str, list[tuple[int, int, str]]] = field(default_factory=dict)
    # group id -> names its changed lines declare or un-declare
    declares: dict[str, frozenset[str]] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)

    def groups_of_file(self, rel: str) -> set[str]:
        return {g for _, _, os in self.changed.get(rel, []) for g in os} \
            | {g for _, _, g in self.spans.get(rel, [])}


@dataclass
class BuildResult:
    ok: bool | None                 # True: built; False: failed; None: the state could not be written
    errors: list[CompileError] = field(default_factory=list)   # `rel` set
    applied: Applied = field(default_factory=Applied)
    unwritable: dict[str, str] = field(default_factory=dict)    # group id -> why (ok is None)
    note: str = ""                  # "build timeout", "no file:line compiler error: ..."
    tag: str = ""


@dataclass
class Charge:
    """One compile error and who may have caused it."""

    error: str                      # file:line: message
    suspects: list[str]
    how: str                        # changed-line | declaration | changed-line+declaration |
    #                                 declarations | declaration-weak | unit-span | file | none
    certain: bool
    names: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, object]:
        return {"error": self.error, "suspects": self.suspects, "how": self.how,
                "certain": self.certain, "names": self.names}


def error_symbols(e: CompileError) -> tuple[set[str], set[str]]:
    """(strong, weak) names of a compile error.

    Strong: the name the compiler says it could not resolve or apply. Weak:
    any other identifier of the message, of its detail lines and of the
    offending source line (an `incompatible types` at a call names the changed
    method only there)."""
    strong = {m.name for m in missing_symbols([e])}
    msg = e.message.strip()
    for pat in _EXTRA_STRONG:
        m = pat.search(msg)
        if m:
            strong.add(m.group("name"))
    text = [msg] + [b for b in e.body if _DETAIL.match(b)] + e.body[:1]
    weak = {t for line in text for t in _IDENT.findall(line)} - strong - _STOPWORDS
    return strong, weak


def classify(result: BuildResult, live: Iterable[str]) -> list[Charge]:
    """One Charge per distinct compile error of a failed build (see module doc)."""
    live = set(live)
    ap = result.applied
    for e in result.errors:
        if not e.body and e.rel in ap.sources:
            lines = ap.sources[e.rel].splitlines()
            if 0 < e.line <= len(lines):
                e.body = [lines[e.line - 1]]
    out: list[Charge] = []
    for e in _dedup(result.errors):
        where = f"{e.rel or e.path}:{e.line}: {e.message}"
        owners = sorted({g for a, b, os in ap.changed.get(e.rel or "", [])
                         if a <= e.line <= b for g in os} & live)
        strong, weak = error_symbols(e)

        def changers(names: set[str]) -> list[str]:
            return sorted(g for g in live
                          if g not in owners and ap.declares.get(g, frozenset()) & names)

        sc = changers(strong)
        wc = [g for g in changers(weak) if g not in sc]
        hit = sorted((strong | weak) & set().union(*[ap.declares.get(g, frozenset())
                                                      for g in sc + wc])) if sc or wc else []
        if owners and not sc and not wc:
            out.append(Charge(where, owners, "changed-line", True))
        elif owners:
            out.append(Charge(where, sorted(set(owners) | set(sc) | set(wc)),
                              "changed-line+declaration", False, hit))
        elif len(sc) == 1:
            out.append(Charge(where, sc, "declaration", True, hit))
        elif sc:
            out.append(Charge(where, sc, "declarations", False, hit))
        elif wc:
            out.append(Charge(where, wc, "declaration-weak", False, hit))
        else:
            inside = sorted({g for a, b, g in ap.spans.get(e.rel or "", [])
                             if a <= e.line <= b} & live)
            in_file = sorted(ap.groups_of_file(e.rel or "") & live)
            if inside:
                out.append(Charge(where, inside, "unit-span", False))
            elif in_file:
                out.append(Charge(where, in_file, "file", False))
            else:
                out.append(Charge(where, [], "none", False))
    return out


@dataclass
class LayerOutcome:
    kept: list[str]                                   # sorted
    failed: dict[str, dict[str, object]]              # group id -> {status, error, how, ...}
    evidence: list[dict[str, object]]                 # every round and subset build, in order
    builds: int = 0


def settle_layer(
    accepted: Sequence[str], layer: Iterable[str],
    build: Callable[[str, list[str]], BuildResult], *,
    hard_dependents: Callable[[str], Iterable[str]] | None = None,
    max_isolation_builds: int = 40, max_reinstate_builds: int = 30,
    tag: str = "layer",
) -> LayerOutcome:
    """Find which groups of `layer` build on top of `accepted`.

    `build(tag, groups)` builds exactly that state. `hard_dependents(g)`: the
    groups that cannot compile without `g`; those of the same layer are taken
    out with it (`BLOCKED_BY`). Returns the kept groups, the failed ones with
    the evidence for each, and the log of builds."""
    base = sorted(accepted)
    live = sorted(set(layer))
    failed: dict[str, dict[str, object]] = {}
    evidence: list[dict[str, object]] = []
    count = {"all": 0, "isolate": 0, "reinstate": 0}

    def run(name: str, groups: Iterable[str], kind: str = "all") -> BuildResult:
        count["all"] += 1
        if kind != "all":
            count[kind] += 1
        res = build(f"{tag}.{name}", base + sorted(groups))
        res.tag = f"{tag}.{name}"
        return res

    def drop(g: str, status: str, **detail: object) -> None:
        if g not in live:
            return
        live.remove(g)
        failed[g] = {"status": status, **detail}
        for d in sorted(hard_dependents(g)) if hard_dependents else []:
            if d in live:
                live.remove(d)
                failed[d] = {"status": "BLOCKED_BY", "blocked_by": g, "same_layer": True}

    def first_error(res: BuildResult, charges: list[Charge], g: str | None = None) -> str:
        for c in charges:
            if g is None or g in c.suspects:
                return c.error[:300]
        return (res.note or "build failed")[:300]

    rnd = 0
    while live:
        rnd += 1
        res = run(f"{rnd:02d}", live)
        if res.ok:
            evidence.append({"build": res.tag, "groups": len(live), "ok": True})
            break
        if res.ok is None:
            mine = sorted(g for g in res.unwritable if g in live)
            evidence.append({"build": res.tag, "ok": None, "unwritable": dict(res.unwritable)})
            if mine:
                for g in mine:
                    drop(g, "INTEGRATION_FAIL", error=res.unwritable[g][:300], how="merge")
            else:
                why = "; ".join(list(res.unwritable.values())[:2])[:300]
                for g in list(live):
                    drop(g, "INTEGRATION_FAIL", how="merge",
                         error="conflicts with an accepted group: " + why)
            continue
        if res.note.startswith("build timeout"):      # nothing to read, and subsets would time out too
            evidence.append({"build": res.tag, "groups": len(live), "ok": False, "note": res.note})
            for g in list(live):
                drop(g, "BUILD_FAIL", error="unattributed in its layer: build timeout",
                     how="unattributed in its layer", round=res.tag)
            break
        charges = classify(res, live)
        certain = sorted({g for c in charges if c.certain for g in c.suspects})
        evidence.append({"build": res.tag, "groups": len(live), "ok": False, "note": res.note,
                         "charges": [c.as_dict() for c in charges][:40], "certain": certain})
        if certain:
            for g in certain:
                c = next(c for c in charges if c.certain and g in c.suspects)
                drop(g, "BUILD_FAIL", error=f"{c.error} ({c.how})"[:300], how=c.how,
                     names=c.names, round=res.tag)
            continue
        # no certain charge: isolate by subset builds, never by guessing
        suspects = sorted({g for c in charges for g in c.suspects})
        if not suspects or any(not c.suspects for c in charges) or not charges:
            suspects = list(live)
        culprits, how = _isolate(live, suspects, run, evidence, count, max_isolation_builds)
        if not culprits:                         # cannot be: the full state fails
            culprits, how = list(live), "unattributed in its layer"
        for g in culprits:
            drop(g, "BUILD_FAIL", how=how, round=res.tag, isolated=how != "unattributed in its layer",
                 error=(("" if how.startswith("isolated") else how + ": ")
                        + first_error(res, charges, g if any(g in c.suspects for c in charges) else None))[:300])

    # every group taken out for a compile error is tried once more on what was kept
    for g in sorted(k for k, v in failed.items()
                    if v["status"] == "BUILD_FAIL" and "build timeout" not in str(v.get("error"))):
        if count["reinstate"] >= max_reinstate_builds:
            failed[g]["reinstatement"] = "not tried: build budget of the layer used up"
            continue
        mine = [d for d, v in failed.items() if v.get("blocked_by") == g and v.get("same_layer")]
        res = run(f"reinstate-{count['reinstate'] + 1:02d}", live + [g] + mine, "reinstate")
        evidence.append({"build": res.tag, "reinstate": g, "with": mine, "ok": bool(res.ok)})
        if res.ok:
            for x in [g, *mine]:
                failed.pop(x, None)
                live.append(x)
            live.sort()
        else:
            failed[g]["reinstatement"] = "tried on the kept state: does not build"
    return LayerOutcome(kept=sorted(live), failed=failed, evidence=evidence, builds=count["all"])


def _isolate(live: list[str], suspects: list[str], run, evidence: list, count: dict,
             budget: int) -> tuple[list[str], str]:
    """Which of `suspects` make the state fail, by subset builds on top of the
    other live groups. -> (culprits, how)."""
    rest = [g for g in live if g not in suspects]

    def test(name: str, sub: list[str]) -> bool | None:
        """Does rest + sub build? None when the budget is used up."""
        if count["isolate"] >= budget:
            return None
        res = run(name, rest + sub, "isolate")
        evidence.append({"build": res.tag, "subset": sub, "with_rest": len(rest), "ok": bool(res.ok)})
        return bool(res.ok)

    if rest:
        ok = test("isolate-rest", [])
        if ok is None:
            return suspects, "unattributed in its layer"
        if not ok:                               # the cause is not confined to the suspects
            suspects, rest = list(live), []
    if len(suspects) == 1:
        return suspects, "isolated: the rest of the layer builds without it"

    n = [0]
    out_of_budget = [False]

    def find(sub: list[str], known_failing: bool) -> list[str]:
        if not known_failing:
            n[0] += 1
            ok = test(f"isolate-{n[0]:02d}", sub)
            if ok is None:
                out_of_budget[0] = True
                return sub
            if ok:
                return []
        if len(sub) == 1:
            return sub
        mid = len(sub) // 2
        left, right = find(sub[:mid], False), find(sub[mid:], False)
        if left or right or out_of_budget[0]:
            return left + right
        # each half builds, together they do not: reduce to a minimal failing set
        core = list(sub)
        for x in list(sub):
            if len(core) <= 2:
                break
            n[0] += 1
            ok = test(f"isolate-{n[0]:02d}", [g for g in core if g != x])
            if ok is None:
                out_of_budget[0] = True
                break
            if not ok:
                core.remove(x)
        evidence.append({"jointly_inconsistent": core})
        return core

    culprits = sorted(set(find(suspects, True)))
    if out_of_budget[0]:
        return culprits, "unattributed in its layer"
    return culprits, "isolated: fails on top of the rest of the layer"
