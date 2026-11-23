"""Deterministic import post-pass for a ported Java / Scala file.

A unit is usually a function; the import block is usually outside it. A
candidate that faithfully uses a type the upstream change starts using then
fails to compile for a reason no edit to the function can fix. This pass
closes exactly that gap and nothing else:

    a simple name the candidate ADDS to the file,
    that the upstream AFTER version of the same file imports explicitly,
    that the fork file neither imports, declares nor already resolves,
    and whose import target exists for the fork (the caller decides how:
    `exists(import)`),

gets that upstream import added to the fork file. Every edit is returned as a
record (`added`), and every name that was considered and left alone is
returned with the reason (`skipped`). Nothing is guessed: wildcard imports are
never added, an import whose target cannot be shown to exist is not added.

Removal of imports the candidate made unused is OFF unless the caller says an
unused import fails the build it validates with (`remove_unused=True`): see
`unused_import_fails_build`.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field

from rcap.fidelity import tokens

_JAVA_IMPORT = re.compile(r"^[ \t]*import[ \t]+(static[ \t]+)?([\w$.]+(?:\.\*)?)[ \t]*;[ \t]*$")
_SCALA_IMPORT = re.compile(r"^[ \t]*import[ \t]+(.+?)[ \t]*$")
_PACKAGE = re.compile(r"^[ \t]*package[ \t]+([\w.]+)[ \t]*;?[ \t]*$")
_IDENT = re.compile(r"^[A-Za-z_$][\w$]*$")
_DECL = r"\b(?:class|interface|enum|trait|object|record|type|def|val|var)\s+{name}\b"


@dataclass(frozen=True)
class Import:
    line: int                 # 0-based line index in the file
    fqn: str                  # a.b.C (static: a.b.C.member); wildcard keeps its package
    simple: str               # the name it brings into scope ("*" for a wildcard)
    static: bool = False
    wildcard: bool = False
    original: str | None = None   # Scala rename `{original => simple}`
    grouped: bool = False     # one of several selectors on a Scala line

    def render(self, lang: str) -> str:
        if lang == "java":
            return f"import {'static ' if self.static else ''}{self.fqn};"
        if self.original and self.original != self.simple:
            pkg = self.fqn.rsplit(".", 1)[0]
            return f"import {pkg}.{{{self.original} => {self.simple}}}"
        return f"import {self.fqn}"


def parse_imports(text: str, lang: str) -> list[Import]:
    """Top-level import statements. Scala selectors are expanded one per name."""
    out: list[Import] = []
    for n, line in enumerate(text.splitlines()):
        if lang == "java":
            m = _JAVA_IMPORT.match(line)
            if not m:
                continue
            fqn = m.group(2)
            wild = fqn.endswith(".*")
            if wild:
                fqn = fqn[:-2]
            out.append(Import(n, fqn, "*" if wild else fqn.rsplit(".", 1)[-1],
                              static=bool(m.group(1)), wildcard=wild))
            continue
        m = _SCALA_IMPORT.match(line)
        if not m or line[:1] in " \t":      # an indented Scala import is local to a scope
            continue
        for clause in _split_top(m.group(1).split("//")[0].strip()):
            clause = clause.strip()
            brace = re.match(r"^([\w$.]+)\.\{(.*)\}$", clause)
            if brace:
                sels = [s.strip() for s in brace.group(2).split(",") if s.strip()]
                for sel in sels:
                    ren = re.match(r"^([\w$]+)\s*=>\s*([\w$]+)$", sel)
                    if sel == "_":
                        out.append(Import(n, brace.group(1), "*", wildcard=True, grouped=len(sels) > 1))
                    elif ren:
                        out.append(Import(n, f"{brace.group(1)}.{ren.group(1)}", ren.group(2),
                                          original=ren.group(1), grouped=len(sels) > 1))
                    elif _IDENT.match(sel):
                        out.append(Import(n, f"{brace.group(1)}.{sel}", sel, grouped=len(sels) > 1))
            elif clause.endswith("._"):
                out.append(Import(n, clause[:-2], "*", wildcard=True))
            elif re.match(r"^[\w$.]+$", clause) and "." in clause:
                out.append(Import(n, clause, clause.rsplit(".", 1)[-1]))
    return out


def _split_top(s: str) -> list[str]:
    out, depth, cur = [], 0, ""
    for ch in s:
        depth += ch == "{"
        depth -= ch == "}"
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    return out + [cur] if cur.strip() else out


def _body_idents(text: str, lang: str) -> Counter[str]:
    """Identifier counts outside import/package lines, comments and strings."""
    keep = [ln for ln in text.splitlines()
            if not re.match(r"^[ \t]*(?:import|package)\b", ln)]
    return Counter(t for t in tokens("\n".join(keep)) if _IDENT.match(t))


def unused_import_fails_build(lang: str, *, checkstyle: bool, scala_fatal_warnings: bool) -> bool:
    """Does an unused import break the build that validation runs?

    javac has no unused-import lint, so for Java only checkstyle's
    `UnusedImports` can fail it. For Scala it takes `-Xlint:unused` (or
    `-Ywarn-unused`) together with `-Xfatal-warnings`. The caller reads both
    facts from the build it actually runs; this function only combines them."""
    return checkstyle if lang == "java" else scala_fatal_warnings


@dataclass
class ImportFix:
    text: str
    added: list[dict[str, object]] = field(default_factory=list)
    removed: list[dict[str, object]] = field(default_factory=list)
    skipped: list[dict[str, object]] = field(default_factory=list)

    @property
    def changed(self) -> bool:
        return bool(self.added or self.removed)

    def as_dict(self) -> dict[str, object]:
        return {"added": self.added, "removed": self.removed, "skipped": self.skipped}


def fix_imports(original: str, merged: str, upstream_after: str, lang: str, *,
                exists: Callable[[Import], str | None],
                remove_unused: bool = False) -> ImportFix:
    """`original`: the fork file; `merged`: the fork file with candidates
    applied; `upstream_after`: the same file upstream after the change.
    `exists(import)` returns WHY the import target is available to the fork
    (a short string that is recorded) or None when that cannot be shown."""
    fix = ImportFix(text=merged)
    if lang not in ("java", "scala"):
        return fix
    up = parse_imports(upstream_after, lang)
    have = parse_imports(merged, lang)
    have_simple = {i.simple for i in have if not i.wildcard}
    have_fqn = {i.fqn for i in have if not i.wildcard}
    wild = {(i.fqn, i.static) for i in have if i.wildcard}
    before, after = _body_idents(original, lang), _body_idents(merged, lang)
    introduced = [n for n in after if after[n] > before.get(n, 0)]
    by_simple: dict[str, list[Import]] = {}
    for i in up:
        if not i.wildcard:
            by_simple.setdefault(i.simple, []).append(i)
    to_add: list[tuple[Import, str]] = []
    for name in introduced:
        cands = by_simple.get(name)
        if not cands:
            continue
        if before.get(name, 0) > 0:
            # the fork file already used this name before the port: it resolves somehow
            continue

        def skip(why: str, imp: Import) -> None:
            fix.skipped.append({"name": name, "import": imp.render(lang), "why": why})

        imp = cands[0]
        if len({c.fqn for c in cands}) > 1:
            skip("upstream imports this simple name from more than one place", imp)
            continue
        if name in have_simple or imp.fqn in have_fqn:
            continue                                    # already imported (possibly from elsewhere)
        if (imp.fqn.rsplit(".", 1)[0], imp.static) in wild or \
                (lang == "scala" and (imp.fqn.rsplit(".", 1)[0], False) in wild):
            continue                                    # covered by a wildcard the fork has
        if re.search(_DECL.format(name=re.escape(name)), merged):
            skip("declared in the file itself", imp)
            continue
        why = exists(imp)
        if why is None:
            skip("import target not shown to exist for the fork", imp)
            continue
        to_add.append((imp, why))

    lines = merged.split("\n")
    if to_add:
        for imp, why in sorted(to_add, key=lambda x: (x[0].static, x[0].fqn)):
            at = _insert_at(lines, lang, imp.static)
            stmt = imp.render(lang)
            lines.insert(at, stmt)
            fix.added.append({"name": imp.simple, "import": stmt, "line": at + 1, "why": why,
                              "kind": "static" if imp.static else "type"})
    if remove_unused:
        now = _body_idents("\n".join(lines), lang)
        drop: list[Import] = []
        for i in parse_imports("\n".join(lines), lang):
            if i.wildcard or i.grouped:
                continue
            if before.get(i.simple, 0) > 0 and now.get(i.simple, 0) == 0:
                drop.append(i)
        for i in sorted(drop, key=lambda x: -x.line):
            fix.removed.append({"name": i.simple, "import": lines[i.line].strip(), "line": i.line + 1,
                                "why": "the port removed its last use and an unused import fails this build"})
            del lines[i.line]
    fix.text = "\n".join(lines)
    return fix


def _insert_at(lines: list[str], lang: str, static: bool) -> int:
    """Index to insert a new import line at: after the last top-level import of
    the same kind, else after the last import, else after the package clause."""
    last_same = last_any = pkg = None
    for n, ln in enumerate(lines):
        if lang == "java":
            m = _JAVA_IMPORT.match(ln)
            if m:
                last_any = n
                if bool(m.group(1)) == static:
                    last_same = n
            elif re.match(r"^[ \t]*(?:(?:public|final|abstract|@\w+)[ \t]+)*(?:class|interface|enum|record)\b", ln):
                break
        else:
            if ln[:1] not in " \t" and _SCALA_IMPORT.match(ln):
                last_any = last_same = n
            elif re.match(r"^(?:(?:private|protected|final|abstract|sealed|case|implicit|@\w+)\S*[ \t]+)*"
                          r"(?:class|object|trait)\b", ln):
                break
        if pkg is None and _PACKAGE.match(ln):
            pkg = n
    if last_same is not None:
        return last_same + 1
    if last_any is not None:
        return last_any + 1
    return (pkg + 1) if pkg is not None else 0
