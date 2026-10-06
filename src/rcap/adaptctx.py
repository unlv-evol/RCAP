"""What the names a failed build could not resolve ARE: context for one re-prompt.

Re-prompting with the compiler errors alone does not help (rounds 1-3: the
model reverts to the fork's code), because the request never shows the model
what it would need to know to adapt: what the missing name is upstream, and
what the fork has instead. This module builds that, per missing name, as
`rcap.reprompt.SymbolContext`:

added_by_change    the name is DECLARED on an added line of the same upstream
                   change, outside the unit. Shown: the upstream definition
                   (signature + body when it is at most MAX_BODY_LINES lines,
                   else doc comment + signature), stated as not present in the
                   fork. The model is asked to inline a small helper, or to
                   decline naming it.
renamed_by_change  the change renames a member and the fork still declares the
                   old name. Shown: the fork's declaration; keep its name.
moved_by_change    the change moves a type to another package and the fork
                   still declares it at the old place: keep the fork's import.
predates_change    the name is declared upstream BEFORE the change and the fork
                   lacks it in this form. Shown: the upstream declaration and
                   the fork's own members of the receiver type (same-named
                   overloads first, then the closest names); for a type, where
                   the fork declares a type of that name.
unknown            none of the above could be established; said so.

Deterministic. No git and no build system here: the driver supplies readers
for the upstream tree before/after the change and for the fork (`Sources`),
the parsed diff of the change, and where the unit sits in it. Parsing is
SALP's tree-sitter layer, as everywhere else.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from salp.structural import grammar_for, parse

from rcap import cochange as cc
from rcap.feedback import CompileError, MissingSymbol, owner_types, rank_members
from rcap.reprompt import SymbolContext

MAX_BODY_LINES = 25       # a definition up to this long is shown with its body
MAX_DOC_LINES = 12        # doc comment lines shown with a signature-only definition
MAX_FORK_MEMBERS = 6
MAX_SYMBOLS = 6
MAX_SIGNATURE_CHARS = 400

_WS = re.compile(r"\s+")
_KEYWORDS = frozenset({"this", "super", "new", "return", "if", "else", "for", "while"})


@dataclass
class Sources:
    """How the builder reads the three trees. Every reader returns None for a
    path / type it does not have."""

    upstream_after: Callable[[str], str | None]      # upstream file after the change
    upstream_before: Callable[[str], str | None]     # upstream file before the change
    fork: Callable[[str], str | None]                # fork file, unmodified
    upstream_type_files: Callable[[str], list[str]]  # type name -> upstream paths (before the change)
    fork_type_files: Callable[[str], list[str]]      # type name -> fork paths
    # identifier -> does it occur anywhere in the fork's sources? (None: not checked)
    fork_has_word: Callable[[str], bool] | None = None


@dataclass
class Declaration:
    name: str
    kind: str            # method | field | type
    signature: str       # header up to the body, whitespace-collapsed
    text: str            # the whole declaration, dedented
    doc: str             # doc comment directly above it, dedented ("" if none)
    first_line: int      # 1-based
    last_line: int
    params: int = 0

    @property
    def lines(self) -> int:
        return self.last_line - self.first_line + 1


def _lang(path_or_lang: str | None) -> str:
    ext = (path_or_lang or "java").rsplit(".", 1)[-1].lower()
    return "scala" if ext in ("scala", "sc") else "java"


def _dedent(raw: bytes, node) -> str:
    """Node text with the indentation of its first line removed from every line."""
    text = raw[node.start_byte:node.end_byte].decode(errors="replace")
    col = node.start_point[1]
    lines = text.split("\n")
    return "\n".join([lines[0]] + [ln[col:] if ln[:col].strip() == "" else ln.lstrip()
                                   for ln in lines[1:]])


def _name_of(node, raw: bytes, grammar) -> list[str]:
    ident = node.child_by_field_name("name")
    if ident is not None:
        return [raw[ident.start_byte:ident.end_byte].decode(errors="replace")]
    names = []
    if node.type in grammar.field_like:
        # Java: field_declaration -> variable_declarator(name); Scala: val_definition(pattern)
        for child in node.named_children:
            sub = child.child_by_field_name("name")
            if sub is not None:
                names.append(raw[sub.start_byte:sub.end_byte].decode(errors="replace"))
        pat = node.child_by_field_name("pattern")
        if pat is not None:
            names.append(raw[pat.start_byte:pat.end_byte].decode(errors="replace"))
    return names


def declarations(source: str | None, language: str | None,
                 name: str | None = None) -> list[Declaration]:
    """Methods, fields and types a source file declares (all, or those named `name`)."""
    grammar = grammar_for(_lang(language))
    tree = parse(source, grammar) if source and grammar is not None else None
    if tree is None:
        return []
    raw = source.encode()
    wanted = grammar.method_like | grammar.field_like | grammar.class_like \
        | {"enum_constant", "annotation_type_declaration"}
    out: list[Declaration] = []
    stack = [tree.root_node]
    while stack:
        node = stack.pop()
        stack.extend(reversed(node.children))
        if node.type not in wanted:
            continue
        names = _name_of(node, raw, grammar)
        if not names or (name is not None and name not in names):
            continue
        kind = ("method" if node.type in grammar.method_like else
                "type" if node.type in grammar.class_like or node.type == "annotation_type_declaration"
                else "field")
        body = node.child_by_field_name("body")
        head_end = body.start_byte if body is not None and kind != "field" else node.end_byte
        signature = _WS.sub(" ", raw[node.start_byte:head_end].decode(errors="replace")).strip()
        prev = node.prev_named_sibling
        doc = ""
        if prev is not None and "comment" in prev.type \
                and node.start_point[0] - prev.end_point[0] <= 1:
            doc = _dedent(raw, prev)
        params = 0
        for child in node.children:
            if child.type in grammar.parameters:
                params += sum(1 for c in child.named_children if c.type in grammar.parameter)
        for n in ([name] if name is not None else names):
            out.append(Declaration(
                name=n, kind=kind, signature=signature[:MAX_SIGNATURE_CHARS],
                text=_dedent(raw, node), doc=doc, first_line=node.start_point[0] + 1,
                last_line=node.end_point[0] + 1, params=params))
    return out


def definition_at(source: str | None, language: str | None, name: str,
                  line: int) -> Declaration | None:
    """The declaration of `name` that holds 1-based `line` (else the nearest one)."""
    found = declarations(source, language, name)
    if not found:
        return None
    holding = [d for d in found if d.first_line <= line <= d.last_line]
    pool = holding or found
    # the innermost declaration holding the line; else the closest by start line
    return min(pool, key=lambda d: (d.lines if holding else abs(d.first_line - line)))


def infer_owner(usage: str, name: str, scope_text: str, default_type: str | None) -> str | None:
    """The receiver type of `name` in the offending line, when it can be read off
    the text: `Type.name` -> Type; `this.name` / a bare `name` -> the enclosing
    type; `var.name` -> the declared type of `var` in `scope_text` when it is
    declared there with an explicit type. None otherwise: never guessed."""
    m = re.search(rf"([A-Za-z_$][\w$]*)\s*(?:\([^()]*\))?\s*\.\s*{re.escape(name)}(?![\w$])", usage)
    if m is None:
        if re.search(rf"\bnew\s+{re.escape(name)}\b", usage) or name[:1].isupper():
            return None                      # the name is itself a type
        if re.search(rf"\.\s*{re.escape(name)}(?![\w$])", usage):
            return None                      # a receiver expression we cannot type
        return default_type
    receiver = m.group(1)
    if receiver in ("this", "super"):
        return default_type
    if receiver[:1].isupper():
        return receiver
    if receiver in _KEYWORDS:
        return None
    java = re.search(rf"(?<![\w$.])([A-Z][\w$]*)(?:<[^;=(){{}}]*>)?(?:\[\])*\s+{re.escape(receiver)}\s*[=;,):]",
                     scope_text)
    if java:
        return java.group(1)
    scala = re.search(rf"\b{re.escape(receiver)}\s*:\s*(?:[a-z][\w.]*\.)?([A-Z][\w$]*)", scope_text)
    if scala:
        return scala.group(1)
    return None


def _pick_file(paths: Sequence[str], near: str | None) -> str | None:
    """One file among several declaring a type name: same Gradle project as
    `near` first, production before test."""
    if not paths:
        return None
    if len(paths) == 1:
        return paths[0]
    root = (near or "").split("/src/")[0]
    same = [p for p in paths if p.split("/src/")[0] == root]
    main = [p for p in (same or paths) if "/src/main/" in p]
    pool = main or same or list(paths)
    return pool[0] if len(pool) == 1 else None


def _package(path: str) -> str:
    m = re.search(r"/src/\w+/(?:java|scala)/(?:unit/|integration/)?(.+)/[^/]+$", path)
    return m.group(1).replace("/", ".") if m else path


def _fork_members(name: str, text: str, path: str) -> list[str]:
    """Fork-side signatures for a missing member: every declaration of the same
    name (overloads) first, then the closest other names."""
    decls = [d for d in declarations(text, path) if d.kind != "type"]
    out: list[str] = []
    for d in decls:
        if d.name == name and d.signature not in out:
            out.append(d.signature)
    others = [(d.name, d.signature) for d in decls if d.name != name]
    for sig in rank_members(name, others, limit=MAX_FORK_MEMBERS):
        if sig not in out:
            out.append(sig)
    return out[:MAX_FORK_MEMBERS]


_CODE_NOISE = re.compile(r'/\*.*?\*/|//[^\n]*|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])\'', re.DOTALL)
_IDENT = re.compile(r"(?<![\w$.])[A-Za-z_$][\w$]*")          # not a member selected with `.`
_ANY_IDENT = re.compile(r"[A-Za-z_$][\w$]*")
_IMPORT = re.compile(r"^\s*import\s+(static\s+)?([\w.]+)\s*;?\s*$", re.MULTILINE)
_LANG_WORDS = frozenset("""
abstract assert boolean break byte case catch char class const continue default do double else enum
extends final finally float for goto if implements import instanceof int interface long native new
package private protected public return short static strictfp super switch synchronized this throw
throws transient try void volatile while true false null var def val object trait match with yield
lazy override implicit sealed type String Object Integer Long Boolean Double Short Byte Character
Exception RuntimeException Throwable Override Deprecated SuppressWarnings Math System Class Void
""".split())


def inlining_needs(definition: str, name: str, upstream_file_text: str | None,
                   unit_file_text: str | None, sources: Sources) -> tuple[list[str], list[str]]:
    """What inlining a helper's body into a unit of another file would need:
    (names the definition uses that occur nowhere in the fork, fully-qualified
    types it uses that the unit's file does not import). Token level, like the
    screen that proposed inlining: a name that occurs somewhere in the fork is
    not proven accessible from the unit."""
    code = _CODE_NOISE.sub(" ", definition)
    declared = set(cc.declared_names(code.splitlines())) | {name}
    declared |= set(re.findall(r"[\w>\]]\s+([a-z_$][\w$]*)\s*[,)=:;]", code))   # parameters / locals
    missing: list[str] = []
    if sources.fork_has_word is not None:
        for ident in dict.fromkeys(_ANY_IDENT.findall(code)):
            if ident in _LANG_WORDS or ident in declared or len(ident) < 2:
                continue
            if not sources.fork_has_word(ident):
                missing.append(ident)
    qualify: list[str] = []
    used = set(_IDENT.findall(code))
    unit = unit_file_text or ""
    unit_imports = {m.group(2) for m in _IMPORT.finditer(unit)}
    unit_simple = {i.rsplit(".", 1)[-1] for i in unit_imports}
    pkg = re.search(r"^\s*package\s+([\w.]+)", unit, re.MULTILINE)
    up_pkg = re.search(r"^\s*package\s+([\w.]+)", upstream_file_text or "", re.MULTILINE)
    candidates = [(m.group(2), bool(m.group(1))) for m in _IMPORT.finditer(upstream_file_text or "")]
    for fqn, static in candidates:
        simple = fqn.rsplit(".", 1)[-1]
        if static or simple not in used or simple in unit_simple or simple in missing:
            continue
        if pkg and fqn.rsplit(".", 1)[0] == pkg.group(1):
            continue
        if fqn.rsplit(".", 1)[0] + ".*" in unit_imports or fqn.rsplit(".", 1)[0] + "._" in unit_imports:
            continue
        if fqn not in qualify:
            qualify.append(fqn)
    # same-package types of the helper's file need qualifying from another package
    if up_pkg and pkg and up_pkg.group(1) != pkg.group(1):
        for simple in sorted(used):
            if not simple[:1].isupper() or simple in unit_simple or simple in _LANG_WORDS \
                    or any(q.endswith("." + simple) for q in qualify) or simple in missing:
                continue
            if any(_package(p) == up_pkg.group(1) for p in sources.fork_type_files(simple)):
                qualify.append(f"{up_pkg.group(1)}.{simple}")
    return missing, qualify


@dataclass
class _Ctx:
    files: Sequence[cc.FileDiff]
    region: cc.UnitRegion | None
    unit_file: str
    source_file: str
    sources: Sources
    scope_text: str
    renames: Sequence[object]
    hints: dict[str, list[str]] = field(default_factory=dict)
    owners: dict[str, str] = field(default_factory=dict)


def _describe(m: MissingSymbol, usage: str, c: _Ctx) -> SymbolContext:
    name = m.name
    default_type = c.unit_file.rsplit("/", 1)[-1].split(".")[0]
    inferred = infer_owner(usage, name, c.scope_text, default_type) if usage else None
    owner = (owner_types(m.owner) or [None])[0] or inferred or c.owners.get(name)
    owner_how = ("compiler" if m.owner else "inferred from the offending line" if inferred
                 else "supplied by the caller" if c.owners.get(name) else None)
    if owner and not owner[:1].isupper():
        owner, owner_how = None, None        # a package segment, not a type
    if m.kind in ("class", "import") and name[:1].isupper():
        owner, owner_how = None, None        # the missing name is itself a type
    if owner is None and m.kind == "signature" and name[:1].isupper():
        owner, owner_how = name, "constructor of the named type"
    is_type = owner is None and name[:1].isupper() and "." not in name
    sc = SymbolContext(symbol=name, owner=owner if owner != name else None,
                       usage=usage.strip()[:200], hints=list(c.hints.get(name, [])),
                       why={"compiler_kind": m.kind, "owner_from": owner_how})

    def fork_side() -> None:
        if is_type or (owner == name and name[:1].isupper()):
            paths = c.sources.fork_type_files(name)
            if owner == name and paths:          # a constructor / signature of a fork type
                path = _pick_file(paths, c.unit_file)
                text = c.sources.fork(path) if path else None
                if text:
                    sc.fork_heading = f"The fork's {name} ({path}) declares"
                    sc.fork_members = [d.signature for d in declarations(text, path, name)
                                       if d.kind != "type"][:MAX_FORK_MEMBERS] \
                        or [d.signature for d in declarations(text, path, name)][:1]
                    return
            # how the source file gets the type, and whether the fork could get it the same way
            fqn = next((m_.group(2) for m_ in _IMPORT.finditer(c.sources.upstream_after(c.source_file) or "")
                        if not m_.group(1) and m_.group(2).rsplit(".", 1)[-1] == name), None)
            sc.why["upstream_import"] = fqn
            here = [p for p in paths if fqn and _package(p) == fqn.rsplit(".", 1)[0]]
            if fqn and (fqn.startswith(("java.", "javax.", "scala.")) or here):
                sc.fork_heading = (
                    f"The source file imports it as `{fqn}`. That type is available to the fork ("
                    + ("JDK / Scala library" if not here else here[0])
                    + "), but this file does not import it: write it fully qualified.")
            elif paths:
                sc.fork_heading = f"The fork declares a type `{name}` in"
                sc.fork_members = [f"{_package(p)} ({p})" for p in paths[:MAX_FORK_MEMBERS]]
                if fqn:
                    sc.fork_members.append(f"(the source file imports it as {fqn}, which the fork lacks)")
            else:
                sc.fork_heading = f"The fork declares no type named `{name}`." + (
                    f" The source file imports it as `{fqn}`." if fqn else "")
            return
        if owner is None:
            sc.fork_heading = ("The receiver type of this name could not be determined, so the "
                               "fork's members are not listed.")
            return
        path = _pick_file(c.sources.fork_type_files(owner), c.unit_file)
        text = c.sources.fork(path) if path else None
        if text is None:
            sc.fork_heading = f"The fork declares no type named `{owner}`."
            return
        members = _fork_members(name, text, path)
        sc.why["fork_file"] = path
        if members:
            sc.fork_heading = f"The fork's {owner} ({path}) declares"
            sc.fork_members = members
        else:
            sc.fork_heading = (f"The fork's {owner} ({path}) declares no member named "
                               f"`{name}` or close to it.")

    # 1. renamed by the same change, and the fork still has the old name
    for r in c.renames:
        if r.new != name:
            continue
        path = _pick_file(c.sources.fork_type_files(owner), c.unit_file) if owner else None
        text = (c.sources.fork(path) if path else None) or c.sources.fork(c.unit_file) or ""
        olds = [d.signature for d in declarations(text, path or c.unit_file, r.old)]
        if olds or re.search(rf"(?<![\w$]){re.escape(r.old)}(?![\w$])", text):
            decl = next((h for f in c.files for h in f.hunks if h.hunk_id == r.decl_hunk), None)
            sc.origin, sc.old_name = "renamed_by_change", r.old
            sc.upstream_file = decl.path if decl is not None else None
            sc.fork_heading = (f"The fork's declaration ({path})" if olds and path
                               else f"The fork still uses `{r.old}`")
            sc.fork_members = olds[:2]
            return sc

    # 2b. a type the same change moves (file renamed, or deleted here and added
    # there) while the fork still has it at the old place; or a type it adds
    if is_type:
        stem = lambda p: (p or "").rsplit("/", 1)[-1].split(".")[0]   # noqa: E731
        new_files = [f for f in c.files if f.status in ("A", "R") and stem(f.new_path) == name]
        if new_files:
            f = new_files[0]
            fork_paths = c.sources.fork_type_files(name)
            moved = f.status == "R" or any(g.status == "D" and stem(g.old_path) == name
                                           for g in c.files)
            sc.upstream_file = f.new_path
            if moved and fork_paths:
                sc.origin = "moved_by_change"
                sc.old_name = _package(fork_paths[0]) + "." + name
                sc.fork_heading = f"The fork declares `{name}` in"
                sc.fork_members = [f"{_package(p)} ({p})" for p in fork_paths[:MAX_FORK_MEMBERS]]
                return sc
            sc.origin = "added_by_change"
            d = next((x for x in declarations(c.sources.upstream_after(f.new_path), f.new_path, name)
                      if x.kind == "type"), None)
            if d is not None:
                doc = "\n".join(d.doc.splitlines()[:MAX_DOC_LINES])
                sc.upstream_text = sc.upstream_signature = (doc + "\n" if doc else "") + d.signature
                sc.upstream_lines = d.lines
                sc.why["declaration_kind"] = "type"
            fork_side()
            if not sc.fork_members:
                sc.fork_heading = ""
            return sc

    # 2. declared on an added line of the same change, outside the unit
    ev = cc.find_evidence([name], c.files, c.region).get(name, [])
    decls = [e for e in ev if e.declaration]
    if decls:
        e = next((x for x in decls if owner and x.file.rsplit("/", 1)[-1].split(".")[0] == owner),
                 decls[0])
        d = definition_at(c.sources.upstream_after(e.file), e.file, name, e.line)
        sc.origin, sc.upstream_file, sc.upstream_line = "added_by_change", e.file, e.line
        if d is not None:
            sc.upstream_signature, sc.upstream_lines = d.signature, d.lines
            doc = "\n".join(d.doc.splitlines()[:MAX_DOC_LINES])
            if d.lines <= MAX_BODY_LINES and d.kind != "type":
                sc.upstream_text, sc.upstream_complete = d.text, True
                sc.missing_in_fork, sc.qualify = inlining_needs(
                    d.text, name, c.sources.upstream_after(e.file), c.sources.fork(c.unit_file),
                    c.sources)
                sc.names_checked = c.sources.fork_has_word is not None
            else:
                sc.upstream_text = (doc + "\n" if doc else "") + d.signature
                sc.upstream_signature = sc.upstream_text
            sc.why["declaration_kind"] = d.kind
        else:
            sc.upstream_text = sc.upstream_signature = e.text
        # the fork may have the name with another signature (the change alters it)
        if owner and not is_type:
            path = _pick_file(c.sources.fork_type_files(owner), c.unit_file)
            text = c.sources.fork(path) if path else None
            same = [x.signature for x in declarations(text, path, name)] if text else []
            if same:
                sc.fork_heading = (f"The fork's {owner} ({path}) has `{name}` only with a "
                                   "different declaration")
                sc.fork_members = same[:MAX_FORK_MEMBERS]
        return sc

    # 3. declared upstream before the change
    before_paths = c.sources.upstream_type_files(name if is_type else (owner or ""))
    up_path = _pick_file(before_paths, c.source_file)
    if is_type:
        if before_paths:
            sc.origin = "predates_change"
            sc.upstream_file = up_path or before_paths[0]
            sc.upstream_signature = f"type {name} in package {_package(sc.upstream_file)}"
    elif up_path:
        found = [d for d in declarations(c.sources.upstream_before(up_path), up_path, name)]
        found = [d for d in found if d.kind != "type"] or found
        if found:
            sc.origin, sc.upstream_file = "predates_change", up_path
            sc.upstream_signature = "; ".join(d.signature for d in found[:2])
            sc.upstream_line = found[0].first_line
    if sc.origin == "unknown" and c.region is not None and re.search(
            rf"(?<![\w$]){re.escape(name)}(?![\w$])", c.region.before_text or ""):
        # upstream used it in this very function before the change: it predates
        # the change even though its declaration was not located
        sc.origin = "predates_change"
        sc.why["predates_evidence"] = "used by the source function before the change"
    fork_side()
    return sc


_IMPORT_LINE = re.compile(r"^\s*import\s+(?:static\s+)?([\w.]+?)(?:\.\{|\._|\.\*)?\s*;?\s*$")
_NO_PACKAGE = re.compile(r"^package ([\w.]+) does not exist|"
                         r"^object (\w+) is not a member of package ([\w.]+)")


def import_symbols(errors: Sequence[CompileError]) -> list[MissingSymbol]:
    """Types an import could not resolve (`package p does not exist`, `object X
    is not a member of package p`): rcap.feedback.missing_symbols does not name
    them, yet a moved or relocated type is exactly this error."""
    out: list[MissingSymbol] = []
    seen: set[str] = set()
    for e in errors:
        if not _NO_PACKAGE.search(e.message.strip()):
            continue
        line = next((b for b in e.body if b.strip().startswith("import")), "")
        m = _IMPORT_LINE.match(line)
        name = m.group(1).rsplit(".", 1)[-1] if m else None
        if name and name[:1].isupper() and name not in seen:
            seen.add(name)
            out.append(MissingSymbol(name=name, kind="import", owner=None,
                                     path=e.rel or e.path, line=e.line))
    return out


def build_symbol_context(
    missing: Sequence[MissingSymbol], errors: Sequence[CompileError], *,
    files: Sequence[cc.FileDiff], region: cc.UnitRegion | None,
    unit_file: str, source_file: str, sources: Sources, scope_text: str = "",
    renames: Sequence[object] = (), hints: dict[str, list[str]] | None = None,
    owners: dict[str, str] | None = None, max_symbols: int = MAX_SYMBOLS,
) -> tuple[list[SymbolContext], int]:
    """Describe the missing names of one failed build for one unit.

    `missing` / `errors`: rcap.feedback.missing_symbols and the compile errors
    they came from (for the offending source line). `files`: the upstream
    change as parsed by rcap.cochange; `region`: where the unit sits in it, so
    the unit's own added lines are not taken for a co-change. `scope_text`:
    the unit's candidate and the fork file, for typing a receiver variable.
    `renames`: rcap.prorder.find_renames of the change. `hints[name]`: extra
    fork-side lines from another deterministic analysis; `owners[name]`: a
    receiver type to fall back on. Returns (contexts, names not described)."""
    ctx = _Ctx(files=files, region=region, unit_file=unit_file, source_file=source_file,
               sources=sources, scope_text=scope_text, renames=renames,
               hints=dict(hints or {}), owners=dict(owners or {}))
    out: list[SymbolContext] = []
    seen: set[str] = set()
    for m in missing:
        if m.name in seen:
            continue
        seen.add(m.name)
        err = next((e for e in errors if e.line == m.line
                    and (e.rel or e.path) == m.path), None)
        usage = next((b for b in (err.body if err else [])
                      if b.strip() and b.strip() != "^" and not b.lstrip().startswith(
                          ("symbol:", "location:", "required:", "found:", "reason:"))), "")
        out.append(_describe(m, usage, ctx))
    return out[:max_symbols], max(0, len(out) - max_symbols)
