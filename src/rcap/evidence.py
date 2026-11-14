"""Edit evidence: what else the same upstream change edits, for a function unit's prompt.

Off by default (ExecutionConfig.edit_evidence); off, every context and prompt is
byte-identical to what RCAP wrote before. On, a function unit's context carries

* `other_edits`: the change's edits OUTSIDE functions in the same SAP (imports,
  fields, class or file-level code): for each such unit, the lines its
  source.before -> source.after removes and adds, from the materialized
  payloads (never re-read from disk). These units leave the "signatures only"
  list, where they could only be shown as `?`.
* `pr_functions`: the signatures (never bodies) of the functions the same pull
  request adds, removes or re-signs in its OTHER files (SAPs), read from those
  SAPs' function payloads. Functions whose body alone changes are left out:
  calls to them stay valid, and on large PRs they are most of the list.

Both are SAP evidence only: nothing from the fork, nothing invented. Rendering is
deterministic and capped (`render_sections`), with an explicit truncation note.

ExecutionConfig.evidence_filter = "names" (see "Relevance filter" below) keeps only the
entries whose declared, imported or re-signed names the function uses, for both these
sections and the same file's "signatures only" list; "none" keeps every entry.
"""

from __future__ import annotations

import difflib
import functools
import json
import re
from pathlib import Path

from pydantic import BaseModel

from rcap.unitkind import FILE, FUNCTION, REGION, TYPE, text_kind

_HUNK_HEADER = re.compile(r"^\s*@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@")
_COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*", re.DOTALL)

# Section caps, in characters of the rendered section body (about 3.5 characters per
# token on Java, so together under 3k tokens). Measured uncapped on the 2,604 Java function
# units of study 2026-11-candidates-32k: other edits p95 4.7k / max 23k chars (61 prompts cut
# at 6k); re-signed functions of other files p90 3.6k / max 27k chars (210 prompts cut at 4k).
OTHER_EDITS_CHARS = 6000
PR_FUNCTIONS_CHARS = 4000


class OtherEdit(BaseModel):
    """One non-function unit of the same SAP and the lines its change removes / adds."""

    entity: str
    kind: str            # imports | type | file | region | other
    lines: list[str]     # "- old line" / "+ new line", in diff order, stripped


class PRFunction(BaseModel):
    """A function the same PR edits in another file (SAP): its signature only."""

    file: str
    entity: str
    signature_before: str | None = None
    signature_after: str | None = None

    @property
    def status(self) -> str:
        if self.signature_before is None and self.signature_after is not None:
            return "added"
        if self.signature_after is None and self.signature_before is not None:
            return "removed"
        if self.signature_before != self.signature_after:
            return "signature changed"
        return "body changed"


def is_function_payload(before: str | None, after: str | None, language: str) -> bool:
    """A unit is a function unit when each side is exactly one function or empty
    (an added or removed function), and at least one side is a function."""
    kb, ka = text_kind(before, language), text_kind(after, language)
    return FUNCTION in (kb, ka) and (kb == FUNCTION or not (before or "").strip()) \
        and (ka == FUNCTION or not (after or "").strip())


def change_lines(before: str, after: str) -> list[str]:
    """The lines a before/after pair removes ("- ") and adds ("+ "), in diff
    order, whitespace-stripped; blank lines and diff-excerpt headers ignored."""
    a, b = before.splitlines(), after.splitlines()
    out: list[str] = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        out.extend("- " + x.strip() for x in a[i1:i2] if x.strip() and not _HUNK_HEADER.match(x))
        out.extend("+ " + x.strip() for x in b[j1:j2] if x.strip() and not _HUNK_HEADER.match(x))
    return out


def _kind_label(entity: str, before: str, after: str, language: str) -> str:
    if entity.endswith("_imports"):
        return "imports"
    kinds = {text_kind(before, language), text_kind(after, language)}
    for k in (TYPE, FILE, REGION):
        if k in kinds:
            return {TYPE: "class/type level", FILE: "file level", REGION: "file excerpt"}[k]
    return "outside functions"


def other_edits(sibling_entities, materialized, language: str) -> tuple[list[OtherEdit], set[str]]:
    """(the non-function units among `sibling_entities` with their changed
    lines, the set of those entities). Order is the siblings' order."""
    edits: list[OtherEdit] = []
    moved: set[str] = set()
    for entity in sibling_entities:
        roles = materialized.by_role(entity)
        before = roles.get("source.before")
        after = roles.get("source.after")
        if before is None or after is None:
            continue
        if is_function_payload(before.content, after.content, language):
            continue
        moved.add(entity)
        lines = change_lines(before.content, after.content)
        if lines:
            edits.append(OtherEdit(entity=entity.split("/", 1)[-1],
                                   kind=_kind_label(entity, before.content, after.content, language),
                                   lines=lines))
    return edits, moved


def declaration_of(text: str) -> str | None:
    """A function's declaration: its text up to the body's `{` (or the `;` of a
    bodyless declaration) at parenthesis depth 0, so a brace in an annotation
    argument does not end it; comments dropped (a moved or reworded javadoc is
    not a signature change), whitespace collapsed. None without a `(`."""
    text = _COMMENT.sub(" ", text)
    depth, quote, prev = 0, None, ""
    for i, ch in enumerate(text):
        if quote:
            if ch == quote and prev != "\\":
                quote = None
        elif ch in "\"'":
            quote = ch
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch in "{;" and depth <= 0:
            head = re.sub(r"\s+", " ", text[:i]).strip()
            return head if "(" in head else None
        prev = ch
    return None


@functools.lru_cache(maxsize=4096)
def _sap_functions(sap_dir: str) -> tuple[PRFunction, ...]:
    """The function units of one SAP with their before/after signatures."""
    d = Path(sap_dir)
    try:
        man = json.loads((d / "sap.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ()
    src = man.get("source_file") or man.get("target_file") or ""
    language = src.rsplit(".", 1)[-1].lower() if "." in src.rsplit("/", 1)[-1] else "java"
    out = []
    for fn in man.get("functions") or []:
        fdir = d / "functions" / fn
        texts = {}
        for role in ("source.before", "source.after"):
            f = next((p for p in sorted(fdir.glob(role + ".*")) if p.is_file()), None)
            try:
                texts[role] = f.read_bytes().decode("utf-8") if f else ""
            except (OSError, UnicodeDecodeError):
                texts[role] = ""
        b, a = texts["source.before"], texts["source.after"]
        if b == a or not is_function_payload(b, a, language):
            continue
        sb = declaration_of(b) if b.strip() else None
        sa = declaration_of(a) if a.strip() else None
        if sb is None and sa is None:
            continue
        out.append(PRFunction(file=src.rsplit("/", 1)[-1], entity=fn,
                              signature_before=sb, signature_after=sa))
    return tuple(out)


def pr_functions(sap_dir, pr_manifest: dict | None) -> list[PRFunction]:
    """Function units of the PR's OTHER SAPs whose signature the change alters:
    signature changed, then removed, then added; each group in pr.json order,
    then each SAP's function order."""
    if not pr_manifest:
        return []
    here = Path(sap_dir).resolve()
    pr_dir = here.parent
    found: list[PRFunction] = []
    for s in pr_manifest.get("saps") or []:
        rel = str(s.get("path") or "").rstrip("/")
        if not rel:
            continue
        other = (pr_dir / rel).resolve()
        if other == here or not other.is_dir():
            continue
        found.extend(_sap_functions(str(other)))
    order = ("signature changed", "removed", "added")
    return [f for status in order for f in found if f.status == status]


def _capped(lines: list[str], cap: int, what: str) -> list[str]:
    out, used = [], 0
    for n, line in enumerate(lines):
        if used + len(line) + 1 > cap:
            out.append(f"[... {len(lines) - n} more {what} not shown: cut at {cap} characters]")
            return out
        out.append(line)
        used += len(line) + 1
    return out


def render_sections(context, lang: str) -> list[str]:
    """The edit-evidence prompt sections of a context (empty when it has none)."""
    sections: list[str] = []
    filtered = bool(getattr(context, "evidence_filtered", False))
    edits = getattr(context, "other_edits", None) or []
    if edits:
        body: list[str] = []
        for e in edits:
            body.append(f"# {e.entity} ({e.kind})")
            body.extend(e.lines)
        body = _capped(body, OTHER_EDITS_CHARS, "lines")
        sections.append(
            "## Other edits made by this same change (outside functions)\n"
            "The same upstream change also makes these edits in this file, outside any function\n"
            "(lines removed `-` and added `+`). They are ported separately: do not copy them into\n"
            "your answer, but the function may use what they introduce (imports, fields, types).\n"
            + ("Only the edits that mention a name this function uses are shown.\n" if filtered else "")
            + "```diff\n" + "\n".join(body) + "\n```")
    funcs = getattr(context, "pr_functions", None) or []
    if funcs:
        lines = []
        for f in funcs:
            if f.status == "signature changed":
                lines.append(f"- {f.file}: `{f.signature_after}` (changed from `{f.signature_before}`)")
            elif f.status == "removed":
                lines.append(f"- {f.file}: `{f.signature_before}` (removed)")
            else:
                lines.append(f"- {f.file}: `{f.signature_after}`" + (" (added)" if f.status == "added" else ""))
        lines = _capped(lines, PR_FUNCTIONS_CHARS, "functions")
        sections.append(
            "## Functions this same change adds, removes or re-signs in other files (signatures only)\n"
            "Calls to these must match their signatures after the change."
            + (" Only functions whose name this function uses are listed." if filtered else "")
            + "\n"
            + "\n".join(lines))
    return sections


# ---------------------------------------------------------------------------------------------
# Relevance filter (ExecutionConfig.evidence_filter = "names"; "none" = every entry, as in 08e7a59).
#
# SALP emits only `aligned_to` edges, so semantic reduction cannot tell which of a change's other
# edits matter to the function being ported. This filter keeps an evidence entry only when a name
# it declares, imports or re-signs is a name the function uses: an identifier of the function's
# source.before, source.after or fork target (comments and string literals removed). Measured on
# the 2,604 units of study 2026-11-candidates-32k: restricting the names to the upstream DIFF lines
# lost callees whose call is unchanged but whose declaration gained or lost a `throws` (e.g.
# bitcoinj Wallet.decrypt, MnemonicCode.toMnemonic), so the whole function counts.
#
# Other edits (non-function units of the same SAP) are split into statements (an import, a field,
# an annotation with what it annotates, a multi-line initializer up to its `;`, a body up to its
# closing brace; comment lines go with the statement after them):
# * import: its key is the simple name it imports (`import a.b.C;` -> C, `import static a.B.m;`
#   -> m); kept when the function uses C or c (the `@Autowired C c;` field convention).
# * declaration (field, constant, method or class header, enum constant): its keys are the names
#   it declares (a class header: every name in it); kept when the function uses one.
# * anything else (a fragment of code SALP left outside functions, e.g. a call site elsewhere):
#   kept only when it mentions a name the function's own change introduces or removes (the same
#   replacement made elsewhere), not merely a name both versions use (`andExpect`, `jsonPath`).
# A removed and an added statement of one run that are equal up to whitespace (moved or
# reformatted) are dropped. A removed and an added statement of one run that share a key are
# kept together, so a modified declaration shows both forms.
#
# Function signatures (the same file's "signatures only" list and the other files' list): kept when
# the method / constructor name before or after the change is a name the function uses. Dropped:
# test methods (`@Test` and the like: nothing calls them), entries whose change touches only
# annotations (`@Override` added: callers are unaffected), and same-file functions whose signature
# does not change when they match only through the function's own name (an overload).
#
# Nothing survives -> the section is left out. Deterministic: sets only decide keep / drop; entries
# keep their original order. The caps of render_sections still apply after the filter.
# ---------------------------------------------------------------------------------------------

EVIDENCE_FILTERS = ("none", "names")

_IDENT = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
_LITERAL = re.compile(r'"""(?:.|\n)*?"""|"(?:\\.|[^"\\\n])*"|\'(?:\\.|[^\'\\\n])*\'')
_ANNOTATION = re.compile(r"@\s*(?!interface\b)[A-Za-z_$][\w$.]*(?:\s*\((?:[^()]|\([^()]*\))*\))?")
_MODIFIERS = re.compile(r"^(?:(?:public|private|protected|static|final|abstract|transient|volatile|"
                        r"synchronized|native|default|strictfp|sealed|non-sealed)\s+)*")
_TYPE_DECL = re.compile(r"^(?:class|interface|enum|record|@interface)\s+([A-Za-z_$][\w$]*)")
_ENUM_CONSTANT = re.compile(r"^([A-Z][A-Z0-9_]*)\s*(?:\(|,|;|\{|$)")
_TEST_ANNOTATION = re.compile(r"@(?:org\.junit\S*\.)?(?:Test|ParameterizedTest|RepeatedTest|TestFactory|"
                              r"TestTemplate|Theory)\b")

def _words(text: str) -> list[str]:
    return text.split()


# Never keys: Java's reserved words and literals.
JAVA_KEYWORDS = frozenset(_words("""
abstract assert boolean break byte case catch char class const continue default do double else enum
extends final finally float for goto if implements import instanceof int interface long native new
package private protected public return short static strictfp super switch synchronized this throw
throws transient try void volatile while true false null var record yield sealed permits
"""))
PRIMITIVES = frozenset(_words("boolean byte char double float int long short void var"))
# Never keys: java.lang names every class uses without an import (an edit mentioning them is not
# thereby related to a function that also does) and the standard annotations.
JAVA_LANG = frozenset(_words("""
String Object Integer Long Boolean Double Float Short Byte Character Number Void Math System Class
Exception RuntimeException Error Throwable IllegalArgumentException IllegalStateException
NullPointerException UnsupportedOperationException Override Deprecated SuppressWarnings
FunctionalInterface SafeVarargs Iterable Comparable Runnable StringBuilder Thread
"""))
# Not keys of a code fragment (still keys of the import that brings them in): the java.util types
# and helpers nearly every class uses.
COMMON_TYPES = frozenset(_words("""
List Map Set Collection Collections ArrayList HashMap HashSet LinkedHashMap LinkedHashSet TreeMap
TreeSet LinkedList Arrays Objects Optional Iterator Entry Stream Collectors Function Supplier
Consumer Predicate
"""))
# Not keys of a signature: java.lang.Object's methods, which every class has.
OBJECT_METHODS = frozenset(_words("toString equals hashCode clone finalize getClass"))
# Names of one or two characters (`e`, `i`, `id`, `it`) are locals far more often than API names.
MIN_NAME = 3


def _strip_code(text: str) -> str:
    return _LITERAL.sub(" ", _COMMENT.sub(" ", text or ""))


def identifiers(text: str) -> set[str]:
    """The identifiers of Java (or Scala/Kotlin) code, comments and string literals removed."""
    return set(_IDENT.findall(_strip_code(text)))


def _usable(names, extra=frozenset()) -> set[str]:
    return {n for n in names if len(n) >= MIN_NAME and n not in JAVA_KEYWORDS
            and n not in JAVA_LANG and n not in extra}


class FunctionNames(BaseModel):
    """What the filter matches against, for one function unit."""

    used: set[str]          # identifiers of source.before, source.after and target
    changed: set[str]       # identifiers in exactly one of source.before / source.after
    own: set[str]           # the function's own name(s), before and after


def function_names(before: str | None, after: str | None, target: str | None) -> FunctionNames:
    b, a, t = identifiers(before or ""), identifiers(after or ""), identifiers(target or "")
    own = {n for n in (method_name(declaration_of(x)) for x in (before, after) if x and x.strip()) if n}
    return FunctionNames(used=_usable(b | a | t), changed=_usable(a ^ b), own=own)


def _is_comment(code: str) -> bool:
    return code.startswith(("//", "/*", "*", "*/"))


class EvidenceStatement(BaseModel):
    """One statement of an other-edits entry: consecutive lines on one side of a run."""

    run: int
    sign: str              # "-" or "+"
    lines: list[str]
    kind: str              # import | declaration | fragment
    keys: list[str]        # sorted
    code: str              # comments removed, whitespace removed (for the moved/reformatted test)


def _angle_depth_cut(code: str) -> int:
    """Index of the first `=`, `;`, `(` or `,` outside `<...>` (len(code) when none)."""
    depth = 0
    for i, ch in enumerate(code):
        if ch == "<":
            depth += 1
        elif ch == ">" and depth:
            depth -= 1
        elif depth == 0 and ch in "=;(,":
            return i
    return len(code)


def _declared(code: str) -> set[str] | None:
    """The names a statement declares (None: not a declaration)."""
    s = _MODIFIERS.sub("", _ANNOTATION.sub(" ", code).strip()).strip()
    m = _TYPE_DECL.match(s)
    if m:   # a class / interface / enum header: everything it names (extends, implements)
        return _usable(_IDENT.findall(s), COMMON_TYPES) | {m.group(1)}
    m = _ENUM_CONSTANT.match(s)
    if m and not s.startswith(("return", "throw")):
        return {m.group(1)}
    if s.startswith("<"):        # a generic method: <T> T name(...)
        close = s.find(">")
        s = s[close + 1:].strip() if close > 0 else s
    head = s[:_angle_depth_cut(s)].rstrip()
    m = re.search(r"([A-Za-z_$][\w$]*)\s*(?:\[\s*\]\s*)*$", head)
    if not m:
        return None
    prefix = head[:m.start()].rstrip()
    if not prefix or not re.search(r"[\w$>\]]$", prefix) or "." in prefix.split()[-1][-1:]:
        return None
    first = _IDENT.match(prefix)
    if first is None or (first.group(0) in JAVA_KEYWORDS and first.group(0) not in PRIMITIVES):
        return None   # return x; / throw new X(...); / new Foo(...)
    if re.search(r"[=(){};]", prefix):
        return None
    return {m.group(1)}


def _import_keys(code: str) -> set[str] | None:
    m = re.match(r"import\s+(?:static\s+)?([\w$.]+?)(\.\*)?\s*;", code)
    if not m:
        return None
    return {m.group(1).rsplit(".", 1)[-1]}


def statements(lines: list[str]) -> list[EvidenceStatement]:
    """Split an other-edits entry's "- "/"+ " lines into statements: runs of changed lines (removed
    lines then added lines), each side cut after a code line that closes its brackets and ends with
    `;`, `{`, `}` or `,` (an enum constant); `{` opens a body, so the statement runs to its closing
    brace."""
    out: list[EvidenceStatement] = []
    run, prev_sign = 0, None
    cur: list[str] = []
    depth = 0

    def flush(sign):
        nonlocal cur, depth
        if cur:
            code = " ".join(_strip_code(ln[2:]) for ln in cur if not _is_comment(ln[2:].strip())).strip()
            keys = _import_keys(code) if code.startswith("import ") else None
            if keys is not None:
                kind = "import"
            else:
                keys = _declared(code) if code else None
                kind = "declaration" if keys is not None else "fragment"
                if keys is None:
                    keys = _usable(_IDENT.findall(_ANNOTATION.sub(" ", code)), COMMON_TYPES)
            out.append(EvidenceStatement(run=run, sign=sign, lines=cur, kind=kind,
                                         keys=sorted(keys), code=re.sub(r"\s+", "", code)))
        cur, depth = [], 0

    for line in lines:
        sign, text = line[:1], line[2:].strip()
        if sign != prev_sign:
            flush(prev_sign)
            if prev_sign == "+":
                run += 1          # removed lines after added ones: the next run
            prev_sign = sign
        cur.append(line)
        if _is_comment(text):
            continue
        code = _strip_code(text)
        depth += code.count("(") + code.count("{") + code.count("[") \
            - code.count(")") - code.count("}") - code.count("]")
        if depth < 0 or (depth == 0 and code.rstrip().endswith((";", "{", "}", ","))):
            flush(sign)
    flush(prev_sign)
    return out


def _statement_hit(s: EvidenceStatement, names: FunctionNames) -> bool:
    keys = set(s.keys)
    if s.kind == "import":
        return bool(keys & names.used) or any(k[:1].lower() + k[1:] in names.used for k in keys)
    if s.kind == "declaration":
        return bool(keys & names.used)
    return bool(keys & names.changed)


def filter_statements(lines: list[str], names: FunctionNames) -> tuple[list[EvidenceStatement], list[EvidenceStatement]]:
    """(kept, dropped) statements of one other-edits entry, each in their original order."""
    sts = statements(lines)
    moved = set()
    for i, s in enumerate(sts):
        for j, t in enumerate(sts):
            if i != j and s.run == t.run and s.sign != t.sign and s.code and s.code == t.code:
                moved.add(i)
    hit = [i not in moved and _statement_hit(s, names) for i, s in enumerate(sts)]
    # partners: statements of the other side of the same run that share a key, or that stand at
    # the same position when both sides have as many statements (a 1:1 replacement, e.g. a rename)
    pos: dict[tuple[int, str], list[int]] = {}
    for i, s in enumerate(sts):
        pos.setdefault((s.run, s.sign), []).append(i)
    keep = list(hit)
    for i, s in enumerate(sts):
        if keep[i] or i in moved:
            continue
        mine, other = pos[(s.run, s.sign)], pos.get((s.run, "+" if s.sign == "-" else "-"), [])
        aligned = other[mine.index(i)] if len(mine) == len(other) else None
        keep[i] = any(hit[j] and (j == aligned or set(s.keys) & set(sts[j].keys)) for j in other)
    return [s for s, k in zip(sts, keep) if k], [s for s, k in zip(sts, keep) if not k]


def filter_other_edits(edits: list[OtherEdit], names: FunctionNames) -> tuple[list[OtherEdit], int, int]:
    """(the entries reduced to their kept statements, statements kept, statements dropped).
    An entry with no kept statement is left out."""
    out: list[OtherEdit] = []
    n_kept = n_dropped = 0
    for e in edits:
        kept, dropped = filter_statements(e.lines, names)
        n_kept += len(kept)
        n_dropped += len(dropped)
        if kept:
            out.append(OtherEdit(entity=e.entity, kind=e.kind, lines=[ln for s in kept for ln in s.lines]))
    return out, n_kept, n_dropped


def method_name(signature: str | None) -> str | None:
    """The method or constructor name of a declaration header (annotations skipped)."""
    if not signature:
        return None
    s = _ANNOTATION.sub(" ", _strip_code(signature))
    m = re.search(r"([A-Za-z_$][\w$]*)\s*(?:\[[^\]]*\]\s*)?\(", s)
    return m.group(1) if m else None


def _without_annotations(signature: str | None) -> str | None:
    return None if signature is None else re.sub(r"\s+", " ", _ANNOTATION.sub(" ", signature)).strip()


def signature_relevant(before: str | None, after: str | None, names: FunctionNames,
                       same_file: bool = False) -> bool:
    """Whether a function signature entry (before / after the change) is relevant to the function."""
    if any(s and _TEST_ANNOTATION.search(s) for s in (before, after)):
        return False                                     # a test method: nothing calls it
    keys = {n for n in (method_name(before), method_name(after)) if n and n not in OBJECT_METHODS}
    unchanged = before is not None and after is not None \
        and _without_annotations(before) == _without_annotations(after)
    if unchanged and not same_file:
        return False                                     # only annotations change: callers unaffected
    if unchanged:                                        # body-only sibling: kept when the function calls it
        return bool(keys & (names.used - names.own))
    return bool(keys & names.used)
