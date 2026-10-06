"""What kind of text a unit's payloads are, and narrowing a whole-file target.

RCAP's request and its answer contract were written for one shape only: a
function on the source side and the corresponding function in the fork. SALP
emits two other shapes, and both were sent with the function wording:

* a FILE-LEVEL unit: the change is outside any method (imports, fields,
  declarations), the source payloads are diff excerpts (`@@ ... @@` headers)
  and the target payload is the fork's whole file;
* a MISLOCALIZED unit: the source payloads are one function, but SALP found no
  counterpart in the fork and fell back to the whole file as the target. Asked
  to "return the complete adapted target function", the model returns the one
  function it was shown, and the splice replaces the WHOLE FILE with it (the
  19 "MODEL_SYNTAX" build failures of the 477 run).

Two deterministic remedies live here, both pure text + tree-sitter:

`text_kind` names the shape of a payload, so request synthesis can describe
the unit truthfully and generation can reject an answer of the wrong shape.

`narrow_targets` repairs the second case where it can be repaired without
guessing: SALP searches the fork for the function under its AFTER name, so a
function the change renames is never found although the fork has it under the
BEFORE name. When the fork's file declares exactly one method with the source
function's before-name (else its after-name; overloads are resolved by
parameter count), the target payload is narrowed to that method and the unit
is an ordinary function unit again. Nothing is narrowed on an ambiguous or
absent match: the unit then stays a whole-file unit and is described as one.
"""

from __future__ import annotations

import difflib
import hashlib
import re

from salp.structural import grammar_for, parse

FUNCTION = "function"   # exactly one method / constructor / def
FILE = "file"           # a compilation unit: package clause, or no code at all
TYPE = "type"           # top-level type declaration(s) without a package clause
REGION = "region"       # diff excerpts of file-level code
OTHER = "other"         # anything else (several members, a fragment, unparsed)

# The shapes whose answer must be the complete text, not one function.
WHOLE_KINDS = frozenset({FILE, TYPE})

_HUNK_HEADER = re.compile(r"\s*@@ -\d+(?:,\d+)? \+\d+(?:,\d+)? @@")
_PROBE_HEAD = "class __RcapProbe {\n"
_IDENT = re.compile(r"[A-Za-z_$][\w$]*|\d+|[^\s\w]")
# before/after of a renamed function share most tokens; two functions SALP sliced apart do not
RENAME_SIMILARITY = 0.6


def _code_children(node) -> list:
    return [c for c in node.named_children if "comment" not in c.type]


def _lone_method(text: str, grammar):
    """The single method node of `text` parsed as a class body, else None."""
    tree = parse(_PROBE_HEAD + text + "\n}", grammar)
    if tree is None or tree.root_node.has_error:
        return None
    top = _code_children(tree.root_node)
    if len(top) != 1 or top[0].type not in grammar.class_like:
        return None
    body = next((c for c in top[0].named_children if c.type == grammar.class_body), None)
    members = _code_children(body) if body is not None else []
    if len(members) == 1 and members[0].type in grammar.method_like:
        return members[0]
    return None


def text_kind(text: str | None, language: str | None) -> str:
    """The shape of a payload (or of an answer) in `language`."""
    if not text or not text.strip():
        return OTHER
    if _HUNK_HEADER.match(text):
        return REGION
    grammar = grammar_for((language or "java").lstrip("."))
    if grammar is None:
        return OTHER
    tree = parse(text, grammar)
    if tree is None:
        return OTHER
    top = _code_children(tree.root_node)
    if any(c.type == grammar.package for c in top):
        return FILE
    if not top and not tree.root_node.has_error:
        # Comments only: a file whose code is commented out is still a file,
        # and never a function.
        return FILE
    if _lone_method(text, grammar) is not None:
        return FUNCTION
    if top and not tree.root_node.has_error \
            and any(c.type in grammar.class_like for c in top) \
            and all(c.type in grammar.class_like or c.type == grammar.import_declaration
                    for c in top):
        return TYPE
    return OTHER


def function_name(text: str, language: str | None) -> tuple[str, int] | None:
    """(name, parameter count) of a payload that is one function, else None."""
    grammar = grammar_for((language or "java").lstrip("."))
    if grammar is None:
        return None
    node = _lone_method(text, grammar)
    if node is None:
        return None
    name = node.child_by_field_name("name")
    if name is None:
        return None
    src = (_PROBE_HEAD + text + "\n}").encode()
    return src[name.start_byte:name.end_byte].decode(), _arity(node, grammar)


def _arity(method, grammar) -> int:
    n = 0
    stack = [c for c in method.children if c.type in grammar.parameters]
    for params in stack:
        n += sum(1 for c in params.named_children if c.type in grammar.parameter)
    return n


def find_methods(file_text: str, name: str, language: str | None) -> list[tuple[int, int, int]]:
    """(start byte, end byte, parameter count) of every method named `name`."""
    grammar = grammar_for((language or "java").lstrip("."))
    tree = parse(file_text, grammar) if grammar is not None else None
    if tree is None:
        return []
    raw = file_text.encode()
    out, stack = [], [tree.root_node]
    while stack:
        node = stack.pop()
        if node.type in grammar.method_like:
            ident = node.child_by_field_name("name")
            if ident is not None and raw[ident.start_byte:ident.end_byte].decode() == name:
                out.append((node.start_byte, node.end_byte, _arity(node, grammar)))
        stack.extend(node.children)
    return sorted(out)


def locate_counterpart(source_before: str, source_after: str, target_file: str,
                       language: str | None) -> dict[str, object]:
    """Where the source function's counterpart sits in a whole-file target.

    Returns {"found": bool, "reason": str, and when found: "text", "name",
    "matched_as" (before_name | after_name), "match" (unique | arity),
    "line"}. Deterministic; never picks among equally good candidates."""
    before, after = function_name(source_before, language), function_name(source_after, language)
    if before is None and after is None:
        return {"found": False, "reason": "the source payloads are not a single function"}
    if before is not None and after is not None and before[0] != after[0]:
        # Two names: a rename, or SALP sliced two different functions. Only a
        # rename has a counterpart to narrow to.
        sim = difflib.SequenceMatcher(a=_IDENT.findall(source_before),
                                      b=_IDENT.findall(source_after), autojunk=False).ratio()
        if sim < RENAME_SIMILARITY:
            return {"found": False, "reason": (
                f"the source payloads are two different functions (`{before[0]}` / `{after[0]}`, "
                f"token similarity {sim:.2f}), not one function before and after")}
    raw = target_file.encode()
    tried = []
    for label, fn in (("before_name", before), ("after_name", after)):
        if fn is None or any(t[0] == fn[0] for t in tried):
            continue
        name, arity = fn
        hits = find_methods(target_file, name, language)
        tried.append((name, len(hits)))
        how = "unique"
        if len(hits) > 1:
            hits, how = [h for h in hits if h[2] == arity], "arity"
        if len(hits) != 1:
            continue
        start, end, _ = hits[0]
        text = raw[start:end].decode() + "\n"
        if target_file.count(text) != 1:
            # the method's text could not be put back by an exact-text splice
            tried[-1] = (name, -1)
            continue
        return {"found": True, "text": text, "name": name, "matched_as": label, "match": how,
                "line": raw[:start].count(b"\n") + 1,
                "reason": f"the fork's file declares exactly one method `{name}` "
                          f"({label.replace('_', ' ')} of the source function"
                          + (", resolved by parameter count" if how == "arity" else "") + ")"}
    seen = ", ".join(f"`{n}`: {('text not unique' if c < 0 else str(c) + ' declaration(s)')}"
                     for n, c in tried)
    return {"found": False, "reason": f"no unique counterpart in the fork's file ({seen})"}


def narrow_targets(case, materialized) -> dict[str, dict[str, object]]:
    """Narrow whole-file target payloads of function units, in place.

    For every materialized entity whose source payloads are one function and
    whose target payload is a whole file (or type), `locate_counterpart`
    decides; on a hit the target payload's content becomes that method (hash
    recomputed, derivation recorded on the payload and in
    `materialized.narrowed`). Returns what was decided per entity, including
    the misses, so nothing is narrowed or left alone silently."""
    decided: dict[str, dict[str, object]] = {}
    entities = dict.fromkeys(p.entity for p in materialized.payloads)
    for entity in entities:
        roles = materialized.by_role(entity)
        if not all(r in roles for r in ("source.before", "source.after", "target")):
            continue
        target = roles["target"]
        if text_kind(target.content, case.language) not in WHOLE_KINDS:
            continue
        if text_kind(roles["source.after"].content, case.language) != FUNCTION \
                and text_kind(roles["source.before"].content, case.language) != FUNCTION:
            continue
        hit = locate_counterpart(roles["source.before"].content, roles["source.after"].content,
                                 target.content, case.language)
        info = {k: v for k, v in hit.items() if k != "text"}
        if hit["found"]:
            info.update({"from": "whole_file", "whole_file_sha256": target.sha256,
                         "whole_file_lines": target.content.count("\n") + 1})
            target.content = hit["text"]
            target.sha256 = hashlib.sha256(hit["text"].encode()).hexdigest()
            target.derivation = ("narrowed from the SAP's whole-file target payload: "
                                 + str(hit["reason"]))
            materialized.diagnostics.append(f"{entity}: target {target.derivation}")
        else:
            materialized.diagnostics.append(
                f"{entity}: source is one function but the target payload is the whole file; "
                f"not narrowed: {hit['reason']}")
        decided[entity] = info
    materialized.narrowed.update(decided)
    return decided
