"""Grammar lookup for RCAP: SALP's grammars (Java, Scala) plus C.

SALP mints SAPs for Java and Scala only, and RCAP took its grammars from it.
The Vim -> Neovim benchmark of PPatHF (Pan et al., ISSTA'24) is C, single
functions, and supplies the function pairs itself, so no SAP exists for it; to
run RCAP's request, acceptance, re-prompt and scoring stages on it they need a
C grammar. It is declared HERE, in SALP's own `Grammar` vocabulary, so SALP is
not touched and every module keeps one code path: `grammar_for` answers with
SALP's grammar when it has one and with `C` for `c` / `h`.

What differs in C, and what the callers do about it (`is_classless`):

* there is no class to wrap a lone function in: a function is parsed bare, and
  it is `function_definition` at the top of the translation unit;
* its name is not a `name` field but sits at the bottom of the declarator
  chain (`function_name_node`);
* real C does not always parse without an ERROR node: a function-like macro
  in a signature (`FUNC_ATTR_NONNULL_ALL`), a type passed to a macro
  (`ALLOC_ONE(char_u *)`), a condition split over `#ifdef` branches are valid
  to the compiler and an error to tree-sitter. 27 of the 310 Neovim functions
  and 136 of the 310 Vim functions of the benchmark have one, and so do 35 of
  the 300 stored zero-shot answers, 10 of them token-identical to the
  developer's port. A parse error is therefore no evidence of a broken answer
  in C (`tolerates_parse_errors`): `rcap.generate` keeps such an answer and
  records the fact; whether it is valid code is the compile check's question.

Removable statement types of the program cut are per language too
(`removable_types`): Java's set is the recorded one, unchanged.
"""

from __future__ import annotations

from typing import Any

from salp.structural import grammar_for as _salp_grammar_for
from salp.structural import (
    parse,  # noqa: F401  (re-exported: one import site for callers)
)
from salp.structural.grammars import ControlFlow, Grammar

# Fields newer SALP versions added to Grammar; passed only when this SALP has them,
# so RCAP runs against both the pinned and the newer SALP.
_NEWER = {"comment": frozenset({"comment"})}
_HAS = set(getattr(Grammar, "__dataclass_fields__", {}))

C = Grammar(
    **{k: v for k, v in _NEWER.items() if k in _HAS},
    name="C",
    module="tree_sitter_c",
    extensions=frozenset({"c", "h"}),
    root="translation_unit",
    class_like=frozenset(),
    method_like=frozenset({"function_definition"}),
    parameters=frozenset({"parameter_list"}),
    parameter=frozenset({"parameter_declaration", "variadic_parameter"}),
    package="",
    package_name=(),
    import_declaration="preproc_include",
    import_prefixes=("#include",),
    if_node="if_statement",
    else_token="else",
    block="compound_statement",
    class_body="",
    field_like=frozenset({"field_declaration"}),
    invocation=frozenset({"call_expression"}),
    reference_like=frozenset({"identifier", "type_identifier"}),
    control_flow={
        "for_statement": ControlFlow.FOR,
        "while_statement": ControlFlow.WHILE,
        "do_statement": ControlFlow.DO,
        "switch_statement": ControlFlow.SWITCH,
        "case_statement": ControlFlow.SWITCH_CASE,
        "return_statement": ControlFlow.RETURN,
        "break_statement": ControlFlow.BREAK,
        "continue_statement": ControlFlow.CONTINUE,
    },
)

_LOCAL = {ext: C for ext in C.extensions}

# Removable subtree types of the program cut for C (rcap.reduction_program).
C_REMOVABLE_TYPES = frozenset({
    "if_statement", "for_statement", "while_statement", "do_statement", "switch_statement",
})


def _key(ext: str | None) -> str:
    return (ext or "").lstrip(".").lower()


def grammar_for(ext: str | None) -> Grammar | None:
    """SALP's grammar for the extension, else RCAP's own (C); None when neither
    exists or the tree-sitter binding is not installed."""
    grammar = _salp_grammar_for(ext)
    if grammar is not None:
        return grammar
    local = _LOCAL.get(_key(ext))
    return local if local is not None and local.available() else None


def is_classless(grammar: Grammar | None) -> bool:
    """A language whose functions stand at the top of the file (C)."""
    return grammar is not None and not grammar.class_like


def tolerates_parse_errors(language: str | None) -> bool:
    """Valid code of this language may hold tree-sitter ERROR nodes (C: macros
    and preprocessor branches)."""
    return _key(language) in _LOCAL


def removable_types(language: str | None, default: frozenset[str]) -> frozenset[str]:
    return C_REMOVABLE_TYPES if _key(language) in _LOCAL else default


def function_name_node(node: Any) -> Any | None:
    """The identifier that names a method-like node: its `name` field (Java,
    Scala) or the identifier at the bottom of its declarator chain (C)."""
    name = node.child_by_field_name("name")
    if name is not None:
        return name
    cur = node.child_by_field_name("declarator")
    while cur is not None:
        if cur.type == "identifier":
            return cur
        nxt = cur.child_by_field_name("declarator")
        if nxt is None:
            nxt = next((c for c in cur.children if c.type in (
                "function_declarator", "pointer_declarator", "parenthesized_declarator",
                "identifier")), None)
        cur = nxt
    return None
