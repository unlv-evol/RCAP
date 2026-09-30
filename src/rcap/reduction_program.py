"""Program-Context Reduction: the AST-guided cut (RCAP ref section 8, after PPatHF).

Runs only after Semantic Evidence Reduction has fixed what is required, and is
constrained by it (I9): the protected set is the changed source regions, the
target edit region, and every line a retained semantic relationship needs
(renamed entities, replaced-API call sites). Removable candidates are
compound-statement subtrees (and, by the same walk, nested blocks) that do not
touch a protected line; each is replaced by an indexed placeholder comment with
a recovery map so the artifact is reconstructable. A placeholder may never
overlap a protected region — that is a context-construction failure, not a
smaller cut. Parsing reuses SALP's structural layer (tree-sitter, per-language
Grammar), so the stage inherits Java and Scala support.
"""

from __future__ import annotations

import difflib
import hashlib
import re

from pydantic import BaseModel, Field
from salp.structural import grammar_for, parse

from rcap.config import ExecutionConfig
from rcap.materialize import MaterializationRecord
from rcap.model import CaseModel
from rcap.reduction_semantic import SemanticReductionManifest

# Removable subtree types for Java (a recorded configuration choice, not code).
REMOVABLE_TYPES = frozenset({
    "if_statement", "for_statement", "enhanced_for_statement", "while_statement",
    "do_statement", "try_statement", "try_with_resources_statement",
    "switch_expression", "synchronized_statement",
})

PLACEHOLDER = "/* RCAP_PH_{n} */"
_WS = re.compile(r"\s+")


class ContextConstructionFailure(Exception):
    def __init__(self, case_id: str, diagnostics: list[str]):
        self.stage = "program_reduction"
        self.diagnostics = diagnostics
        super().__init__(f"context-construction failure for {case_id}: {'; '.join(diagnostics)}")


class PlaceholderEntry(BaseModel):
    ph_id: str
    start_byte: int
    end_byte: int
    original_text: str
    sha256: str


class ReducedArtifact(BaseModel):
    entity: str
    role: str
    reduced_text: str
    placeholders: list[PlaceholderEntry] = Field(default_factory=list)
    protected_lines: list[int] = Field(default_factory=list)  # 0-based
    nodes_before: int
    nodes_after: int
    reduction_reason: str | None = None  # why reduction was a no-op, when it was


class ProgramReductionRecord(BaseModel):
    case_id: str
    artifacts: list[ReducedArtifact] = Field(default_factory=list)

    def artifact(self, entity: str, role: str) -> ReducedArtifact:
        return next(a for a in self.artifacts if a.entity == entity and a.role == role)


def recover(artifact: ReducedArtifact) -> str:
    """Reconstruct the original text from the reduced text and the recovery map."""
    text = artifact.reduced_text
    for entry in artifact.placeholders:
        text = text.replace(PLACEHOLDER.format(n=entry.ph_id), entry.original_text, 1)
    return text


def _changed_lines(a: str, b: str) -> tuple[set[int], set[int]]:
    """0-based changed line sets for (a, b), from the diff between them (PPatHF section 4.1)."""
    sm = difflib.SequenceMatcher(a=a.splitlines(), b=b.splitlines(), autojunk=False)
    changed_a: set[int] = set()
    changed_b: set[int] = set()
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "delete"):
            changed_a.update(range(i1, i2))
        if tag in ("replace", "insert"):
            changed_b.update(range(j1, j2))
    return changed_a, changed_b


def _evidence_identifiers(case: CaseModel, reduction: SemanticReductionManifest) -> set[str]:
    """Identifiers that retained semantic evidence names; their lines are protected (I9)."""
    names: set[str] = set()
    for oid in reduction.retained_ids():
        rec = case.evidence.get(oid)
        if rec is None:
            continue
        for key in ("from", "to", "entity", "api"):
            value = rec.attributes.get(key)
            if isinstance(value, str) and value.isidentifier():
                names.add(value)
    return names


def _lines_naming(text: str, names: set[str]) -> set[int]:
    out: set[int] = set()
    if not names:
        return out
    pattern = re.compile(r"\b(" + "|".join(re.escape(n) for n in sorted(names)) + r")\b")
    for i, line in enumerate(text.splitlines()):
        if pattern.search(line):
            out.add(i)
    return out


def _count_nodes(node) -> int:
    return 1 + sum(_count_nodes(c) for c in node.children)


def _normalize(s: str) -> str:
    return _WS.sub("", s)


def _candidates(root, source: str, protected: set[int], min_lines: int):
    """Outermost removable subtrees not touching a protected line, by recursive walk."""
    found = []

    def walk(node):
        start, end = node.start_point[0], node.end_point[0]
        if (node.type in REMOVABLE_TYPES
                and end - start + 1 >= min_lines
                and not any(start <= p <= end for p in protected)):
            found.append(node)
            return  # outermost wins; never descend into a placeheld subtree
        for child in node.children:
            walk(child)

    walk(root)
    return found


def _reduce_one(
    entity: str, role: str, text: str, protected: set[int],
    grammar, min_lines: int, removable_norm: set[str] | None,
) -> ReducedArtifact:
    tree = parse(text, grammar)
    root = tree.root_node
    nodes_before = _count_nodes(root)

    nodes = _candidates(root, text, protected, min_lines)
    if removable_norm is not None:  # target side: only segments the source also removes
        nodes = [n for n in nodes
                 if _normalize(text[n.start_byte:n.end_byte]) in removable_norm]

    # tree-sitter offsets are BYTE offsets: splice in bytes, decode once at the
    # end. Splicing the str with byte offsets silently corrupts any payload
    # containing non-ASCII characters before a placeheld node.
    raw = text.encode()
    reduced_bytes = raw
    placeholders: list[PlaceholderEntry] = []
    for i, node in enumerate(sorted(nodes, key=lambda n: n.start_byte, reverse=True), 1):
        original = raw[node.start_byte:node.end_byte].decode()
        ph_id = f"{role}.{len(nodes) - i + 1}"
        placeholders.insert(0, PlaceholderEntry(
            ph_id=ph_id, start_byte=node.start_byte, end_byte=node.end_byte,
            original_text=original,
            sha256=hashlib.sha256(original.encode()).hexdigest(),
        ))
        reduced_bytes = reduced_bytes[:node.start_byte] \
            + PLACEHOLDER.format(n=ph_id).encode() + reduced_bytes[node.end_byte:]
    reduced = reduced_bytes.decode()

    nodes_after = _count_nodes(parse(reduced, grammar).root_node)
    return ReducedArtifact(
        entity=entity, role=role, reduced_text=reduced, placeholders=placeholders,
        protected_lines=sorted(protected), nodes_before=nodes_before,
        nodes_after=nodes_after,
    )


def reduce_program(
    case: CaseModel,
    reduction: SemanticReductionManifest,
    materialized: MaterializationRecord,
    config: ExecutionConfig | None = None,
    *,
    enabled: bool = True,
    evidence_protection: bool = True,
) -> ProgramReductionRecord:
    """One code path for all four harness configs (section 15): `enabled=False`
    passes artifacts through unreduced; `evidence_protection=False` (the
    Program-Reduction-Only config) protects only the changed regions, mirroring
    PPatHF, so the semantic contribution to the protected set is isolable."""
    config = config or ExecutionConfig()
    grammar = grammar_for("java")
    if grammar is None:
        raise ContextConstructionFailure(case.case_id, ["tree-sitter-java is not installed"])

    record = ProgramReductionRecord(case_id=case.case_id)
    names = _evidence_identifiers(case, reduction) if evidence_protection else set()

    for entity in reduction.materialize:
        roles = materialized.by_role(entity)
        f_s, f_sp, f_t = (roles[r].content for r in ("source.before", "source.after", "target"))
        ch_s, ch_sp = _changed_lines(f_s, f_sp)

        if not enabled:
            for role, text, protected in (("source.before", f_s, ch_s),
                                          ("source.after", f_sp, ch_sp),
                                          ("target", f_t, set())):
                n = _count_nodes(parse(text, grammar).root_node)
                record.artifacts.append(ReducedArtifact(
                    entity=entity, role=role, reduced_text=text,
                    protected_lines=sorted(protected),
                    nodes_before=n, nodes_after=n,
                    reduction_reason="program_reduction_disabled"))
            continue

        art_s = _reduce_one(entity, "source.before", f_s,
                            ch_s | _lines_naming(f_s, names), grammar,
                            config.min_placeholder_lines, None)
        art_sp = _reduce_one(entity, "source.after", f_sp,
                             ch_sp | _lines_naming(f_sp, names), grammar,
                             config.min_placeholder_lines, None)
        removable_norm = {_normalize(p.original_text)
                          for p in art_s.placeholders + art_sp.placeholders}
        art_t = _reduce_one(entity, "target", f_t,
                            _lines_naming(f_t, names), grammar,
                            config.min_placeholder_lines, removable_norm or None)

        for art in (art_s, art_sp, art_t):
            _assert_no_protected_overlap(case, art)
            record.artifacts.append(art)

    return record


def _assert_no_protected_overlap(case: CaseModel, art: ReducedArtifact) -> None:
    """No placeholder may overlap a hunk-touched/protected line — hard failure, not a cut."""
    original = recover(art)
    # Placeholder spans are byte offsets; measure lines in bytes to match.
    offsets, pos = [], 0
    for line in original.splitlines(keepends=True):
        n = len(line.encode())
        offsets.append((pos, pos + n))
        pos += n
    problems = []
    for entry in art.placeholders:
        for p in art.protected_lines:
            lo, hi = offsets[p]
            if entry.start_byte < hi and lo < entry.end_byte:
                problems.append(f"{art.role}: placeholder {entry.ph_id} overlaps protected line {p}")
    if problems:
        raise ContextConstructionFailure(case.case_id, problems)
