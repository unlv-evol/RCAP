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

Target-side hunk protection (TARGET_HUNK_PROTECTION_RULE): the target lines an
upstream hunk maps to are protected exactly like the changed source lines, so
the code the model has to edit is never hidden behind a placeholder. SALP
localizes a hunk only to the target FUNCTION, so the lines are found by
aligning source.before with the target (`target_hunk_lines`). Before this rule
the target was protected only by evidence identifiers, and a source side that
cut nothing (its only compound statement holds the change) left the target
filter empty, which meant "cut every removable subtree": the whole body of
e.g. KafkaLog4jAppender.close() (PR-15922) became one placeholder. The target
is first cut as before; only when a placeholder covers a hunk-mapped line is
it cut again with those lines protected, so a target without the problem keeps
exactly its previous cut (and its prompt stays byte-identical).
"""

from __future__ import annotations

import difflib
import hashlib
import re

from pydantic import BaseModel, Field
from rcap.grammars import grammar_for, parse, removable_types, tolerates_parse_errors

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
# Recorded in every target artifact and in run provenance; bump when the rule changes.
TARGET_HUNK_PROTECTION_RULE = "target-hunk-protection-v1"
# A changed line that the alignment cannot place is also looked up by its text,
# but only when it is distinctive enough (normalized length) to mean the same code.
_TEXT_MATCH_MIN_CHARS = 8
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
    # target only: the subset of protected_lines that an upstream hunk maps to,
    # and the rule that computed it (None: the rule was off)
    hunk_protected_lines: list[int] = Field(default_factory=list)
    protection_rule: str | None = None


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


def target_hunk_lines(f_s: str, f_sp: str, f_t: str) -> tuple[set[int], str]:
    """0-based lines of the target `f_t` that the upstream change f_s -> f_sp maps to.

    1. Hunk lines of source.before: every line the diff replaces or deletes, and
       both neighbours of a pure insertion (where the new lines go).
    2. Each is carried to the target by a line alignment of source.before with
       the target (whitespace-insensitive): an equal line maps to its partner, a
       line in a replaced block maps to the whole replacing block of the target,
       a line the target lacks maps to the two target lines around the gap.
    3. A hunk line the alignment could not pair with an identical target line is
       also matched by its text anywhere in the target, when it is distinctive.
    When source.before has hunk lines but none of them reaches the target, the
    change cannot be placed at all and EVERY target line is returned: keep the
    code visible rather than guess. Returns (lines, how) for the record."""
    s, sp, t = f_s.splitlines(), f_sp.splitlines(), f_t.splitlines()
    if not t:
        return set(), "empty_target"
    hunk_s: set[int] = set()
    sm = difflib.SequenceMatcher(a=s, b=sp, autojunk=False)
    for tag, i1, i2, _j1, _j2 in sm.get_opcodes():
        if tag in ("replace", "delete"):
            hunk_s.update(range(i1, i2))
        elif tag == "insert":
            hunk_s.update(i for i in (i1 - 1, i1) if 0 <= i < len(s))
    if not hunk_s:
        return set(), "no_source_change"

    ns, nt = [_normalize(x) for x in s], [_normalize(x) for x in t]
    out: set[int] = set()
    unpaired: set[int] = set()
    al = difflib.SequenceMatcher(a=ns, b=nt, autojunk=False)
    for tag, i1, i2, j1, j2 in al.get_opcodes():
        hit = [i for i in range(i1, i2) if i in hunk_s]
        if not hit:
            continue
        if tag == "equal":
            out.update(j1 + (i - i1) for i in hit)
            continue
        unpaired.update(hit)
        if tag == "replace":
            out.update(range(j1, j2))
        elif tag == "delete":
            out.update(j for j in (j1 - 1, j1) if 0 <= j < len(t))
    for i in sorted(unpaired):
        if len(ns[i]) >= _TEXT_MATCH_MIN_CHARS:
            out.update(j for j, line in enumerate(nt) if line == ns[i])
    if not out:
        return set(range(len(t))), "unmapped_all_protected"
    return out, "aligned"


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
    """Node count of a subtree, iteratively.

    Recursion here overflowed the stack on real inputs: Kafka's KafkaConfig
    nests deeply enough that a recursive walk raises RecursionError, which
    crashed the case instead of producing a measurement (4 SAPs in the 477 run).
    An explicit stack has no depth limit.
    """
    total, stack = 0, [node]
    while stack:
        n = stack.pop()
        total += 1
        stack.extend(n.children)
    return total


def _normalize(s: str) -> str:
    return _WS.sub("", s)


def _candidates(root, source: str, protected: set[int], min_lines: int,
                types: frozenset[str] = REMOVABLE_TYPES, clean_only: bool = False):
    """Outermost removable subtrees not touching a protected line, by recursive walk.
    `clean_only` (C): a subtree that holds a parse error is never cut, because
    its extent is then a guess of the parser's (a macro, an `#ifdef` branch)."""
    found = []

    def walk(node):
        start, end = node.start_point[0], node.end_point[0]
        if (node.type in types
                and end - start + 1 >= min_lines
                and not (clean_only and node.has_error)
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
    language: str | None = None,
) -> ReducedArtifact:
    tree = parse(text, grammar)
    root = tree.root_node
    nodes_before = _count_nodes(root)

    nodes = _candidates(root, text, protected, min_lines,
                        removable_types(language, REMOVABLE_TYPES),
                        tolerates_parse_errors(language))
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
    PPatHF, so the semantic contribution to the protected set is isolable.
    The changed regions include their target counterpart (`target_hunk_lines`)
    whenever `config.protect_target_hunks` is on (the default), in every config."""
    config = config or ExecutionConfig()
    grammar = grammar_for(case.language)
    if grammar is None:
        raise ContextConstructionFailure(
            case.case_id, [f"no tree-sitter grammar available for {case.language}"])

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
                            config.min_placeholder_lines, None, case.language)
        art_sp = _reduce_one(entity, "source.after", f_sp,
                             ch_sp | _lines_naming(f_sp, names), grammar,
                             config.min_placeholder_lines, None, case.language)
        removable_norm = {_normalize(p.original_text)
                          for p in art_s.placeholders + art_sp.placeholders}
        names_t = _lines_naming(f_t, names)
        art_t = _reduce_one(entity, "target", f_t, names_t, grammar,
                            config.min_placeholder_lines, removable_norm or None, case.language)
        if config.protect_target_hunks:
            # Only a target whose cut hides a hunk-mapped line is reduced again,
            # with those lines protected; every other target keeps exactly the
            # cut it had (protecting a line can make the walk descend into a
            # block and cut smaller pieces of it, which would change prompts
            # that never had the problem).
            hunk_t, how = target_hunk_lines(f_s, f_sp, f_t)
            if _covering(art_t, hunk_t):
                art_t = _reduce_one(entity, "target", f_t, names_t | hunk_t, grammar,
                                    config.min_placeholder_lines, removable_norm or None,
                                    case.language)
                how += ":reduced_again"
            # recorded as protected either way, so the overlap check below holds it
            art_t.protected_lines = sorted(names_t | hunk_t)
            art_t.hunk_protected_lines = sorted(hunk_t)
            art_t.protection_rule = f"{TARGET_HUNK_PROTECTION_RULE}:{how}"

        for art in (art_s, art_sp, art_t):
            _assert_no_protected_overlap(case, art)
            record.artifacts.append(art)

    return record


def _line_spans(text: str) -> list[tuple[int, int]]:
    """Byte span of each line (placeholder spans are byte offsets)."""
    offsets, pos = [], 0
    for line in text.splitlines(keepends=True):
        n = len(line.encode())
        offsets.append((pos, pos + n))
        pos += n
    return offsets


def _covering(art: ReducedArtifact, lines: set[int]) -> bool:
    """Does any placeholder of `art` cover one of `lines` (0-based)?"""
    if not lines or not art.placeholders:
        return False
    offsets = _line_spans(recover(art))
    return any(e.start_byte < offsets[p][1] and offsets[p][0] < e.end_byte
               for e in art.placeholders for p in lines if p < len(offsets))


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
