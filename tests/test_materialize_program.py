"""Materialization (section 7) and Program-Context Reduction (section 8) tests."""

import json
import shutil

import pytest

from rcap.intake import load_case
from rcap.materialize import MaterializationFailure, materialize
from rcap.reduction_program import (
    recover,
    reduce_program,
)
from rcap.reduction_semantic import reduce_semantic
from rcap.selection import select


@pytest.fixture(scope="module")
def case(sap_dir, pr_manifest):
    return load_case(sap_dir, pr_manifest=pr_manifest)


@pytest.fixture(scope="module")
def pipeline(case):
    sel = select(case)
    red = reduce_semantic(case, sel)
    mat = materialize(case, red)
    return case, red, mat


# --- materialization (section 7) ---------------------------------------------

def test_materializes_exactly_the_retained_entities(pipeline):
    """I7: the materialized set equals the retained program-entity set."""
    _, red, mat = pipeline
    assert {p.entity for p in mat.payloads} == set(red.materialize)
    assert {p.role for p in mat.payloads} == {"source.before", "source.after", "target"}


def test_payloads_carry_hash_and_ref(pipeline):
    _, _, mat = pipeline
    for p in mat.payloads:
        assert p.sha256 and p.ref.startswith("functions/")
    assert any("fidelity flags" in d for d in mat.diagnostics)


def test_missing_payload_is_error_not_silent(case, sap_dir, tmp_path, pr_manifest):
    broken = tmp_path / "sap-NoTarget"
    shutil.copytree(sap_dir, broken)
    (broken / "functions" / "fn-1" / "target.java").rename(
        broken / "functions" / "fn-1" / "targe_.java")
    # bypass schema gate on purpose: materialization must catch it itself
    bcase = case.model_copy(deep=True)
    bcase.sap_dir = str(broken)
    bcase.functions["fn-1"].refs.pop("target.java")
    red = reduce_semantic(bcase, select(bcase))
    with pytest.raises(MaterializationFailure) as err:
        materialize(bcase, red)
    assert any("target" in d for d in err.value.diagnostics)


# --- program-context reduction (section 8) -----------------------------------

def test_untouched_for_loop_is_placeheld(pipeline):
    case, red, mat = pipeline
    rec = reduce_program(case, red, mat)
    art = rec.artifact("functions/fn-1", "source.before")
    assert len(art.placeholders) >= 1
    assert any("for (Node n" in p.original_text for p in art.placeholders)
    assert "RCAP_PH_" in art.reduced_text


def test_changed_region_never_placeheld(pipeline):
    """The edited if-statement stays; the change itself is always protected."""
    case, red, mat = pipeline
    rec = reduce_program(case, red, mat)
    for role, marker in (("source.before", "if (v > 2)"), ("source.after", "if (v >= 2)")):
        art = rec.artifact("functions/fn-1", role)
        assert marker in art.reduced_text
        assert all(marker not in p.original_text for p in art.placeholders)


def test_recovery_reconstructs_original_exactly(pipeline):
    case, red, mat = pipeline
    rec = reduce_program(case, red, mat)
    by_role = mat.by_role("functions/fn-1")
    for art in rec.artifacts:
        assert recover(art) == by_role[art.role].content


def test_target_reduction_mirrors_source_segments(pipeline):
    """A target segment is removable only when the source also removes it."""
    case, red, mat = pipeline
    rec = reduce_program(case, red, mat)
    art_t = rec.artifact("functions/fn-1", "target")
    # the fork's for-loop body drifted (registerNode), so it must NOT be placeheld
    assert "registerNode" in art_t.reduced_text
    # but the identical null-check IS removable on both sides
    assert any("c == null" in p.original_text for p in art_t.placeholders)


def test_size_measured_and_reduced(pipeline):
    case, red, mat = pipeline
    rec = reduce_program(case, red, mat)
    for art in rec.artifacts:
        assert art.nodes_before > 0
        assert art.nodes_after <= art.nodes_before


def test_program_reduction_deterministic(pipeline):
    case, red, mat = pipeline
    assert reduce_program(case, red, mat) == reduce_program(case, red, mat)


def test_evidence_identifier_lines_protected(enriched_pr_dir, enriched_sap_dir):
    """I9: a retained rename (setup -> setupInternal) protects the call-site line."""
    manifest = json.loads((enriched_pr_dir / "pr.json").read_text(encoding="utf-8"))
    ecase = load_case(enriched_sap_dir, pr_manifest=manifest)
    red = reduce_semantic(ecase, select(ecase))
    mat = materialize(ecase, red)
    rec = reduce_program(ecase, red, mat)
    for role in ("source.before", "source.after"):
        art = rec.artifact("functions/fn-1", role)
        assert "setup(v);" in art.reduced_text
        assert all("setup(v)" not in p.original_text for p in art.placeholders)


def test_count_nodes_survives_deep_nesting():
    """A deeply nested AST must not overflow the stack.

    Recursive counting raised RecursionError on Kafka's KafkaConfig, crashing
    the case rather than producing a measurement.
    """
    from salp.structural import grammar_for
    from salp.structural.syntax import parse

    from rcap.reduction_program import _count_nodes

    grammar = grammar_for("java")
    # nesting far past the default recursion limit
    expr = "1" + " + 1" * 4000
    src = "class D { int f() { return " + expr + "; } }"
    tree = parse(src, grammar)
    assert _count_nodes(tree.root_node) > 4000


# --- hunk regions in place of an oversized source context ---------------------

def _with_regions(case, sap_dir, tmp_path, *, before="int a;\n", after="int b;\n"):
    """A copy of the fixture SAP whose fn-1 hunks carry region files."""
    d = tmp_path / "sap-Regions"
    shutil.copytree(sap_dir, d)
    c = case.model_copy(deep=True)
    c.sap_dir = str(d)
    for t in c.transformations:
        if t.fn_id == "fn-1":
            h = d / "hunks" / t.hunk_id
            (h / "region.before.java").write_text(before)
            (h / "region.after.java").write_text(after)
            t.region_before_ref = f"hunks/{t.hunk_id}/region.before.java"
            t.region_after_ref = f"hunks/{t.hunk_id}/region.after.java"
    return c


def test_oversized_source_context_uses_hunk_regions(case, sap_dir, tmp_path, monkeypatch):
    import rcap.materialize as m
    c = _with_regions(case, sap_dir, tmp_path)
    monkeypatch.setattr(m, "REGION_THRESHOLD_CHARS", 10)
    mat = materialize(c, reduce_semantic(c, select(c)))
    roles = mat.by_role("functions/fn-1")
    n = sum(1 for t in c.transformations if t.fn_id == "fn-1")
    assert roles["source.before"].content == "int a;\n" * n
    assert roles["source.after"].content == "int b;\n" * n
    assert "hunk regions" in roles["source.before"].derivation
    assert roles["target"].derivation is None          # the target is never touched
    assert any("using the hunk regions" in x for x in mat.diagnostics)


def test_source_context_under_threshold_is_kept(case, sap_dir, tmp_path, monkeypatch):
    import rcap.materialize as m
    c = _with_regions(case, sap_dir, tmp_path)
    monkeypatch.setattr(m, "REGION_THRESHOLD_CHARS", 10**9)
    monkeypatch.setattr(m, "REGION_ON_FOREIGN_CHANGES", False)
    mat = materialize(c, reduce_semantic(c, select(c)))
    assert mat.by_role("functions/fn-1")["source.before"].derivation is None


def test_no_region_files_keeps_the_context(case, sap_dir, tmp_path, monkeypatch):
    """Older SALP output: nothing to swap in, so the whole context stays."""
    import rcap.materialize as m
    monkeypatch.setattr(m, "REGION_THRESHOLD_CHARS", 10)
    mat = materialize(case, reduce_semantic(case, select(case)))
    assert mat.by_role("functions/fn-1")["source.before"].derivation is None
    assert any("not every hunk has region files" in x for x in mat.diagnostics)


def test_foreign_changes_counts_only_other_hunks_edits():
    from rcap.materialize import foreign_changes
    before = "class A {\n  int snaphot;\n  void f() { use(snaphot); }\n}\n"
    after = "class A {\n  int snapshot;\n  void f() { use(snapshot); }\n}\n"
    field = [("  int snaphot;\n", "  int snapshot;\n")]
    assert foreign_changes(before, after, field) == 2       # f()'s line, both sides
    both = field + [("  void f() { use(snaphot); }\n", "  void f() { use(snapshot); }\n")]
    assert foreign_changes(before, after, both) == 0


def _class_context(c, sap_dir_copy):
    """Make fn-1's source context a class whose rename runs through a method too."""
    entry = c.functions["fn-1"]
    for role, word in (("source.before", "snaphot"), ("source.after", "snapshot")):
        ref = next(r for n, r in entry.refs.items() if n.startswith(role))
        (sap_dir_copy / ref).write_text(
            f"class A {{\n    int {word};\n\n    void f() {{\n        use({word});\n    }}\n}}\n")


def test_class_context_with_other_hunks_changes_uses_regions(case, sap_dir, tmp_path, monkeypatch):
    """A field hunk's class context also holds a method's edit: under the size
    threshold, it still switches to the field hunk's own region."""
    import rcap.materialize as m
    c = _with_regions(case, sap_dir, tmp_path, before="    int snaphot;\n", after="    int snapshot;\n")
    _class_context(c, tmp_path / "sap-Regions")
    monkeypatch.setattr(m, "REGION_THRESHOLD_CHARS", 10**9)
    mat = materialize(c, reduce_semantic(c, select(c)))
    roles = mat.by_role("functions/fn-1")
    assert "other hunks" in roles["source.before"].derivation
    assert "int snapshot;" in roles["source.after"].content and "use(" not in roles["source.after"].content
    assert any("changed line(s) of other hunks" in x for x in mat.diagnostics)


def test_method_context_is_never_switched_for_other_hunks(case, sap_dir, tmp_path, monkeypatch):
    """A method context under the threshold stays whole even when the regions
    do not cover all of its edits (it may have to be located in a class target)."""
    import rcap.materialize as m
    c = _with_regions(case, sap_dir, tmp_path)
    monkeypatch.setattr(m, "REGION_THRESHOLD_CHARS", 10**9)
    mat = materialize(c, reduce_semantic(c, select(c)))
    assert mat.by_role("functions/fn-1")["source.before"].derivation is None


def test_context_with_only_own_changes_is_kept(case, sap_dir, tmp_path, monkeypatch):
    """Regions that account for every change of the context: nothing to fix."""
    import rcap.materialize as m
    roles0 = materialize(case, reduce_semantic(case, select(case))).by_role("functions/fn-1")
    n = sum(1 for t in case.transformations if t.fn_id == "fn-1")
    # one hunk carries the whole change, the others none (same text both sides)
    c = _with_regions(case, sap_dir, tmp_path, before="", after="")
    first = next(t for t in c.transformations if t.fn_id == "fn-1")
    h = tmp_path / "sap-Regions" / "hunks" / first.hunk_id
    (h / "region.before.java").write_text(roles0["source.before"].content)
    (h / "region.after.java").write_text(roles0["source.after"].content)
    monkeypatch.setattr(m, "REGION_THRESHOLD_CHARS", 10**9)
    mat = materialize(c, reduce_semantic(c, select(c)))
    assert n >= 1
    assert mat.by_role("functions/fn-1")["source.before"].derivation is None
