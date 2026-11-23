"""Target-side hunk protection (rcap.reduction_program.target_hunk_lines).

Invariant: no target placeholder may cover a target line that an upstream hunk
maps to. Regression fixture: PR-15922, KafkaLog4jAppender.close(), where the
source cut nothing (its only `if` holds the change), the empty source filter
meant "cut everything" on the target, and the whole body of the fork's close()
became one placeholder, so the model echoed it.
"""

from __future__ import annotations

import json
import random

import pytest
from conftest import build_fixture_pr

from rcap.config import ExecutionConfig
from rcap.generate import StubBackend, generate
from rcap.intake import load_case
from rcap.materialize import materialize
from rcap.pipeline import run_case
from rcap.provenance import run_provenance
from rcap.reduction_program import (
    TARGET_HUNK_PROTECTION_RULE,
    _reduce_one,
    recover,
    reduce_program,
    target_hunk_lines,
)
from rcap.reduction_semantic import reduce_semantic
from rcap.selection import select

CLOSE_BEFORE = """@Override
    public void close() {
        if (!this.closed) {
            this.closed = true;
            producer.close();
        }
    }
"""
CLOSE_AFTER = """@Override
    public void close() {
        if (!this.closed) {
            this.closed = true;
            if (producer != null) {
                producer.close();
            }
        }
    }
"""
CLOSE_TARGET = CLOSE_BEFORE  # the fork's close() is upstream's before-version
CLOSE_DIFF = """@@ -370,7 +376,9 @@
     public void close() {
         if (!this.closed) {
             this.closed = true;
-            producer.close();
+            if (producer != null) {
+                producer.close();
+            }
         }
     }
"""
ADAPTED = "```java\n" + CLOSE_AFTER + "```"


def _build(tmp_path_factory, name, payloads):
    pr = build_fixture_pr(tmp_path_factory.mktemp(name), payloads=payloads)
    return pr / "sap-Validator", json.loads((pr / "pr.json").read_text(encoding="utf-8"))


def _stages(sap, manifest):
    case = load_case(sap, pr_manifest=manifest)
    red = reduce_semantic(case, select(case))
    mat = materialize(case, red)
    return case, red, mat


def _overlapping(art, lines):
    """ph_ids of placeholders covering any of `lines` (0-based lines of the original)."""
    offs, pos = [], 0
    for line in recover(art).splitlines(keepends=True):
        n = len(line.encode())
        offs.append((pos, pos + n))
        pos += n
    return [e.ph_id for e in art.placeholders
            if any(e.start_byte < offs[i][1] and offs[i][0] < e.end_byte for i in lines)]


@pytest.fixture(scope="module")
def close_sap(tmp_path_factory):
    return _build(tmp_path_factory, "close",
                  (CLOSE_BEFORE, CLOSE_AFTER, CLOSE_TARGET, CLOSE_DIFF))


# --- the PR-15922 regression -------------------------------------------------

def test_close_body_stays_visible(close_sap):
    rec = reduce_program(*_stages(*close_sap))
    t = rec.artifact("functions/fn-1", "target")
    assert t.placeholders == []
    assert "producer.close();" in t.reduced_text
    assert "if (!this.closed)" in t.reduced_text
    lines = CLOSE_TARGET.splitlines()
    assert [lines[i].strip() for i in t.hunk_protected_lines] == ["producer.close();"]
    assert set(t.hunk_protected_lines) <= set(t.protected_lines)
    assert t.protection_rule == f"{TARGET_HUNK_PROTECTION_RULE}:aligned:reduced_again"


def test_close_old_rule_reproduces_the_defect(close_sap):
    """The flag off is ede1b40's reduction: the whole body is one placeholder."""
    rec = reduce_program(*_stages(*close_sap),
                         ExecutionConfig(protect_target_hunks=False))
    t = rec.artifact("functions/fn-1", "target")
    assert len(t.placeholders) == 1
    assert "producer.close();" not in t.reduced_text
    assert "producer.close();" in t.placeholders[0].original_text
    assert t.protection_rule is None and t.hunk_protected_lines == []


def test_close_prompt_shows_the_line_to_edit(close_sap):
    sap, manifest = close_sap
    res = run_case(sap, manifest, StubBackend(ADAPTED))
    p = res.request.prompt
    section = p[p.index("## Target function in the fork"):]
    section = section[:section.index("```", section.index("```") + 3)]
    assert "producer.close();" in section
    assert "RCAP_PH_target" not in section
    assert res.generation.outcome == "completion"


def test_source_side_unchanged_by_the_rule(close_sap):
    """Only the target's protected set grows; the source artifacts are identical."""
    stages = _stages(*close_sap)
    on = reduce_program(*stages)
    off = reduce_program(*stages, ExecutionConfig(protect_target_hunks=False))
    for role in ("source.before", "source.after"):
        assert on.artifact("functions/fn-1", role) == off.artifact("functions/fn-1", role)


# --- the invariant -----------------------------------------------------------

def _invariant_holds(case, red, mat, config=None):
    rec = reduce_program(case, red, mat, config)
    for entity in red.materialize:
        by_role = mat.by_role(entity)
        t = rec.artifact(entity, "target")
        hunk, _ = target_hunk_lines(by_role["source.before"].content,
                                    by_role["source.after"].content,
                                    by_role["target"].content)
        assert _overlapping(t, hunk) == [], f"placeholder over a hunk-mapped line in {entity}"
        for art in rec.artifacts:
            assert recover(art) == mat.by_role(art.entity)[art.role].content
    return rec


def test_invariant_on_default_fixture(sap_dir, pr_manifest):
    _invariant_holds(*_stages(sap_dir, pr_manifest))


def _random_function(rng, n_blocks):
    body = []
    for b in range(n_blocks):
        kind = rng.choice(["if", "for", "while", "plain"])
        inner = [f"        call{b}_{k}(x{k});" for k in range(rng.randint(1, 3))]
        if kind == "plain":
            body += [f"    int v{b} = compute{b}();"]
        elif kind == "if":
            body += [f"    if (flag{b}) {{", *inner, "    }"]
        elif kind == "for":
            body += [f"    for (Item i{b} : items{b}) {{", *inner, "    }"]
        else:
            body += [f"    while (more{b}()) {{", *inner, "    }"]
    return ["void run(Item x) {", *body, "}"]


def _mutate(rng, lines):
    """One upstream edit: replace, delete or insert one body line."""
    out = list(lines)
    i = rng.randrange(1, len(out) - 1)
    op = rng.choice(["replace", "delete", "insert"])
    if op == "replace":
        out[i] = out[i].replace("(", "Changed(", 1) if "(" in out[i] else out[i] + " // changed"
    elif op == "delete" and len(out) > 3:
        del out[i]
    else:
        out.insert(i, "        added(x);")
    return out


def _drift(rng, lines):
    """The fork's copy: a renamed call here, a fork-only block there."""
    out = list(lines)
    if rng.random() < 0.5:
        j = rng.randrange(1, len(out) - 1)
        out[j] = out[j].replace("call", "forkCall", 1)
    if rng.random() < 0.5:
        j = rng.randrange(1, len(out) - 1)
        if out[j].rstrip().endswith(";") or out[j].strip() == "}":
            out[j + 1:j + 1] = ["    if (forkOnly) {", "        forkThing();", "        more();", "    }"]
    return out


@pytest.mark.parametrize("seed", range(24))
def test_invariant_on_random_functions(tmp_path_factory, seed):
    rng = random.Random(12535 + seed)
    before = _random_function(rng, rng.randint(2, 6))
    after = _mutate(rng, before)
    target = _drift(rng, before)
    payloads = ("\n".join(before) + "\n", "\n".join(after) + "\n",
                "\n".join(target) + "\n", "@@ -1 +1 @@\n")
    if payloads[0] == payloads[1]:
        pytest.skip("mutation was a no-op")
    _invariant_holds(*_stages(*_build(tmp_path_factory, f"rand{seed}", payloads)))


# --- untouched code is still reduced where safe ------------------------------

TWO_BLOCKS_BEFORE = """void flush() {
    for (Batch b : pending) {
        send(b);
        count++;
    }
    if (!this.closed) {
        this.closed = true;
        producer.close();
    }
}
"""
TWO_BLOCKS_AFTER = TWO_BLOCKS_BEFORE.replace(
    "        producer.close();\n",
    "        if (producer != null) {\n            producer.close();\n        }\n")


def test_mirrored_untouched_block_is_still_cut(tmp_path_factory):
    """The untouched for-loop is cut on both sides; the edited if stays visible."""
    sap = _build(tmp_path_factory, "two",
                 (TWO_BLOCKS_BEFORE, TWO_BLOCKS_AFTER, TWO_BLOCKS_BEFORE, "@@ -1 +1 @@\n"))
    rec = _invariant_holds(*_stages(*sap))
    t = rec.artifact("functions/fn-1", "target")
    assert len(t.placeholders) == 1
    assert "send(b);" in t.placeholders[0].original_text
    assert "producer.close();" in t.reduced_text


def test_fork_only_block_is_still_cut_when_source_cuts_nothing(tmp_path_factory):
    """Source cuts nothing (its only if holds the change). The fork's extra,
    untouched while-loop is still cut, as before; only the hunk's lines stay."""
    target = CLOSE_TARGET.replace(
        "    public void close() {\n",
        "    public void close() {\n        while (draining()) {\n"
        "            drainOne();\n            waitABit();\n        }\n")
    sap = _build(tmp_path_factory, "forkonly",
                 (CLOSE_BEFORE, CLOSE_AFTER, target, CLOSE_DIFF))
    rec = _invariant_holds(*_stages(*sap))
    t = rec.artifact("functions/fn-1", "target")
    assert len(t.placeholders) == 1
    assert "drainOne();" in t.placeholders[0].original_text
    assert "producer.close();" in t.reduced_text


def _target(stages, on):
    cfg = ExecutionConfig(protect_target_hunks=on)
    return reduce_program(*stages, cfg).artifact("functions/fn-1", "target")


def test_target_without_the_problem_is_unchanged(sap_dir, pr_manifest, tmp_path_factory):
    """A target whose old cut hid no hunk line keeps exactly that cut."""
    cases = [_stages(sap_dir, pr_manifest),
             _stages(*_build(tmp_path_factory, "two2", (TWO_BLOCKS_BEFORE, TWO_BLOCKS_AFTER,
                                                         TWO_BLOCKS_BEFORE, "@@ -1 +1 @@\n")))]
    for stages in cases:
        on, off = _target(stages, True), _target(stages, False)
        assert on.reduced_text == off.reduced_text and on.placeholders == off.placeholders
        assert on.protection_rule.endswith(":aligned")


NESTED_BEFORE = """void poll() {
    if (open) {
        for (Record r : buffer) {
            handle(r);
            seen++;
        }
        lastPoll = now();
    }
}
"""
NESTED_AFTER = NESTED_BEFORE.replace("lastPoll = now();", "lastPoll = clock.now();")
# the fork's outer if differs (so it is not a mirrored cut), the inner loop is identical
NESTED_TARGET = NESTED_BEFORE.replace("if (open) {", "if (open && !paused) {")


def test_outer_block_with_the_hunk_kept_inner_mirrored_block_cut(tmp_path_factory):
    """Old cut: nothing (the outer if is not mirrored, the walk never descends).
    No hunk line was hidden, so the cut is kept as it was: nothing cut."""
    stages = _stages(*_build(tmp_path_factory, "nested",
                             (NESTED_BEFORE, NESTED_AFTER, NESTED_TARGET, "@@ -1 +1 @@\n")))
    on, off = _target(stages, True), _target(stages, False)
    assert on == off.model_copy(update={"protected_lines": on.protected_lines,
                                        "hunk_protected_lines": on.hunk_protected_lines,
                                        "protection_rule": on.protection_rule})
    assert on.placeholders == []


@pytest.mark.parametrize("seed", range(24))
def test_random_targets_without_the_problem_are_unchanged(tmp_path_factory, seed):
    rng = random.Random(12535 + seed)
    before = _random_function(rng, rng.randint(2, 6))
    after = _mutate(rng, before)
    target = _drift(rng, before)
    if before == after:
        pytest.skip("mutation was a no-op")
    stages = _stages(*_build(tmp_path_factory, f"same{seed}",
                             ("\n".join(before) + "\n", "\n".join(after) + "\n",
                              "\n".join(target) + "\n", "@@ -1 +1 @@\n")))
    on, off = _target(stages, True), _target(stages, False)
    hunk, _ = target_hunk_lines(*(stages[2].by_role("functions/fn-1")[r].content
                                  for r in ("source.before", "source.after", "target")))
    if _overlapping(off, hunk):
        assert on.protection_rule.endswith(":reduced_again")
        assert _overlapping(on, hunk) == []
    else:
        assert on.reduced_text == off.reduced_text


def test_reduce_one_unprotected_still_cuts():
    """_reduce_one itself is unchanged: with no protection it cuts what it can."""
    from rcap.grammars import grammar_for
    art = _reduce_one("e", "target", CLOSE_TARGET, set(), grammar_for("java"), 3, None, "java")
    assert len(art.placeholders) == 1


# --- target_hunk_lines -------------------------------------------------------

def test_map_equal_line():
    hl, how = target_hunk_lines(CLOSE_BEFORE, CLOSE_AFTER, CLOSE_TARGET)
    assert how == "aligned" and hl == {4}


def test_map_ignores_indentation_drift():
    tgt = CLOSE_TARGET.replace("            producer.close();", "\tproducer.close();")
    hl, _ = target_hunk_lines(CLOSE_BEFORE, CLOSE_AFTER, tgt)
    assert hl == {4}


def test_map_pure_insertion_protects_both_neighbours():
    after = CLOSE_BEFORE.replace("            producer.close();\n",
                                 "            producer.close();\n            log.info(\"x\");\n")
    hl, _ = target_hunk_lines(CLOSE_BEFORE, after, CLOSE_TARGET)
    assert hl == {4, 5}


def test_map_line_missing_in_target_protects_the_gap():
    tgt = CLOSE_TARGET.replace("            producer.close();\n", "")
    hl, _ = target_hunk_lines(CLOSE_BEFORE, CLOSE_AFTER, tgt)
    lines = tgt.splitlines()
    assert {lines[i].strip() for i in hl} == {"this.closed = true;", "}"}


def test_map_diverged_line_protects_the_replacing_block_and_moved_copy():
    tgt = CLOSE_TARGET.replace("this.closed = true;", "closedFlag.set(true);")
    before = CLOSE_BEFORE
    after = before.replace("this.closed = true;", "this.closed = true; // now")
    hl, _ = target_hunk_lines(before, after, tgt)
    assert {tgt.splitlines()[i].strip() for i in hl} == {"closedFlag.set(true);"}


def test_map_fully_diverged_body_protects_the_whole_replacing_block():
    tgt = "void other() {\n    a();\n    b();\n}\n"
    src = "int f() {\n  return 1;\n}\n"
    hl, how = target_hunk_lines(src, src.replace("1", "2"), tgt)
    assert how == "aligned" and hl == {0, 1, 2}   # all but the shared closing brace


def test_map_unplaceable_change_protects_everything():
    """The changed line falls in a gap at a position the target cannot hold
    (deleted lines beyond a one-line target): nothing maps, keep all visible."""
    hl, how = target_hunk_lines("a();\nb();\n", "a();\n", "")
    assert how == "empty_target"
    # source lines absent from the target AND no neighbour: every line protected
    hl, how = target_hunk_lines("x();\n", "", "x2();\n")
    assert hl == {0}


def test_map_no_change_and_empty_target():
    assert target_hunk_lines(CLOSE_BEFORE, CLOSE_BEFORE, CLOSE_TARGET) == (set(), "no_source_change")
    assert target_hunk_lines(CLOSE_BEFORE, CLOSE_AFTER, "") == (set(), "empty_target")


# --- contracts: determinism, limits, provenance -----------------------------

def test_deterministic(close_sap):
    stages = _stages(*close_sap)
    assert reduce_program(*stages) == reduce_program(*stages)
    sap, manifest = close_sap
    assert run_case(sap, manifest, StubBackend(ADAPTED)).request.prompt == \
        run_case(sap, manifest, StubBackend(ADAPTED)).request.prompt


class _Limited(StubBackend):
    def __init__(self, response, limit):
        super().__init__(response)
        self.request_limit_chars = limit
        self.calls = 0

    def generate(self, request):
        self.calls += 1
        return super().generate(request)


def test_larger_prompt_over_the_limit_is_a_typed_refusal(close_sap):
    """The visible body makes the request longer; set the backend's limit to the
    OLD request's size: the new request must be refused as limits_exceeded,
    before the backend is called, never cut down to fit."""
    sap, manifest = close_sap
    old = run_case(sap, manifest, StubBackend(ADAPTED),
                   ExecutionConfig(protect_target_hunks=False)).request
    new = run_case(sap, manifest, StubBackend(ADAPTED)).request
    assert len(new.prompt) > len(old.prompt)
    backend = _Limited(ADAPTED, limit=len(old.prompt))
    gen = generate(new, backend)
    assert gen.outcome == "limits_exceeded"
    assert backend.calls == 0 and gen.candidate is None
    # and the old request still fits exactly
    assert generate(old, _Limited(ADAPTED, limit=len(old.prompt))).outcome != "limits_exceeded"


def test_rule_recorded_in_provenance():
    assert run_provenance(None)["target_hunk_protection"] == TARGET_HUNK_PROTECTION_RULE
