"""Upstream-fidelity guard (round 3): FAITHFUL / PARTIAL / WEAKENED / NO_CHANGE.

Calibration fixtures are the round-2 units that ended VALID_PR, copied verbatim from
results/round2 (fork original, applied candidate, upstream before/after of the function).
"""

import json
from pathlib import Path

import pytest

from rcap.fidelity import (
    FAITHFUL,
    NO_CHANGE,
    PARTIAL,
    WEAKENED,
    assertions,
    judge,
    tokens,
)

FIXTURES = json.loads((Path(__file__).parent / "fixtures" / "fidelity_round2.json").read_text())


@pytest.mark.parametrize("case", FIXTURES, ids=[c["name"] for c in FIXTURES])
def test_round2_calibration(case):
    f = judge(case["fork_original"], case["candidate"], case["upstream_before"], case["upstream_after"])
    assert f.label == case["expected"], (f.label, f.reasons, f.evidence)
    if f.label == PARTIAL:
        assert f.reasons                       # a PARTIAL always says why


def test_kafka_event_queue_weakening_evidence_names_the_fork_exception():
    case = next(c for c in FIXTURES if c["name"].startswith("PR-13183"))
    f = judge(case["fork_original"], case["candidate"], case["upstream_before"], case["upstream_after"])
    assert f.label == WEAKENED
    assert any("TimeoutException.class" in e and "RejectedExecutionException.class" in e for e in f.evidence)


def test_tokens_drop_comments_and_fold_class_literals():
    assert tokens('assertEquals(Foo.class, x); // c\n/* RCAP_PH_target.1 */ "a\\"b"') == [
        "assertEquals", "(", "Foo.class", ",", "x", ")", ";", '"a\\"b"']


def test_assertions_are_outermost_with_values():
    (a,) = assertions("assertEquals(3, assertThrows(E.class, () -> f()).code());")
    assert a.values == ("assertEquals", "3", "E.class")
    assert assertions("@Test(expected = IllegalStateException.class)\npublic void t() {}")[0].values == (
        "@Test", "IllegalStateException.class")


FORK = "void t() {\n    assertEquals(3, v.version());\n    assertTrue(ok);\n}\n"
BEFORE = FORK
AFTER = "void t() {\n    assertEquals(4, v.version());\n    assertTrue(ok);\n}\n"


def test_no_change_is_an_echo_even_with_comments():
    assert judge(FORK, FORK.replace("{\n", "{ // ported\n", 1), BEFORE, AFTER).label == NO_CHANGE


def test_exact_port_is_faithful():
    assert judge(FORK, AFTER, BEFORE, AFTER).label == FAITHFUL


def test_value_in_neither_is_weakened():
    f = judge(FORK, AFTER.replace("4,", "5,"), BEFORE, AFTER)
    assert f.label == WEAKENED and "`5`" in f.evidence[0]


def test_inverted_assertion_is_weakened():
    after = FORK.replace("assertTrue(ok)", "assertFalse(ok)")
    cand = FORK.replace("assertTrue(ok)", "assertNull(ok)")
    assert judge(FORK, cand, BEFORE, after).label == WEAKENED


def test_kept_old_value_while_changing_something_else_is_weakened():
    after = "void t() {\n    assertEquals(4, v.version());\n    assertTrue(ok);\n    log();\n}\n"
    cand = "void t() {\n    assertEquals(3, v.version());\n    assertTrue(ok);\n    log();\n}\n"
    f = judge(FORK, cand, BEFORE, after)
    assert f.label == WEAKENED and any("old expectation" in e for e in f.evidence)


def test_removing_an_assertion_upstream_keeps_is_weakened():
    cand = "void t() {\n    assertEquals(4, v.version());\n}\n"
    f = judge(FORK, cand, BEFORE, AFTER)
    assert f.label == WEAKENED and any("removed" in e for e in f.evidence)


def test_adaptation_to_the_forks_api_is_faithful():
    fork = "int f() {\n    return Objects.hash(a, b);\n}\n"
    before = "int f() {\n    return Objects.hash(a, x, b);\n}\n"
    after = "int f() {\n    return Objects.hash(Arrays.hashCode(a), x, b);\n}\n"
    cand = "int f() {\n    return Objects.hash(Arrays.hashCode(a), b);\n}\n"
    assert judge(fork, cand, before, after).label == FAITHFUL


def test_missing_part_of_the_change_is_partial_with_reason():
    before = "void f() {\n    a();\n}\n"
    after = "void f() {\n    a();\n    b(1);\n    c(2);\n    d(3);\n}\n"
    cand = "void f() {\n    a();\n    b(1);\n}\n"
    f = judge(before, cand, before, after)
    assert f.label == PARTIAL and f.recall is not None and f.recall < 0.9 and "recall" in f.reasons[0]


def test_invented_code_is_partial():
    before = "void f() {\n    a();\n}\n"
    after = "void f() {\n    a();\n    b();\n}\n"
    cand = "void f() {\n    a();\n    b();\n    if (flag) { reset(counter, other); }\n}\n"
    f = judge(before, cand, before, after)
    assert f.label == PARTIAL and any("novel share" in r for r in f.reasons)


def test_comment_only_upstream_change_is_no_change_with_reason():
    before = "void f() {\n    // a the b\n    a();\n}\n"
    after = "void f() {\n    // a b\n    a();\n}\n"
    f = judge(before, after, before, after)
    assert f.label == NO_CHANGE and "comment/layout-only" in f.reasons[0]
