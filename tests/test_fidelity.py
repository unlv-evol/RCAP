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


# Two round-2 labels change with the guards added after round 3 (the fixtures themselves
# are untouched):
# * TaskMovement was read as a real port, but its applied candidate still carries
#   `/* RCAP_PH_target.1 */` and `/* RCAP_PH_target.2 */` into the fork (the model wrote
#   each placeholder twice; only the first is expanded);
# * SaslChannelBuilderTest (PARTIAL before) replaces the fork's own helper call
#   `defaultApiVersionsSupplier()` with an inline lambda that upstream does not ask for.
RELABELLED = {"PR-12585/TaskMovement_fn1": "PLACEHOLDER_LEAK",
              "PR-16421/SaslChannelBuilderTest_fn1": "FORK_REGRESSION"}


@pytest.mark.parametrize("case", FIXTURES, ids=[c["name"] for c in FIXTURES])
def test_round2_calibration(case):
    f = judge(case["fork_original"], case["candidate"], case["upstream_before"], case["upstream_after"])
    assert f.label == RELABELLED.get(case["name"], case["expected"]), (f.label, f.reasons, f.evidence)
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


def test_text_starting_inside_a_doc_comment():
    before = "   * a the b\n   */\n  def f(): Unit = g()\n"
    after = "   * a b\n   */\n  def f(): Unit = g()\n"
    assert tokens(before) == tokens(after)
    assert judge(before, after, before, after).label == NO_CHANGE


def test_unrelated_edit_is_partial_and_says_so():
    before = "import a.{b}\nput(K, \"x\")\n"
    after = "import a.{b, c}\nput(K, \"x\")\n"
    cand = "import a.{b}\nput(K, \"y\")\n"
    f = judge(before, cand, before, after)
    assert f.label == PARTIAL and f.reasons[0].startswith("UNRELATED")


def test_excerpts_with_hunk_headers_and_cut_comments():
    before = "@@ -1,3 +1,3 @@ class A\n  /**\n   * a the b\n\n@@ -9,3 +9,3 @@ class A\n   * x a y\n   */\n  def f() = 1\n"
    after = before.replace("a the b", "a b").replace("x a y", "x an y")
    assert tokens(before) == tokens(after) == ["def", "f", "(", ")", "=", "1"]


# ---------------------------------------------------------------- fork preservation (after round 3)
from rcap.fidelity import (  # noqa: E402
    FORK_REGRESSION,
    NOTHING_TO_PORT,
    NOTHING_TO_PORT_IN_ISOLATION,
    declared_locally,
    fork_regressions,
    pure_rename,
)

PR15744 = json.loads((Path(__file__).parent / "fixtures" / "fidelity_round3_pr15744.json").read_text())


def test_pr15744_kafkazkclient_apply_is_a_fork_regression():
    """Round 3 labelled this FAITHFUL (recall 1.0, novel 0.0): every token exists upstream
    or in the fork. The candidate replaces the fork's `paginateTopics` constructor argument
    with upstream's new `enableEntityConfigControllerCheck`."""
    c = PR15744
    f = judge(c["fork_original"], c["candidate"], c["upstream_before"], c["upstream_after"])
    assert f.label == FORK_REGRESSION and f.recall == 1.0
    assert "paginateTopics" in f.evidence[0] and "enableEntityConfigControllerCheck" in f.evidence[0]
    # the correct port keeps the fork argument and appends upstream's
    good = c["candidate"].replace("new KafkaZkClient(zooKeeperClient, isSecure, time, enableEntityConfigControllerCheck)",
                                  "new KafkaZkClient(zooKeeperClient, isSecure, time, paginateTopics, enableEntityConfigControllerCheck)")
    assert good != c["candidate"]
    assert fork_regressions(c["fork_original"], good, c["upstream_before"], c["upstream_after"]) == []


RB = "void f() {\n  Client c = make(host, time);\n  c.start();\n}\n"
RA = "void f() {\n  Client c = make(host, time, retries);\n  c.start();\n}\n"
RFORK = "void f() {\n  Client c = make(host, time, forkFlag);\n  c.start();\n  audit(forkLog);\n}\n"


def test_fork_only_argument_must_survive_an_upstream_insertion_next_to_it():
    kept = RFORK.replace("forkFlag)", "forkFlag, retries)")
    assert judge(RFORK, kept, RB, RA).label == FAITHFUL
    displaced = RFORK.replace("forkFlag)", "retries)")
    f = judge(RFORK, displaced, RB, RA)
    assert f.label == FORK_REGRESSION and "forkFlag" in f.evidence[0]
    dropped_stmt = RFORK.replace("forkFlag)", "forkFlag, retries)").replace("  audit(forkLog);\n", "")
    f = judge(RFORK, dropped_stmt, RB, RA)
    assert f.label == FORK_REGRESSION and "forkLog" in f.evidence[0]


def test_fork_only_code_may_go_when_upstream_rewrites_that_expression():
    before = "void f() {\n  Client c = make(host, time);\n  c.start();\n}\n"
    after = "void f() {\n  Client c = Client.builder().build();\n  c.start();\n}\n"
    fork = "void f() {\n  Client c = make(host, time, forkFlag);\n  c.start();\n}\n"
    assert fork_regressions(fork, after, before, after) == []       # upstream removed the call it sat in
    # moved inside the rewritten block: still there
    moved = fork.replace("make(host, time, forkFlag)", "make(forkFlag, host, time)")
    assert fork_regressions(fork, moved, RB, RA) == []
    assert fork_regressions(RFORK, RFORK, RB, RA) == []                 # an echo displaces nothing


def test_pure_rename_detection_and_locality():
    b = "int f(Cfg c) {\n  return c.logConfig().size() + helper(c);\n}\n"
    a = "int f(Cfg c) {\n  return c.topicConfig().size() + helper(c);\n}\n"
    assert pure_rename(b, a) == [("logConfig", "topicConfig")]
    assert pure_rename(b, a.replace("helper(c)", "helper(c, 1)")) is None       # more than a rename
    assert pure_rename(b, b) is None
    assert pure_rename("int f() { return a + a; }", "int f() { return b + a; }") is None   # inconsistent
    local_b, local_a = "int f() {\n  int n = g();\n  return n;\n}\n", "int f() {\n  int count = g();\n  return count;\n}\n"
    assert pure_rename(local_b, local_a) == [("n", "count")]
    assert declared_locally(local_a, "count") and not declared_locally(a, "topicConfig")
    assert declared_locally("def f(count: Int) = count", "count") and declared_locally("xs.map(count -> count)", "count")


def test_unchanged_unit_of_a_pure_rename_is_nothing_to_port_only_when_the_caller_says_so():
    b = "int f(Cfg c) {\n  return c.logConfig().size();\n}\n"
    a = "int f(Cfg c) {\n  return c.topicConfig().size();\n}\n"
    fork = "int f(Cfg c) {\n  // fork\n  return c.logConfig().size();\n}\n"
    assert judge(fork, fork, b, a).label == NO_CHANGE                               # default unchanged
    iso = judge(fork, fork, b, a, rename_context="isolation")
    assert iso.label == NOTHING_TO_PORT_IN_ISOLATION and "logConfig -> topicConfig" in iso.reasons[0]
    assert judge(fork, fork, b, a, rename_context="pr").label == NOTHING_TO_PORT
    # the fork already declares the new name: the rename was portable, this is an echo
    assert judge(fork, fork, b, a, rename_context="isolation", fork_has=lambda n: n == "topicConfig").label == NO_CHANGE
    # a local rename is portable in isolation: unchanged is an echo
    lb, la = "int f() {\n  int n = g();\n  return n;\n}\n", "int f() {\n  int count = g();\n  return count;\n}\n"
    assert judge(lb, lb, lb, la, rename_context="isolation").label == NO_CHANGE
    # a unit that changes more than names is never exempt
    assert judge(fork, fork, b, a.replace("size()", "size() + 1"), rename_context="isolation").label == NO_CHANGE
