"""Scoring fixes and pipeline bug fixes of 2026-10 (audit of 2026-10-06).

The fixtures in fixtures/audit_2026_10.json are the audit's own cases, copied verbatim from the
run outputs (see each entry's `source`). The labels were FITTED on the audit's hand judgements;
these tests pin the fitted behaviour, they are not evidence that it generalises."""

import hashlib
import json
from pathlib import Path

import pytest

from rcap import fidelity as fd
from rcap import scoring as sc
from rcap.accept import duplicated_blocks, has_placeholder_leak, placeholder_marks, rejection
from rcap.generate import StubBackend, generate, generate_with_placeholder_retry, placeholder_report
from rcap.pipeline import run_case
from rcap.provenance import run_provenance
from rcap.reprompt import UNIT_CHANGED, UNIT_ECHO, UNIT_REVERTED, unit_states
from rcap.request import AdaptationRequest
from rcap.splice import CONFLICT, MERGED, SHAPE_MISMATCH, UnitEdit, check_invariants, merge_units
from rcap.unitkind import shape_mismatch

FX = json.loads((Path(__file__).parent / "fixtures" / "audit_2026_10.json").read_text())


def judge(name, **kw):
    c = FX[name]
    up = tuple(c["upstream_file"]) if c.get("upstream_file") else None
    return fd.judge(c["fork_original"], c["candidate"], c["upstream_before"], c["upstream_after"],
                    language=c["language"], upstream_file=up, **kw)


def score(name, **kw):
    c = FX[name]
    return sc.score_unit(c["fork_original"], c["candidate"], c["upstream_before"], c["upstream_after"],
                         language=c["language"], **kw)


# ------------------------------------------------------------------ 1. per-unit echo / reversion
REVERSIONS = ["reverted_pr_12789_abstractherder", "reverted_pr_12478_retrywithtoleranceoperator",
              "reverted_pr_15154_retrywithtoleranceoperator", "reverted_pr_14483_remoteindexcachetest"]


@pytest.mark.parametrize("name", REVERSIONS)
def test_reprompted_unit_back_at_the_forks_code_is_reverted_and_the_row_is_no_port(name):
    c = FX[name]
    # the unit differed from the fork in attempt 1 and is the fork's code after the re-prompt
    assert unit_states([c["attempt1"]], [c["fork_original"]]) == [UNIT_CHANGED]
    assert unit_states([c["candidate"]], [c["fork_original"]], [c["attempt1"]]) == [UNIT_REVERTED]
    u = score(name, earlier=[c["attempt1"]])
    assert u.label == sc.REVERTED and not u.is_code_port
    # without the earlier attempt it is an echo, and still not a port
    assert score(name).label in (fd.NO_CHANGE, fd.NOTHING_TO_PORT) and not score(name).is_code_port
    # the row: a sibling unit with a real change does not make it a port (PR-12789 AbstractHerder arm B)
    sibling = sc.UnitScore(fd.FAITHFUL_CODE)
    row = sc.score_row("VALID", [sibling, u])
    assert row.row_class == sc.ROW_REVERTED and not row.real_port


def test_unit_states_ignore_comments_and_layout():
    fork = "void f() {\n    a();\n}\n"
    assert unit_states(["void f() {\n    // note\n    a();\n}\n", "void f() { b(); }\n"], [fork, fork]) == [UNIT_ECHO, UNIT_CHANGED]
    # an echo from the first attempt on is an echo, not a reversion
    assert unit_states([fork], [fork], [fork]) == [UNIT_ECHO]


def test_row_verdict_is_derived_from_its_applied_units():
    ok, imp, echo = sc.UnitScore(fd.FAITHFUL_CODE), sc.UnitScore(fd.IMPORT_ONLY), sc.UnitScore(fd.NO_CHANGE)
    assert sc.score_row("VALID", [ok, echo]).real_port                       # one real change, nothing bad
    assert sc.score_row("VALID", [echo, echo]).row_class == sc.ROW_NO_CHANGE   # the row passes; nothing was ported
    assert sc.score_row("VALID", [imp, echo]).row_class == sc.ROW_IMPORT and not sc.score_row("VALID", [imp]).real_port
    for bad, cls in ((fd.PLACEHOLDER_LEAK, sc.ROW_LEAK), (fd.DUPLICATED_BLOCK, sc.ROW_DUPLICATE),
                     (fd.FORK_REGRESSION, sc.ROW_REGRESSION), (fd.WEAKENED, sc.ROW_REGRESSION)):
        r = sc.score_row("VALID", [ok, sc.UnitScore(bad)])
        assert r.row_class == cls and not r.real_port
    # a unit that never reached the file counts for nothing, good or bad
    assert sc.score_row("VALID", [echo, sc.UnitScore(fd.FAITHFUL_CODE, applied=False)]).row_class == sc.ROW_NO_CHANGE
    assert sc.score_row("VALID", [ok, sc.UnitScore(fd.PLACEHOLDER_LEAK, applied=False)]).real_port
    # REAL_PORT needs the build verdict as well
    assert sc.score_row("BUILD_FAIL", [ok]).row_class == sc.ROW_CODE and not sc.score_row("BUILD_FAIL", [ok]).real_port
    assert sc.score_row("VALID_PR", [ok]).real_port
    # a hand-confirmed adapted port counts although the automatic label is PARTIAL
    assert sc.score_row("VALID", [sc.UnitScore(fd.PARTIAL, hand_confirmed=True)]).real_port
    assert sc.score_row("VALID", [sc.UnitScore(fd.PARTIAL)]).row_class == sc.ROW_PARTIAL


# ------------------------------------------------------------------ 2. placeholders and duplicated blocks
def test_placeholder_marker_of_any_spelling_is_a_leak():
    line = FX["placeholder_15476"]["line"]                   # /* RCAP_PHname */: no underscore
    assert "RCAP_PH_" not in line and has_placeholder_leak(line)
    assert placeholder_marks(line) == ["RCAP_PHsetUpForLogAppendTimeCase"]
    for text in ("/* RCAP_PH_target.1 */", "// RCAP_PH_1", "/*RCAP_PH*/", "x = RCAP_PHfoo;"):
        assert has_placeholder_leak(text), text
    assert not has_placeholder_leak("int rcapPh = 1; // RCAP placeholder") and not has_placeholder_leak(None)
    fork = "void f() {\n  a();\n}\n"
    assert fd.judge(fork, "void f() {\n  a();\n  " + line.strip() + "\n}\n", fork, fork).label == fd.PLACEHOLDER_LEAK
    assert rejection(line)[0] == "placeholder_leak"


def test_generate_rejects_the_underscore_less_marker():
    p = "## Target\n```java\nvoid f() { a(); }\n```\n"
    r = AdaptationRequest(case_id="c", prompt=p, request_hash=hashlib.sha256(p.encode()).hexdigest())
    g = generate(r, StubBackend("```java\nvoid f() { /* RCAP_PHname */ a(); }\n```"), expected_placeholders=set())
    assert g.outcome == "placeholder_violation" and g.candidate is None
    assert placeholder_report("/* RCAP_PHname */", set())["unknown"] == ["1 malformed mention(s) of RCAP_PH_"]


DUPLICATES = ["dup_batchbuilder", "dup_oauth", "dup_protocolser", "dup_apierror", "dup_kafkabasedlog"]


@pytest.mark.parametrize("name", DUPLICATES)
def test_duplicated_block_fixtures(name):
    c = FX[name]
    dup = duplicated_blocks(c["fork_original"], c["candidate"], c["upstream_before"], c["upstream_after"], language=c["language"])
    assert dup and dup[0]["copies_candidate"] > dup[0]["copies_fork"] >= 1
    assert judge(name).label == fd.DUPLICATED_BLOCK
    u = score(name)
    assert not u.is_code_port and sc.score_row("VALID", [u]).row_class == sc.ROW_DUPLICATE


def test_duplicated_block_leaves_ordinary_repetition_alone():
    fork = ("class T {\n  void a() {\n    Map<String, Object> c = new HashMap<>();\n    c.put(KEY, k);\n    c.put(VALUE, v);\n    use(c);\n  }\n}\n")
    # a NEW method that repeats the set-up lines of its neighbour is ordinary code
    new_test = fork.replace("  }\n}\n", "  }\n  void b() {\n    Map<String, Object> c = new HashMap<>();\n    c.put(KEY, k);\n    c.put(VALUE, v);\n    other(c);\n  }\n}\n")
    assert duplicated_blocks(fork, new_test, language="java") == []
    # the same lines once more in the body that had them is not
    twice = fork.replace("    use(c);\n", "    use(c);\n    Map<String, Object> c = new HashMap<>();\n    c.put(KEY, k);\n    c.put(VALUE, v);\n")
    assert duplicated_blocks(fork, twice, language="java")
    # ... unless upstream repeats the block as well
    assert duplicated_blocks(fork, twice, fork, twice, language="java") == []
    # the real ports of the audit are not flagged
    for name in ("code_real_14026", "partial_false_15154", "partial_false_18611", "fr_kafkazkclient"):
        c = FX[name]
        assert duplicated_blocks(c["fork_original"], c["candidate"], c["upstream_before"], c["upstream_after"], language=c["language"]) == [], name
    # an unparseable candidate is not judged (it does not build either)
    assert duplicated_blocks(fork, twice.replace("class T {", "class T {{{"), language="java") == []


def test_placeholder_kept_and_block_written_out_again_is_rejected_by_the_pipeline(sap_dir, pr_manifest):
    # whatever the stub answers, the pipeline hands `generate` a duplicate check for the unit
    from rcap.pipeline import duplicate_check
    res = run_case(sap_dir, pr_manifest, StubBackend("```java\nvoid validate(Config c) { }\n```"))
    reject = duplicate_check(res.units[0].context, res.case.language)
    art = next(a for a in res.units[0].context.program_context if a.role == "target")
    assert reject is not None and reject(art.reduced_text) is None        # the reduced target itself is fine
    if art.placeholders:
        e = art.placeholders[0]
        if e.original_text.count("\n") >= 3:
            ph = f"/* RCAP_PH_{e.ph_id} */"
            both = art.reduced_text.replace(ph, e.original_text + "\n" + ph, 1)
            out = reject(both)
            assert out and out[0] == "duplicated_block"


# ------------------------------------------------------------------ 7. placeholder retry
class Scripted:
    name, model = "scripted", "s-0"

    def __init__(self, answers):
        self.answers, self.prompts = list(answers), []

    def generate(self, request):
        self.prompts.append(request.prompt)
        return self.answers.pop(0)


@pytest.mark.parametrize("name", ["dup_batchbuilder", "dup_oauth"])
def test_retry_answer_with_a_duplicated_block_is_not_accepted(name):
    """PR-14026 BatchBuilder and PR-13212 OAuthBearerValidationUtils: both wrong passes of the
    first retry. The retry's answer is the stored candidate; it must be refused and the ORIGINAL
    placeholder refusal returned."""
    c = FX[name]
    ph = "/* RCAP_PH_target.1 */"
    p = "## Target\n```java\n" + c["fork_original"] + "```\n"
    r = AdaptationRequest(case_id="c", prompt=p, request_hash=hashlib.sha256(p.encode()).hexdigest())

    def reject(candidate):
        return rejection(candidate.replace(ph, ""), c["fork_original"], c["upstream_before"], c["upstream_after"],
                         language=c["language"])
    answer = "```java\n" + c["candidate"].rstrip("\n").rsplit("}", 1)[0] + ph + "\n}\n```"
    b = Scripted(["```java\n" + c["fork_original"] + "```", answer])
    g, info = generate_with_placeholder_retry(r, b, expected_placeholders={ph}, reject=reject)
    assert len(b.prompts) == 2 and info["retry_outcome"] == "duplicated_block" and not info["fixed"]
    assert g.outcome == "placeholder_violation" and g.candidate is None
    # a sound retry answer is still accepted
    fine = "```java\n" + c["fork_original"].rstrip("\n").rsplit("}", 1)[0] + ph + "\n}\n```"
    g, info = generate_with_placeholder_retry(r, Scripted(["```java\n" + c["fork_original"] + "```", fine]),
                                               expected_placeholders={ph}, reject=reject)
    assert g.outcome == "completion" and info["fixed"]


# ------------------------------------------------------------------ 3. fidelity labels
def test_faithful_is_split_into_code_import_and_comment():
    assert judge("code_real_14026").label == fd.FAITHFUL_CODE
    f = judge("import_12398")
    assert f.label == fd.IMPORT_ONLY and f.kind == "import" and not sc.UnitScore(f.label).is_code_port
    f = judge("import_incomplete_15149")       # the import without the @Inject the unit is about
    assert f.label == fd.IMPORT_ONLY and "code_not_ported" in f.flags
    for name in ("comment_13132", "comment_12984"):
        assert judge(name).label == fd.COMMENT_ONLY, name
    assert fd.FAITHFUL == fd.FAITHFUL_CODE


def test_code_of_a_sibling_unit_needs_the_upstream_file():
    c = FX["sibling_code_15149"]               # unit = the import; the candidate also adds @Inject
    assert c["upstream_file"], "fixture carries the upstream file"
    f = judge("sibling_code_15149")
    assert f.label == fd.FAITHFUL_CODE and "code_of_sibling_unit" in f.flags
    g = fd.judge(c["fork_original"], c["candidate"], c["upstream_before"], c["upstream_after"], language="java")
    assert g.label == fd.PARTIAL and "unrequested_code_edit" in g.flags


def test_fork_regression_true_cases_and_value_equivalent_rewrites():
    f = judge("fr_kafkazkclient")              # fork-only argument paginateTopics displaced
    assert f.label == fd.FORK_REGRESSION and "paginateTopics" in f.evidence[0]
    assert judge("fr_saslchannelbuildertest").label == fd.FORK_REGRESSION
    # TimeUnit.NANOSECONDS.convert(1, HOURS) -> HOURS.toNanos(1); MetaProperties(clusterId = x) -> MetaProperties(x)
    # no regression; the rewrite next to the import is an edit nobody asked for, so not IMPORT_ONLY either
    for name in ("fr_false_13089", "fr_false_14628"):
        f = judge(name)
        assert f.label == fd.PARTIAL and "unrequested_code_edit" in f.flags, (name, f.label, f.evidence[:1])
    # what counts as fork state
    ts = fd.tokens("x = TimeUnit.NANOSECONDS.convert(1, u); client.setFollow(false); f(name = v, w); g().h; helper();")
    state = {t: fd._is_fork_state(ts, i) for i, t in enumerate(ts)}
    assert not state["NANOSECONDS"] and not state["convert"] and not state["name"]
    assert state["setFollow"] and state["helper"] and state["h"] and state["v"]


def test_revert_to_upstream_before_is_a_regression():
    f = judge("revert_16170")
    assert f.label == fd.FORK_REGRESSION and "reverts_to_upstream_before" in f.flags
    fork = "void f() {\n    a(1);\n    b(2);\n    c(3);\n}\n"
    before = "void f() {\n    old(first, x);\n    old(second, y);\n    c(3);\n}\n"
    after = "void f() {\n    fresh(first, x);\n    c(3);\n}\n"
    assert fd.reverts_to_before(fork, before, before, after) == ["old(first, x);", "old(second, y);"]
    assert fd.reverts_to_before(fork, after, before, after) == []


@pytest.mark.parametrize("name", ["partial_false_15154", "partial_false_18611", "partial_false_13885", "partial_false_12250"])
def test_known_partial_false_alarms_are_faithful_code(name):
    f = judge(name)
    assert f.label == fd.FAITHFUL_CODE, (f.label, f.reasons, f.evidence[:2])


@pytest.mark.parametrize("name", ["partial_true_15154", "partial_true_12584", "partial_true_16421"])
def test_real_partials_stay_partial(name):
    assert judge(name).label == fd.PARTIAL


def test_the_rules_behind_the_partial_fixes():
    # a call cut off by the end of a diff excerpt is not an assertion of the next excerpt
    ex = "@@ -1,2 +1,2 @@\n    assertTrue(a,\n@@ -9,1 +9,1 @@\n    x();\n    assertEquals(1, y);\n"
    assert [a.values for a in fd.assertions(ex)] == [("assertEquals", "1")]
    # adapted name: upstream's d3 written as d2 where the fork has no d3
    fork = "void t() {\n    Date d2 = conv(a);\n}\n"
    before = "void t() {\n    Date d3 = conv(a);\n}\n"
    after = "void t() {\n    Date d3 = conv(a);\n    LocalDateTime date3 = of(d3.getTime());\n    assertEquals(want, date3);\n}\n"
    cand = "void t() {\n    Date d2 = conv(a);\n    LocalDateTime date2 = of(d2.getTime());\n    assertEquals(want, date2);\n}\n"
    f = fd.judge(fork, cand, before, after)
    assert f.label == fd.FAITHFUL_CODE and "date3 -> date2" in f.adapted
    # an inconsistent substitution is not an adaptation
    bad = cand.replace("assertEquals(want, date2)", "assertEquals(want, other)")
    assert fd.judge(fork, bad, before, after).label != fd.FAITHFUL_CODE
    # the same character edit on the fork's spelling of a value
    f = fd.judge('x(ibp = "3.0-IV0")\n', 'x(ibp = "3.0-IV1")\n', "x(v = V.IBP_3_0_IV0)\n", "x(v = V.IBP_3_0_IV1)\n")
    assert f.label == fd.FAITHFUL_CODE and f.adapted == ['IBP_3_0_IV1 -> "3.0-IV1"']
    assert fd.judge('x(ibp = "3.0-IV0")\n', 'x(ibp = "3.1-IV0")\n', "x(v = V.IBP_3_0_IV0)\n", "x(v = V.IBP_3_0_IV1)\n").label == fd.PARTIAL


def test_nothing_to_port_is_a_first_class_outcome():
    # (1) a pure rename of a name declared elsewhere, validated alone: the fork's function is right
    fork = "void f() {\n    getOrCreateSnapshot(1);\n}\n"
    after = "void f() {\n    idempotentCreateSnapshot(1);\n}\n"
    u = sc.score_unit(fork, fork, fork, after, rename_context="isolation", fork_has=lambda n: False)
    assert u.label == fd.NOTHING_TO_PORT_IN_ISOLATION and not u.is_code_port
    assert sc.score_row("VALID", [u]).row_class == sc.ROW_NOTHING and not sc.score_row("VALID", [u]).real_port
    # (2) a source function the fork does not have, file returned unchanged
    file_ = "package p;\n\nclass A {\n    void other() {\n        a();\n    }\n}\n"
    b, a = "void gone() {\n    x(1);\n}\n", "void gone() {\n    x(2);\n}\n"
    assert sc.has_no_counterpart(file_, b, a, "java")
    u = sc.score_unit(file_, file_, b, a)
    assert u.label == fd.NOTHING_TO_PORT and "no_counterpart" in u.fidelity.flags
    # the fork HAS the function: an unchanged file is an echo, a model miss
    has = file_.replace("void other()", "void gone()")
    assert not sc.has_no_counterpart(has, b, a, "java") and sc.score_unit(has, has, b, a).label == fd.NO_CHANGE
    # neither failure nor success in a row
    assert sc.score_row("INCONCLUSIVE", [u]).row_class == sc.ROW_NOTHING


# ------------------------------------------------------------------ 5. splice
IMPORTS = "import a.A;\nimport b.Supplier;\nimport c.C;\n\nclass X {\n    void f() {\n        g();\n    }\n}\n"


def test_insertion_at_the_start_of_a_replaced_range_is_applied_once():
    """PR-16421 SaslChannelBuilder / SaslServerAuthenticator: one unit inserts `import Function`
    before the Supplier import, the other replaces the Supplier import by Function + Supplier."""
    whole = IMPORTS
    ins = whole.replace("import b.Supplier;\n", "import b.Function;\nimport b.Supplier;\n")
    # second unit: the import block as its own region, rewritten
    region = "import b.Supplier;\nimport c.C;\n"
    rep = "import b.Function;\nimport b.Sup;\nimport c.C;\n"
    m = merge_units(whole, [UnitEdit("file", whole, ins), UnitEdit("imports", region, rep)])
    assert m.text.count("import b.Function;") == 1 and m.violations == []
    assert "import b.Sup;" in m.text and m.units["file"].status == MERGED and m.units["file"].shared == 1
    # the reverse order of acceptance gives the same file
    m2 = merge_units(whole, [UnitEdit("imports", region, rep), UnitEdit("file", whole, ins)])
    assert m2.text == m.text
    # an insertion at the start that the replacement does NOT contain is a real conflict, not a silent pair
    other = whole.replace("import b.Supplier;\n", "import z.Z;\nimport b.Supplier;\n")
    m3 = merge_units(whole, [UnitEdit("file", whole, other), UnitEdit("imports", region, rep)])
    assert m3.units["file"].status == CONFLICT and m3.text.count("import z.Z;") == 0 and m3.violations == []


def test_same_loop_from_two_units_is_applied_once():
    """PR-15486 Errors: both units add the same loop at the same place, one as an insertion and
    one inside a replacement."""
    whole = "class E {\n    static {\n        init();\n    }\n    int v;\n}\n"
    loop = "        for (E e : values()) {\n            reg(e);\n        }\n"
    u1 = whole.replace("        init();\n", loop + "        init();\n")
    region = "        init();\n    }\n"
    u2 = loop + "        init2();\n    }\n"
    m = merge_units(whole, [UnitEdit("outer", whole, u1), UnitEdit("inner", region, u2)])
    assert m.text.count("for (E e : values())") == 1 and "init2();" in m.text and m.violations == []


def test_invariants_catch_a_hunk_applied_twice():
    from rcap.splice import Hunk, MergeResult
    whole = "a\nb\nc\n"
    h1 = Hunk("u1", 2, 2, "X\n", ["u1"])
    h2 = Hunk("u2", 2, 4, "X\nB\n", ["u2"])
    bad = MergeResult(text="a\nX\nX\nB\nc\n", units={}, applied=[h1, h2], original=whole)
    assert any("applied twice" in v or "overlap" in v for v in check_invariants(bad))
    good = merge_units(whole, [UnitEdit("u", "b\n", "B\n")])
    assert good.violations == [] and check_invariants(good) == []
    # a merged file that is not the original plus the applied hunks
    tampered = MergeResult(text=good.text + "extra\n", units=good.units, applied=good.applied, original=whole)
    assert check_invariants(tampered)


# ------------------------------------------------------------------ 6. shape guard
FILE_ = "package p;\n\nimport a.B;\n\npublic class T {\n    void one() {\n        a();\n    }\n\n    void two() {\n        b();\n    }\n}\n"
METHOD = "void one() {\n        a2();\n    }\n"


def test_one_method_candidate_for_a_whole_file_unit_is_refused():
    """PR-17720 ConnectWorkerIntegrationTest: the stored candidate is one method, the unit's target
    the whole file; the splice replaced the file with the method."""
    assert "one function" in shape_mismatch(FILE_, METHOD, "java")
    assert shape_mismatch(FILE_, FILE_.replace("a();", "a2();"), "java") is None
    assert shape_mismatch("void one() {\n        a();\n    }\n", METHOD, "java") is None     # function units: never checked
    m = merge_units(FILE_, [UnitEdit("u0", FILE_, METHOD)], language="java")
    assert m.units["u0"].status == SHAPE_MISMATCH and m.text == FILE_ and not m.changed
    # without a language the guard is off (the old behaviour: the file is wiped)
    assert merge_units(FILE_, [UnitEdit("u0", FILE_, METHOD)]).text == METHOD
    # stored candidates are scored the same way, and the unit is not applied
    u = sc.score_unit(FILE_, METHOD, "void one() {\n a();\n}\n", "void one() {\n a2();\n}\n")
    assert u.label == sc.SHAPE_MISMATCH and not u.applied
    assert sc.score_row("VALID", [u]).row_class == sc.ROW_NOT_APPLIED
    # a function unit next to it is unaffected
    m = merge_units(FILE_, [UnitEdit("u0", FILE_, METHOD), UnitEdit("u1", "    void two() {\n        b();\n    }\n",
                                                                    "    void two() {\n        b2();\n    }\n")], language="java")
    assert m.units["u1"].status == MERGED and "b2();" in m.text and "package p;" in m.text


# ------------------------------------------------------------------ provenance
def test_result_rows_carry_model_and_code_version():
    class B:
        name, model, model_digest = "ollama", "qwen3-coder:30b", "qwen3moe/30.5B/Q4_K_M"
        params = {"temperature": 0, "seed": 12535, "num_ctx": 16384}
    p = run_provenance(B(), driver="t", k=1)
    assert p["model"] == "qwen3-coder:30b" and p["model_tag"] == "30b" and p["num_ctx"] == 16384
    assert p["temperature"] == 0 and p["seed"] == 12535 and p["driver"] == "t" and "rcap_commit" in p
    assert run_provenance(None)["model"] is None


# ------------------------------------------------------------------ the loop enforces it (not only the report)
def _loop_units(sap_dir, pr_manifest, first):
    from rcap.reduction_program import PLACEHOLDER, recover
    from rcap.reprompt import UnitState
    res = run_case(sap_dir, pr_manifest, StubBackend(first))
    u = res.units[0]
    ent = u.context.target_localization.get("function")
    art = next(a for a in u.context.program_context if a.role == "target" and a.entity == ent)

    def expand(c):
        for e in art.placeholders:
            c = c.replace(PLACEHOLDER.format(n=e.ph_id), e.original_text, 1)
        return c
    # the same unit twice: a two-unit row whose units can be answered differently
    return [UnitState(index=i, entity=ent, context=u.context, request=u.request, generation=u.generation,
                      target_original=recover(art), expand=expand) for i in (0, 1)], art


def test_reprompt_loop_does_not_count_a_reverted_unit_as_success(sap_dir, pr_manifest):
    from rcap.reprompt import Validation, run_reprompt_loop
    first = "```java\nvoid validate(Config c) { /* RCAP_PH_target.1 */ check(c); missing(c); }\n```"
    units, art = _loop_units(sap_dir, pr_manifest, first)
    echo = "```java\n" + art.reduced_text + "\n```"          # the fork's own code
    backend = Scripted([echo])
    backend.params = {}
    calls = []

    def validate(attempt, expanded):
        calls.append(attempt)
        if attempt == 1:
            return Validation(verdict="BUILD_FAIL", kind="build", feedback="- F.java:3: cannot find symbol missing",
                              unit_feedback={1: "- F.java:3: cannot find symbol missing"}, units_with_errors=[1], summary="1 error")
        return Validation(verdict="VALID", summary="tests ran, no new failures")
    rec = run_reprompt_loop(units, backend, validate, k=2)
    a2 = rec["attempts"][1]
    assert a2["unit_states"] == [UNIT_CHANGED, UNIT_REVERTED]      # unit 0 keeps the row different from the fork
    assert a2["echo"] is False and a2["raw_verdict"] == "VALID"   # the old, row-level guard saw nothing
    assert a2["verdict"] == "REVERTED" and rec["final_verdict"] == "REVERTED"
    assert rec["first_success_attempt"] is None and rec["stop_reason"] == "k_exhausted"


def test_reprompt_loop_never_validates_a_leaked_placeholder(sap_dir, pr_manifest):
    from rcap.reprompt import Validation, run_reprompt_loop
    first = "```java\nvoid validate(Config c) { /* RCAP_PH_target.1 */ check(c); missing(c); }\n```"
    units, art = _loop_units(sap_dir, pr_manifest, first)
    units = units[:1]
    object.__setattr__(units[0], "expand", lambda c: c + "// RCAP_PHleft\n")
    rec = run_reprompt_loop(units, Scripted([]), lambda *a: Validation(verdict="VALID"), k=2)
    assert rec["attempts"][0]["verdict"] == "PLACEHOLDER_LEAK" and rec["stop_reason"] == "placeholder_leak"
    assert rec["first_success_attempt"] is None
