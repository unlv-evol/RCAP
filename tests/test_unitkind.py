"""Unit kinds: whole-file targets are described and answered as files, a renamed
function's whole-file target is narrowed to its counterpart, and function units
are left exactly as they were."""

from __future__ import annotations

import hashlib
import shutil

import pytest

from rcap.generate import StubBackend, generate
from rcap.pipeline import _expected_placeholders, run_case
from rcap.reprompt import RepairFeedback, fit_feedback
from rcap.request import (
    INSTRUCTIONS,
    WHOLE_TEMPLATE_ID,
    synthesize,
    unit_kinds,
    untraceable_lines,
)
from rcap.unitkind import (
    FILE,
    FUNCTION,
    OTHER,
    REGION,
    TYPE,
    locate_counterpart,
    text_kind,
)
from tests.conftest import F_S, F_S_PRIME, F_T

JAVA_FILE = """/* licence */
package kafka;

import java.util.List;

public class Validator {
    private final int limit = 3;

    void validate(Config c) {
        if (c == null) {
            throw new NullPointerException("config");
        }
        int v = c.getInt("version");
        if (v > 2)
            setup(v);
        for (Node n : c.nodes()) {
            registerNode(n);
        }
    }

    void setup(int v) {
    }

    void setup(int v, boolean force) {
    }
}
"""
SCALA_FN = """@Test
def testIt(): Unit = {
  val x = compute(1)
  assertEquals(2, x)
}
"""
SCALA_FILE = "package kafka.server\n\nimport kafka.utils.TestUtils\n\nclass FooTest {\n" \
             + "".join("  " + ln + "\n" for ln in SCALA_FN.splitlines()) + "}\n"
REGION_BEFORE = "@@ -3,3 +3,3 @@ package kafka;\nimport java.util.List;\nimport java.util.Map;\n"
REGION_AFTER = "@@ -3,3 +3,3 @@ package kafka;\nimport java.util.List;\nimport java.util.Set;\n"


# ---------------------------------------------------------------- kinds
@pytest.mark.parametrize("text,lang,kind", [
    (F_S, "java", FUNCTION),
    ("@Override\npublic int size() { return 0; }\n", "java", FUNCTION),
    ("StateStore getStore(final String name);\n", "java", FUNCTION),      # interface method
    (JAVA_FILE, "java", FILE),
    ("/* only a licence */\n/*\npackage kafka;\nclass Gone {}\n*/\n", "java", FILE),  # commented out
    ("public class A {\n  int x;\n}\n", "java", TYPE),
    (REGION_BEFORE, "java", REGION),
    ("int a = 1;\nint b = 2;\n", "java", OTHER),
    ("void a() {}\nvoid b() {}\n", "java", OTHER),                        # two members
    ("", "java", OTHER),
    (SCALA_FN, "scala", FUNCTION),
    (SCALA_FILE, "scala", FILE),
])
def test_text_kind(text, lang, kind):
    assert text_kind(text, lang) == kind


# ---------------------------------------------------------------- narrowing
def test_counterpart_is_found_under_the_before_name_of_a_renamed_function():
    before = "void validate(Config c) {\n    check(c);\n    run(c);\n}\n"
    after = before.replace("validate", "validateAll")        # the change renames it
    hit = locate_counterpart(before, after, JAVA_FILE, "java")
    assert hit["found"] and hit["matched_as"] == "before_name" and hit["match"] == "unique"
    assert hit["text"].startswith("void validate(Config c) {") and hit["text"].endswith("}\n")
    assert JAVA_FILE.count(hit["text"]) == 1                 # spliceable by exact text
    assert hit["line"] == 9


def test_counterpart_overloads_are_resolved_by_arity_or_not_at_all():
    two = "void setup(int v, boolean force) {\n    go(v);\n}\n"
    hit = locate_counterpart(two, two.replace("go", "run"), JAVA_FILE, "java")
    assert hit["found"] and hit["match"] == "arity" and "boolean force" in hit["text"]
    three = "void setup(int a, int b, int c) {\n    go(a);\n}\n"
    miss = locate_counterpart(three, three.replace("go", "run"), JAVA_FILE, "java")
    assert not miss["found"] and "`setup`: 2 declaration(s)" in miss["reason"]


def test_counterpart_absent_or_mis_sliced_is_never_guessed():
    gone = "void other(Config c) {\n    go(c);\n}\n"
    miss = locate_counterpart(gone, gone.replace("go", "run"), JAVA_FILE, "java")
    assert not miss["found"] and "`other`: 0 declaration(s)" in miss["reason"]
    # SALP sliced two unrelated functions as before/after: `validate` exists, but it is not
    # the counterpart of anything
    unrelated = "int size(List<String> xs, Map<String, Integer> m) {\n    return xs.size() + m.size();\n}\n"
    miss = locate_counterpart(F_S, unrelated, JAVA_FILE, "java")
    assert not miss["found"] and "two different functions" in miss["reason"]


# ---------------------------------------------------------------- pipeline
def _answer(res, edit=lambda t: t):
    """Generate for `res`'s request with a stub that returns the (reduced) target, edited."""
    text = edit(res.context.transformation["target"])
    return generate(res.request, StubBackend("```java\n" + text + "```"),
                    expected_placeholders=_expected_placeholders(res.context))


def _variant(sap_dir, tmp_path, *, before=None, after=None, target=None):
    """The fixture SAP with other payloads (the SAP records no payload hashes)."""
    pr = tmp_path / "PR-1"
    shutil.copytree(sap_dir.parent, pr)
    fn = pr / sap_dir.name / "functions" / "fn-1"
    for name, text in (("source.before.java", before), ("source.after.java", after),
                       ("target.java", target)):
        if text is not None:
            (fn / name).write_text(text)
    return pr / sap_dir.name


def test_function_unit_is_untouched(sap_dir, pr_manifest):
    res = run_case(sap_dir, pr_manifest, StubBackend("```java\n" + F_T + "```"))
    off = run_case(sap_dir, pr_manifest, StubBackend("```java\n" + F_T + "```"),
                   narrow_whole_file_targets=False)
    assert unit_kinds(res.context) == (FUNCTION, FUNCTION, FUNCTION)
    assert res.request.answer_kind == FUNCTION and res.request.template_id == "rcap-request-v1"
    assert res.request.prompt == off.request.prompt
    assert "narrowing" not in res.context.target_localization
    assert res.context.model_dump_json() == off.context.model_dump_json()
    assert res.materialized.narrowed == {}
    assert _answer(res).outcome == "completion"


def test_whole_file_target_of_a_function_is_narrowed_to_its_counterpart(
        sap_dir, pr_manifest, tmp_path):
    sap = _variant(sap_dir, tmp_path, target=JAVA_FILE)
    res = run_case(sap, pr_manifest, StubBackend("```java\n" + F_T + "```"))
    loc = res.context.target_localization["narrowing"]
    assert loc["found"] and loc["from"] == "whole_file" and loc["name"] == "validate"
    # an ordinary function unit again: the function request, the function contract
    assert unit_kinds(res.context) == (FUNCTION, FUNCTION, FUNCTION)
    assert res.request.template_id == "rcap-request-v1" and res.request.answer_kind == FUNCTION
    assert INSTRUCTIONS.format(lang="java") in res.request.prompt
    shown = res.context.transformation["target"]            # reduced like any function
    assert shown.startswith("void validate(Config c) {") and "package kafka" not in shown
    payload = res.materialized.by_role("functions/fn-1")["target"]
    assert JAVA_FILE.count(payload.content) == 1             # spliceable by exact text
    assert payload.derivation
    assert payload.sha256 == hashlib.sha256(payload.content.encode()).hexdigest()
    assert any("narrowed from the SAP's whole-file target payload" in d
               for d in res.materialized.diagnostics)
    assert _answer(res, lambda t: t.replace("v > 2", "v >= 2")).outcome == "completion"


def test_narrowing_can_be_switched_off(sap_dir, pr_manifest, tmp_path):
    sap = _variant(sap_dir, tmp_path, target=JAVA_FILE)
    res = run_case(sap, pr_manifest, StubBackend("```java\n" + JAVA_FILE + "```"),
                   narrow_whole_file_targets=False)
    assert unit_kinds(res.context)[2] == FILE and res.materialized.narrowed == {}
    assert "narrowing" not in res.context.target_localization


def test_function_without_counterpart_gets_a_file_request_and_a_file_contract(
        sap_dir, pr_manifest, tmp_path):
    whole = JAVA_FILE.replace("void validate(Config c)", "void check(Config c)")
    sap = _variant(sap_dir, tmp_path, target=whole)
    # the answer the 477 run got for such units: the source function alone
    res = run_case(sap, pr_manifest, StubBackend("```java\n" + F_S_PRIME + "```"))
    assert unit_kinds(res.context) == (FUNCTION, FUNCTION, FILE)
    req = res.request
    assert req.template_id == WHOLE_TEMPLATE_ID and req.answer_kind == FILE
    p = req.prompt
    assert "No function of that name was located in the fork" in p
    assert "## Target file in the fork (adapt this; return all of it)" in p
    assert "Return ONLY the complete adapted target file" in p
    assert "do not add the source function to it" in p
    assert "NOTE: the source function was not located in this file: no unique counterpart" in p
    assert "target function" not in p            # nothing calls the file a function
    assert untraceable_lines(p, res.context) == []
    assert synthesize(res.context).request_hash == req.request_hash
    g = res.generation
    assert g.outcome == "shape_mismatch" and g.candidate is None
    assert "the target is a whole file but the answer is one function" in g.diagnostics[0]
    assert res.package is None
    # the whole file, adapted or unchanged, is a completion
    assert _answer(res, lambda t: t.replace("v > 2", "v >= 2")).outcome == "completion"
    assert _answer(res).outcome == "completion"


def test_file_level_unit_is_described_as_a_file(sap_dir, pr_manifest, tmp_path):
    sap = _variant(sap_dir, tmp_path, before=REGION_BEFORE, after=REGION_AFTER, target=JAVA_FILE)
    res = run_case(sap, pr_manifest, StubBackend("```java\nimport java.util.Set;\n```"))
    assert unit_kinds(res.context) == (REGION, REGION, FILE)
    p = res.request.prompt
    assert res.request.template_id == WHOLE_TEMPLATE_ID
    assert "it edits file-level code (imports,\nfields, declarations, or several members)" in p
    assert "## Source file BEFORE the change (changed parts only)" in p
    assert "## Source file AFTER the change (changed parts only)" in p
    assert "## Target file in the fork (adapt this; return all of it)" in p
    assert "Source function" not in p and "target function" not in p
    assert res.materialized.narrowed == {}       # the source is not a function: nothing to locate
    # an excerpt is not an answer to a file request; the whole file is
    assert res.generation.outcome == "shape_mismatch"
    assert _answer(res, lambda t: t.replace(
        "import java.util.List;", "import java.util.List;\nimport java.util.Set;")).outcome == "completion"


def test_added_function_is_described_as_an_addition(sap_dir, pr_manifest, tmp_path):
    sap = _variant(sap_dir, tmp_path, before="// (function does not exist before the change)\n",
                   after=F_S.replace("validate", "brandNew"), target=JAVA_FILE)
    res = run_case(sap, pr_manifest, StubBackend("```java\n" + JAVA_FILE + "```"))
    assert unit_kinds(res.context)[1:] == (FUNCTION, FILE)
    assert "The change ADDS the function shown below" in res.request.prompt


def test_repair_request_of_a_whole_file_unit_speaks_of_the_file(sap_dir, pr_manifest, tmp_path):
    whole = JAVA_FILE.replace("void validate(Config c)", "void check(Config c)")
    sap = _variant(sap_dir, tmp_path, target=whole)
    res = run_case(sap, pr_manifest, StubBackend("```java\n" + whole + "```"))
    fb = RepairFeedback(attempt=2, kind="build", previous_candidate=whole,
                        error_text="- Validator.java:9: cannot find symbol")
    ctx, req = fit_feedback(res.context, fb, None)
    p = req.prompt
    assert req.answer_kind == FILE
    assert "## ORIGINAL file in the fork, before the port" in p
    assert "The file is the whole target: return ALL of it" in p
    assert "ORIGINAL function" not in p and "complete fixed function" not in p
    assert untraceable_lines(p, ctx) == []
    bad = generate(req, StubBackend("```java\n" + F_S + "```"), allow_decline=True)
    assert bad.outcome == "shape_mismatch"
    declined = generate(req, StubBackend("CANNOT_ADAPT: needs check from Validator.java"),
                        allow_decline=True)
    assert declined.outcome == "needs_cochange"
