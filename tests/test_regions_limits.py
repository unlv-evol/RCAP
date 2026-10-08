"""Whole-file units are reduced to the regions their change touches (rcap.regions), and the
limits a request is generated under are explicit, per backend, and recorded (capacity refusals)."""

from __future__ import annotations

import hashlib
import shutil

import pytest

from rcap import regions as rg
from rcap.backend_ollama import GenerationLimits
from rcap.generate import CAPACITY_KINDS, StubBackend, generate, generate_regions
from rcap.pipeline import run_case
from rcap.request import (
    REGION_TEMPLATE_ID,
    WHOLE_TEMPLATE_ID,
    synthesize,
    synthesize_regions,
    untraceable_lines,
)
from rcap.unitkind import FILE, REGION
from tests.conftest import F_T

IMPORTS = "".join(f"import org.example.pkg{n}.Type{n};\n" for n in range(1, 31))
BODY = "".join(
    f"\n    public int method{n}(int value) {{\n        int local{n} = value + {n};\n"
    f"        return helper{n}(local{n});\n    }}\n" for n in range(1, 41))
BIG = ("/* licence */\npackage org.example;\n\n" + IMPORTS + "\npublic class Big {\n"
       "    private static final Logger log = LoggerFactory.getLogger(Big.class);\n" + BODY + "}\n")

# upstream adds an import between Type11 and Type12 (the fork lacks Type13 upstream has)
HUNK_BEFORE = ("@@ -14,6 +14,7 @@ import org.example.pkg9.Type9;\n"
               "import org.example.pkg10.Type10;\nimport org.example.pkg11.Type11;\n"
               "import org.example.pkg12.Type12;\nimport org.example.other.Missing;\n")
HUNK_AFTER = HUNK_BEFORE.replace("import org.example.pkg12.Type12;\n",
                                 "import org.example.added.Added;\nimport org.example.pkg12.Type12;\n")
# and a constant after the logger
HUNK2_BEFORE = ("@@ -40,3 +41,4 @@ public class Big {\n"
                "    private static final Logger log = LoggerFactory.getLogger(Big.class);\n")
HUNK2_AFTER = HUNK2_BEFORE + "    public static final long TIMEOUT_MS = 90_000L;\n"


# ------------------------------------------------------------------------------ rcap.regions
def test_source_hunks_from_excerpts_and_from_a_function_pair():
    hs = rg.source_hunks(HUNK_BEFORE + HUNK2_BEFORE, HUNK_AFTER + HUNK2_AFTER)
    assert [len(h.before) for h in hs] == [4, 1] and [len(h.after) for h in hs] == [5, 2]
    assert hs[0].header.startswith("@@ -14,6 +14,7 @@") and hs[0].before_text() == HUNK_BEFORE
    fn_before = "void f() {\n    a();\n    b();\n    c();\n    d();\n    e();\n    g();\n    h();\n    i();\n}\n"
    hs = rg.source_hunks(fn_before, fn_before.replace("    e();\n", "    e(1);\n"))
    assert len(hs) == 1 and hs[0].header == "" and "    e();" in hs[0].before and "    e(1);" in hs[0].after
    assert len(hs[0].before) == 7                      # three lines of context on both sides
    assert rg.source_hunks("@@ -1,2 +1,2 @@\na\n", "@@ -9,2 +9,2 @@\na\n") is None      # headers disagree
    assert rg.source_hunks("// (function does not exist before the change)\n", "") is None


def test_a_hunk_is_anchored_by_lines_that_occur_once_and_never_guessed():
    plan = rg.plan_regions(HUNK_BEFORE, HUNK_AFTER, BIG)
    assert plan.located and len(plan.regions) == 1
    r = plan.regions[0]
    assert r.anchors == 3                               # Type10, Type11, Type12; `Missing` is not in the fork
    assert BIG[r.start:r.end] == r.text and BIG.count(r.text) == 1
    assert r.text.startswith("import org.example.pkg4.Type4;\n")        # 6 lines of context before Type10
    assert r.text.endswith("import org.example.pkg19.Type19;\n")        # and after the hunk's last line
    assert (r.first_line, r.last_line) == (7, 22)
    # braces, blank lines and `return x;` identify nothing
    nowhere = rg.plan_regions("@@ -1,3 +1,4 @@\n    }\n\n    return x;\n", "@@ -1,3 +1,4 @@\n    }\n    y();\n\n    return x;\n", BIG)
    assert not nowhere.located and "could not be anchored" in nowhere.reason
    # a line that occurs twice in the fork is not an anchor either
    twice = BIG.replace("import org.example.pkg11.Type11;\n", "import org.example.pkg10.Type10;\n")
    assert rg.plan_regions(HUNK_BEFORE, HUNK_AFTER, twice).regions[0].anchors == 1


def test_one_region_per_hunk_joined_when_they_touch_and_whole_file_when_not_worth_it():
    plan = rg.plan_regions(HUNK_BEFORE + HUNK2_BEFORE, HUNK_AFTER + HUNK2_AFTER, BIG)
    assert [r.hunks for r in plan.regions] == [[0], [1]]
    a, b = plan.regions
    assert a.end <= b.start and "LoggerFactory.getLogger" in b.text and "LoggerFactory" not in a.text
    near = rg.plan_regions(HUNK_BEFORE + HUNK2_BEFORE, HUNK_AFTER + HUNK2_AFTER, BIG, context=12)
    assert [r.hunks for r in near.regions] == [[0, 1]]                  # overlapping regions are one
    # one hunk that cannot be located: the unit is not reduced at all
    lost = rg.plan_regions(HUNK_BEFORE + "@@ -90,1 +91,1 @@\n    int gone = compute();\n",
                           HUNK_AFTER + "@@ -90,1 +91,1 @@\n    long gone = compute();\n", BIG)
    assert not lost.located and lost.reason.startswith("hunk 2 of 2")
    small = "\n".join(BIG.splitlines()[:26]) + "\n"
    assert "the whole file is asked for instead" in rg.plan_regions(HUNK_BEFORE, HUNK_AFTER, small).reason
    assert "more than 1" in rg.plan_regions(HUNK_BEFORE + HUNK2_BEFORE, HUNK_AFTER + HUNK2_AFTER, BIG, max_regions=1).reason


def test_region_answers_are_spliced_over_their_exact_spans():
    plan = rg.plan_regions(HUNK_BEFORE + HUNK2_BEFORE, HUNK_AFTER + HUNK2_AFTER, BIG)
    a, b = plan.regions
    ans_a = a.text.replace("import org.example.pkg12.Type12;\n", "import org.example.added.Added;\nimport org.example.pkg12.Type12;\n")
    ans_b = b.text.replace("getLogger(Big.class);\n", "getLogger(Big.class);\n    public static final long TIMEOUT_MS = 90_000L;\n")
    merged, why = rg.merge_regions(BIG, plan.regions, [ans_a, ans_b])
    assert why == "" and merged.count("\n") == BIG.count("\n") + 2
    want = BIG.replace("import org.example.pkg12.Type12;\n", "import org.example.added.Added;\nimport org.example.pkg12.Type12;\n") \
              .replace("getLogger(Big.class);\n", "getLogger(Big.class);\n    public static final long TIMEOUT_MS = 90_000L;\n")
    assert merged == want
    assert rg.merge_regions(BIG, plan.regions, [a.text, b.text])[0] == BIG       # unchanged parts: the file


def test_region_shape_check():
    r = rg.plan_regions(HUNK2_BEFORE, HUNK2_AFTER, BIG).regions[0]
    assert rg.region_shape_error(r.text, r.text) is None
    assert rg.region_shape_error(r.text, "\n".join(r.text.splitlines()[:-2]) + "\n").startswith("the answer does not end")
    assert rg.region_shape_error(r.text, BIG).startswith("the answer does not begin")
    assert rg.region_shape_error(r.text, "\n") == "the answer is empty"


# ------------------------------------------------------------------------------ the pipeline
def _variant(sap_dir, tmp_path, *, before, after, target):
    pr = tmp_path / "PR-1"
    shutil.copytree(sap_dir.parent, pr)
    fn = pr / sap_dir.name / "functions" / "fn-1"
    for name, text in (("source.before.java", before), ("source.after.java", after), ("target.java", target)):
        (fn / name).write_text(text)
    return pr / sap_dir.name


class Scripted(StubBackend):
    """Answers each request with fn(request); counts calls and remembers the budgets it was given."""

    def __init__(self, fn, *, limits=None, usage=None, done_reason=None, raises=None):
        super().__init__("")
        self.fn, self.calls, self.budgets = fn, [], []
        self.usage_, self.done_, self.raises = usage, done_reason, raises
        if limits is not None:
            self.limits = limits
            self.accepts_budget = True
            self.params = {"temperature": 0, "num_ctx": limits.num_ctx}

    def generate(self, request, **budget):
        self.calls.append(request); self.budgets.append(budget)
        self.last_usage = dict(self.usage_) if self.usage_ else None
        self.last_done_reason = self.done_
        if self.raises:
            raise self.raises
        return self.fn(request)


def _adapt(request):
    text = request.region["text"].replace(
        "import org.example.pkg12.Type12;\n", "import org.example.added.Added;\nimport org.example.pkg12.Type12;\n")
    return "```java\n" + text + "```"


def test_whole_file_unit_is_sent_as_a_region_and_comes_back_as_the_whole_file(sap_dir, pr_manifest, tmp_path):
    sap = _variant(sap_dir, tmp_path, before=HUNK_BEFORE, after=HUNK_AFTER, target=BIG)
    backend = Scripted(_adapt)
    res = run_case(sap, pr_manifest, backend)
    ur = res.units[0]
    # the unit is still a whole-file unit with its whole-file request; what is SENT is the region
    assert ur.request.template_id == WHOLE_TEMPLATE_ID and ur.request.answer_kind == FILE
    assert len(ur.region_requests) == 1 and len(backend.calls) == 1
    rq = backend.calls[0]
    assert rq.template_id == REGION_TEMPLATE_ID and rq.answer_kind == REGION
    assert "## Part of the target file in the fork (adapt this; return all of it)" in rq.prompt
    assert "method20" not in rq.prompt and "Return ONLY the complete adapted part" in rq.prompt
    assert len(rq.prompt) < len(ur.request.prompt) / 2
    assert rq.expected_answer_chars == len(rq.region["text"]) + len("import org.example.added.Added;\n")
    assert untraceable_lines(rq.prompt, ur.context) == []
    target = ur.context.transformation["target"]
    assert target[rq.region["start"]:rq.region["end"]] == rq.region["text"]
    g = res.generation
    assert g.outcome == "completion" and g.capacity is None
    assert g.candidate == target.replace(
        "import org.example.pkg12.Type12;\n", "import org.example.added.Added;\nimport org.example.pkg12.Type12;\n")
    assert g.regions[0]["outcome"] == "completion" and g.regions[0]["lines"] == rq.region["lines"]
    assert g.request_hash != ur.request.request_hash
    # deterministic, and switched off it is the whole-file request again
    again = run_case(sap, pr_manifest, Scripted(_adapt))
    assert again.generation.request_hash == g.request_hash
    off = run_case(sap, pr_manifest, StubBackend("```java\n" + BIG + "```"), region_requests=False)
    assert off.units[0].region_requests == [] and off.request.prompt == ur.request.prompt
    assert off.generation.outcome == "completion"


def test_region_answer_that_is_not_the_part_is_refused_and_indentation_is_kept(sap_dir, pr_manifest, tmp_path):
    sap = _variant(sap_dir, tmp_path, before=HUNK2_BEFORE, after=HUNK2_AFTER, target=BIG)
    whole = run_case(sap, pr_manifest, Scripted(lambda r: "```java\n" + BIG + "```"))
    assert whole.generation.outcome == "shape_mismatch"
    assert "region 1 of 1: the answer is not the adapted part" in whole.generation.diagnostics[0]
    # an answer whose first line is indented (the block strips nothing but blank lines)
    def indented(r):
        assert r.region["text"].startswith(" ") or r.region["text"].startswith("import")
        return "Here it is:\n```java\n\n" + r.region["text"].replace(
            "getLogger(Big.class);\n", "getLogger(Big.class);\n    public static final long TIMEOUT_MS = 90_000L;\n") + "\n```"
    res = run_case(sap, pr_manifest, Scripted(indented))
    assert res.generation.outcome == "completion"
    assert "    public static final long TIMEOUT_MS = 90_000L;\n" in res.generation.candidate
    assert res.generation.candidate.count("\n") == res.context.transformation["target"].count("\n") + 1
    # a spliced file that no longer parses is not a completion
    broken = run_case(sap, pr_manifest, Scripted(lambda r: "```java\n" + r.region["text"].replace(
        "getLogger(Big.class);\n", "getLogger(Big.class);\n    public static {{{ long x;\n") + "```"))
    assert broken.generation.outcome == "unparseable"


def test_unlocated_change_falls_back_to_the_whole_file_request(sap_dir, pr_manifest, tmp_path):
    sap = _variant(sap_dir, tmp_path, before="@@ -1,1 +1,2 @@\nimport nowhere.to.be.Found;\n",
                   after="@@ -1,1 +1,2 @@\nimport nowhere.to.be.Found;\nimport another.New;\n", target=BIG)
    backend = Scripted(lambda r: "```java\n" + BIG + "```")
    res = run_case(sap, pr_manifest, backend)
    assert res.units[0].region_requests == [] and not res.units[0].region_plan.located
    assert backend.calls[0].template_id == WHOLE_TEMPLATE_ID and res.generation.outcome == "completion"
    assert synthesize_regions(res.context)[0] == []


def test_function_unit_requests_are_byte_identical_with_and_without_regions(sap_dir, pr_manifest):
    on = run_case(sap_dir, pr_manifest, StubBackend("```java\n" + F_T + "```"))
    off = run_case(sap_dir, pr_manifest, StubBackend("```java\n" + F_T + "```"), region_requests=False)
    assert on.units[0].region_requests == [] and on.request.template_id == "rcap-request-v1"
    assert on.request.prompt == off.request.prompt and on.request.request_hash == off.request.request_hash
    # the golden hashes of test_round2.test_default_request_is_byte_identical
    assert hashlib.sha256(on.request.prompt.encode()).hexdigest() == \
        "4f69a08ef61176bc86789620a7f301a6284e2a85f61576764fc77a6f2c2001c1"
    assert on.request.request_hash == "8927510de0a4a309e7cb607f6e35e954a00bdd3d40b93b82b2918897e4590efb"
    assert synthesize(on.context).model_dump() == on.request.model_dump()
    assert on.generation.limits == {} and on.generation.capacity is None      # a backend that declares no limits


# ------------------------------------------------------------------------------ limits
def test_limits_plan_is_explicit_and_scales_with_the_expected_answer():
    lim = GenerationLimits(num_ctx=16384)
    p = lim.plan(48000, 9600, "java")
    assert p["expected_prompt_tokens"] == 10000 and p["expected_output_tokens"] == 2000 and p["fits_window"]
    assert p["output_budget_tokens"] == 3000 and p["request_limit_chars"] == 58982
    assert p["timeout_s"] == 180 + 10000 / 100 + 3000 / 8
    big = lim.plan(48000, 48000, "java")
    assert big["fits_window"] is False                               # 10000 + 10000 > 16384
    assert lim.plan(2000, 400, "java")["output_budget_tokens"] == 512        # the floor
    assert lim.plan(2000, 400, "scala")["chars_per_token_expected"] == 4.4
    assert lim.plan(2000, None, "java")["output_budget_tokens"] == 16384 - 417   # answer size unknown: the rest of the window
    slow = GenerationLimits(num_ctx=16384, min_output_tokens_per_second=1.0, timeout_max_s=900)
    assert slow.plan(48000, 9600, "java")["timeout_s"] == 900
    assert GenerationLimits(request_limit_chars=10).request_limit("java") == 10
    assert set(lim.as_dict()) >= {"num_ctx", "output_slack", "min_output_tokens_per_second", "chars_per_token_expected"}


def _whole_request(sap_dir, pr_manifest, tmp_path, target):
    sap = _variant(sap_dir, tmp_path, before="@@ -1,1 +1,2 @@\nimport nowhere.to.be.Found;\n",
                   after="@@ -1,1 +1,2 @@\nimport nowhere.to.be.Found;\nimport another.New;\n", target=target)
    return run_case(sap, pr_manifest, StubBackend("```java\n" + target + "```")).request


def test_request_whose_answer_cannot_fit_is_refused_up_front_as_capacity(sap_dir, pr_manifest, tmp_path):
    req = _whole_request(sap_dir, pr_manifest, tmp_path, BIG)
    assert req.expected_answer_chars == len(BIG) + len("import another.New;\n")
    tight = GenerationLimits(num_ctx=2500)          # prompt ~1560 tokens + answer ~1100: does not fit
    backend = Scripted(lambda r: "```java\n" + BIG + "```", limits=tight)
    g = generate(req, backend)
    assert g.outcome == "limits_exceeded" and g.capacity == "answer_cannot_fit_window" and backend.calls == []
    assert "not generated" in g.diagnostics[0] and g.limits["fits_window"] is False
    assert g.limits["expected_output_tokens"] == -(-req.expected_answer_chars * 10 // 48)
    # the request alone too large: the older check, now labelled
    g = generate(req, Scripted(lambda r: "", limits=GenerationLimits(num_ctx=1000)))
    assert (g.outcome, g.capacity) == ("limits_exceeded", "request_too_large")
    # it fits: generated with a budget and a timeout derived from the expected answer, both recorded
    roomy = Scripted(lambda r: "```java\n" + BIG + "```", limits=GenerationLimits(num_ctx=16384),
                     usage={"input_tokens": 1500, "output_tokens": 1100})
    g = generate(req, roomy)
    assert g.outcome == "completion" and g.capacity is None
    assert roomy.budgets == [{"num_predict": g.limits["output_budget_tokens"], "timeout": g.limits["timeout_s"]}]
    assert g.limits["output_budget_tokens"] == -(-g.limits["expected_output_tokens"] * 3 // 2)
    assert g.limits["request_chars"] == len(req.prompt) and g.usage == {"input_tokens": 1500, "output_tokens": 1100}


@pytest.mark.parametrize("kwargs,capacity", [
    (dict(raises=TimeoutError("generation timed out after 500s")), "generation_timeout"),
    (dict(done_reason="length", usage={"input_tokens": 1500, "output_tokens": 1650}), "output_budget_exhausted"),
    (dict(usage={"input_tokens": 9000, "output_tokens": 8000}), "window_overflow"),
    (dict(usage={"input_tokens": 8194, "output_tokens": 50}), "input_truncated"),
])
def test_capacity_refusals_are_labelled_and_distinct_from_model_errors(sap_dir, pr_manifest, tmp_path, kwargs, capacity):
    req = _whole_request(sap_dir, pr_manifest, tmp_path, BIG)
    g = generate(req, Scripted(lambda r: "```java\n" + BIG + "```", limits=GenerationLimits(num_ctx=16384), **kwargs))
    assert g.outcome == "limits_exceeded" and g.capacity == capacity and capacity in CAPACITY_KINDS
    assert g.limits["output_budget_tokens"] and g.limits["timeout_s"]


def test_model_errors_are_not_capacity(sap_dir, pr_manifest, tmp_path):
    req = _whole_request(sap_dir, pr_manifest, tmp_path, BIG)
    lim = GenerationLimits(num_ctx=16384)
    assert generate(req, Scripted(lambda r: "", limits=lim, raises=RuntimeError("connection refused"))).outcome == "backend_error"
    for answer, outcome in (("```java\nclass {{{\n```", "unparseable"), ("", "no_output"),
                            ("```java\nvoid f() { }\n```", "shape_mismatch")):
        g = generate(req, Scripted(lambda r, a=answer: a, limits=lim))
        assert (g.outcome, g.capacity) == (outcome, None) and g.limits["num_ctx"] == 16384


def test_a_capacity_refusal_of_one_region_is_the_units(sap_dir, pr_manifest, tmp_path):
    sap = _variant(sap_dir, tmp_path, before=HUNK_BEFORE + HUNK2_BEFORE, after=HUNK_AFTER + HUNK2_AFTER, target=BIG)
    res = run_case(sap, pr_manifest, StubBackend("```java\n```"))
    ur = res.units[0]
    assert len(ur.region_requests) == 2
    backend = Scripted(_adapt, limits=GenerationLimits(num_ctx=16384), raises=TimeoutError("generation timed out after 250s"))
    g, _ = generate_regions(ur.request, ur.region_requests, ur.context.transformation["target"], backend)
    assert (g.outcome, g.capacity) == ("limits_exceeded", "generation_timeout")
    assert len(backend.calls) == 1 and len(g.regions) == 1          # the second region is not generated
    assert g.diagnostics[0].startswith("region 1 of 2:") and len(g.limits["regions"]) == 1


def test_a_region_never_begins_or_ends_inside_a_block_comment():
    doc = "".join(f"     * javadoc line {n} of a long comment\n" for n in range(1, 15))
    target = BIG.replace("    private static final Logger log", "    /**\n" + doc + "     */\n    private static final Logger log")
    r = rg.plan_regions(HUNK2_BEFORE, HUNK2_AFTER, target).regions[0]
    assert r.text.splitlines()[0].strip() == "/**"            # 6 lines of context would have begun at `* javadoc line 10`
    tail = BIG.replace("    public int method1(int value) {", "    /*\n" + doc + "     */\n    public int method1(int value) {")
    r = rg.plan_regions(HUNK2_BEFORE, HUNK2_AFTER, tail).regions[0]
    assert r.text.splitlines()[-1].strip() == "*/"
    assert rg._comment_state(["a /* b */ c", "// not /* open", "x /* open", "still", "done */ y"]) == [False, False, False, True, True, False]
