"""PPatHF's placeholder scheme (ExecutionConfig.placeholder_scheme = "shared-v1").

Three rules, each against the failure it removes (study 2026-11-placeholder-limit-fix):
1. A block is cut only when it is in source.before, source.after AND the target,
   under ONE id the three share. Under rcap-v1 a model that copied upstream code
   carried `/* RCAP_PH_source.after.1 */` into its answer, an id the fork has not
   (704 RePatch rejections); and a source side that cut nothing let the target
   be cut everywhere (19 Trino functions).
2. The prompt says nothing about placeholders (no-cut prompts made models invent them).
3. The answer is recovered tolerantly: nothing is rejected for its placeholders;
   a lost one is counted, a repeated or invented one is removed.
"""

from __future__ import annotations

import json

import pytest
from conftest import build_fixture_pr

from rcap import runoverrides
from rcap.backend_ollama import GenerationLimits
from rcap.config import EvaluationMode, ExecutionConfig
from rcap.generate import StubBackend, tolerate_placeholders
from rcap.intake import load_case
from rcap.materialize import materialize
from rcap.pipeline import run_case
from rcap.reduction_program import recover, reduce_program
from rcap.reduction_semantic import reduce_semantic
from rcap.selection import select

SHARED = ExecutionConfig(placeholder_scheme="shared-v1")

LOOP = """        for (Entry e : entries) {
            check(e);
            count++;
        }
"""
FORK_ONLY = """        if (forkMode) {
            forkSetup();
            forkMore();
        }
"""
UP_ONLY = """        while (upstreamOnly()) {
            spin();
            spinMore();
        }
"""
BEFORE = "void validate(Config c) {\n" + LOOP + "        apply(c);\n" + UP_ONLY + "    }\n"
AFTER = "void validate(Config c) {\n" + LOOP + "        applyChecked(c);\n" + UP_ONLY + "    }\n"
TARGET = "void validate(Config c) {\n" + LOOP + "        apply(c);\n" + FORK_ONLY + "    }\n"
DIFF = """@@ -1,8 +1,8 @@
 void validate(Config c) {
         for (Entry e : entries) {
             check(e);
             count++;
         }
-        apply(c);
+        applyChecked(c);
"""
ADAPTED_TARGET = TARGET.replace("apply(c);", "applyChecked(c);")


def _build(tmp_path_factory, name, payloads):
    pr = build_fixture_pr(tmp_path_factory.mktemp(name), payloads=payloads)
    return pr / "sap-Validator", json.loads((pr / "pr.json").read_text(encoding="utf-8"))


def _stages(sap, manifest):
    case = load_case(sap, pr_manifest=manifest)
    red = reduce_semantic(case, select(case))
    return case, red, materialize(case, red)


@pytest.fixture(scope="module")
def sap(tmp_path_factory):
    return _build(tmp_path_factory, "shared", (BEFORE, AFTER, TARGET, DIFF))


def _arts(rec):
    return {r: rec.artifact("functions/fn-1", r) for r in ("source.before", "source.after", "target")}


# --- rule 1: the cut ----------------------------------------------------------

def test_a_block_in_all_three_sides_is_cut_under_one_shared_id(sap):
    arts = _arts(reduce_program(*_stages(*sap), SHARED))
    ids = {r: [p.ph_id for p in a.placeholders] for r, a in arts.items()}
    assert ids == {"source.before": ["1"], "source.after": ["1"], "target": ["1"]}
    for a in arts.values():
        assert "/* RCAP_PH_1 */" in a.reduced_text and "check(e);" not in a.reduced_text


def test_a_block_missing_from_one_side_is_never_cut(sap):
    arts = _arts(reduce_program(*_stages(*sap), SHARED))
    assert "forkSetup();" in arts["target"].reduced_text        # fork only
    assert "spin();" in arts["source.before"].reduced_text      # upstream only
    assert "spin();" in arts["source.after"].reduced_text


def test_recovery_round_trips(sap):
    case, red, mat = _stages(*sap)
    for a in reduce_program(case, red, mat, SHARED).artifacts:
        assert recover(a) == mat.by_role(a.entity)[a.role].content


def test_rcap_v1_is_unchanged_by_default(sap):
    """The default scheme keeps per-side ids (11b66e4's cut)."""
    arts = _arts(reduce_program(*_stages(*sap)))
    assert all(p.ph_id.startswith("source.after.") for p in arts["source.after"].placeholders)
    assert all(p.ph_id.startswith("target.") for p in arts["target"].placeholders)


def test_source_that_cuts_nothing_leaves_the_target_whole(tmp_path_factory):
    """rcap-v1 read an empty source filter as "cut everything" on the target."""
    before = "void validate(Config c) {\n        apply(c);\n    }\n"
    after = "void validate(Config c) {\n        applyChecked(c);\n    }\n"
    target = "void validate(Config c) {\n" + FORK_ONLY + "        apply(c);\n    }\n"
    diff = "@@ -1,3 +1,3 @@\n void validate(Config c) {\n-        apply(c);\n+        applyChecked(c);\n"
    s = _build(tmp_path_factory, "nocut", (before, after, target, diff))
    old = _arts(reduce_program(*_stages(*s)))
    new = _arts(reduce_program(*_stages(*s), SHARED))
    assert old["target"].placeholders, "the defect this rule removes"
    assert new["target"].placeholders == [] and "forkSetup();" in new["target"].reduced_text


# --- rule 2: the prompt -------------------------------------------------------

def test_prompt_has_no_placeholder_rule_under_shared(sap):
    shared = run_case(*sap, StubBackend("```java\n" + ADAPTED_TARGET + "```"), SHARED)
    default = run_case(*sap, StubBackend("```java\n" + ADAPTED_TARGET + "```"))
    assert "stand for elided code" in default.request.prompt
    assert "stand for elided code" not in shared.request.prompt
    assert "/* RCAP_PH_1 */" in shared.request.prompt  # the cut itself is still shown


def test_no_cut_mode_prompt_has_no_placeholder_rule(sap):
    cfg = ExecutionConfig(mode=EvaluationMode.SEMANTIC_ONLY, placeholder_scheme="shared-v1")
    res = run_case(*sap, StubBackend("```java\n" + ADAPTED_TARGET + "```"), cfg)
    assert "RCAP_PH" not in res.request.prompt


# --- rule 3: tolerant recovery -------------------------------------------------

def _answer(sap, body):
    return run_case(*sap, StubBackend("```java\n" + body + "```"), SHARED).generation


def test_answer_that_copies_upstream_with_its_placeholder_is_a_completion(sap):
    copied = AFTER.replace(LOOP, "        /* RCAP_PH_1 */\n").replace(UP_ONLY, FORK_ONLY)
    g = _answer(sap, copied)
    assert g.outcome == "completion" and "/* RCAP_PH_1 */" in g.candidate
    assert g.placeholders == {"lost": 0, "removed": 0}


def test_lost_placeholder_is_counted_not_rejected(sap):
    g = _answer(sap, "void validate(Config c) {\n        applyChecked(c);\n" + FORK_ONLY + "    }\n")
    assert g.outcome == "completion"
    assert g.placeholders == {"lost": 1, "removed": 0}


def test_invented_and_repeated_marks_are_removed(sap):
    body = ("void validate(Config c) {\n        /* RCAP_PH_1 */\n        /* RCAP_PH_1 */\n"
            "        /* RCAP_PH_9 */\n        applyChecked(c); // RCAP_PH_x\n" + FORK_ONLY + "    }\n")
    g = _answer(sap, body)
    assert g.outcome == "completion"
    assert g.candidate.count("RCAP_PH") == 1
    assert "applyChecked(c);" in g.candidate
    assert g.placeholders == {"lost": 0, "removed": 3}


def test_tolerate_reads_any_spelling_of_an_expected_id():
    text, st = tolerate_placeholders("a();\n/*RCAP_PH_1*/\nb();\n", {"/* RCAP_PH_1 */"})
    assert text == "a();\n/* RCAP_PH_1 */\nb();\n" and st == {"lost": 0, "removed": 0}


def test_strict_policy_still_rejects_by_default(sap):
    g = run_case(*sap, StubBackend("```java\nvoid validate(Config c) {\n"
                                   "        applyChecked(c);\n    }\n```")).generation
    assert g.outcome == "placeholder_violation"


# --- run switches -------------------------------------------------------------

def test_override_sets_the_scheme_and_tags_the_variant(monkeypatch, tmp_path):
    monkeypatch.setenv("RCAP_OVERRIDES", "none")
    assert runoverrides.load().variant is None
    monkeypatch.setenv("RCAP_PLACEHOLDER_SCHEME", "shared-v1")
    ov = runoverrides.load()
    assert ov.apply(ExecutionConfig()).placeholder_scheme == "shared-v1"
    assert ov.variant.endswith(":ph=shared-v1")
    f = tmp_path / "o.json"
    f.write_text(json.dumps({"name": "x", "edit_evidence": True, "evidence_filter": "names"}))
    monkeypatch.delenv("RCAP_PLACEHOLDER_SCHEME")
    assert runoverrides.load(f).variant == "x:ev=1:allow=0:filter=names"  # earlier tags unchanged


def test_output_floor_gives_the_baselines_budget():
    lim = GenerationLimits(output_floor_tokens=4096)
    assert lim.plan(1000, 480, "java")["output_budget_tokens"] == 4096
    assert GenerationLimits(output_floor_tokens=4096).plan(1000, 48_000, "java")["output_budget_tokens"] == 15000
