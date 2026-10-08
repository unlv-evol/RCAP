"""The Vim -> Neovim evaluation: PPatHF's metrics and the gcc feedback (tools/vimnvim)."""
import sys
from pathlib import Path

import pytest

pytest.importorskip("tree_sitter_c")
pytest.importorskip("editdistance")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "vimnvim"))

from rcap import ppathf_metrics as pm  # noqa: E402

PRE = "int f(int a)\n{\n  // old\n  return a;\n}\n"
DEV = "int f(int a)\n{\n  if (a < 0) {\n    return 0;\n  }\n  return a;\n}\n"


def test_tokens_ignore_layout_and_comments():
    assert pm.cleaned_tokens(PRE) == ["int", "f", "(", "int", "a", ")", "{", "return", "a", ";", "}"]
    assert pm.cleaned_tokens("int f(int a) { return a; } /* x */") == pm.cleaned_tokens(PRE)


def test_origin_scores_red_one_and_exact_scores_zero():
    origin = pm.score(PRE, DEV, PRE)
    assert origin["rel_distance"] == 1.0 and not origin["exact_match"] and origin["echo"]
    assert origin["gen_distance"] == origin["src_distance"] == 11
    same = pm.score(DEV.replace("  if", "\n\n      if"), DEV, PRE)
    assert same["exact_match"] and same["rel_distance"] == 0 and same["gen_distance"] == 0


def test_red_is_relative_to_doing_nothing():
    worse = pm.score(PRE.replace("return a;", "x(); y(); z(); q(); r(); s(); return a;"), DEV, PRE)
    assert worse["rel_distance"] > 1
    closer = pm.score(DEV.replace("return 0;", "return 1;"), DEV, PRE)
    assert 0 < closer["rel_distance"] < 0.2 and pm.red_band(closer["rel_distance"]) == "0 < RED < 0.2"


def test_comment_only_port_divides_by_one():
    """src_distance == 0: the package divides by 1 (metrics.py:82)."""
    dev = PRE.replace("// old", "// new")
    assert pm.score(PRE, dev, PRE) == {"exact_match": True, "gen_distance": 0, "src_distance": 0,
                                       "rel_distance": 0.0, "src_zero": True, "echo": True}
    assert pm.score(DEV, dev, PRE)["rel_distance"] == 11.0


def test_incomplete_or_missing_output_is_scored_as_the_pre_patch_function():
    cut = DEV[: DEV.index("return a;")]
    assert not pm.is_complete(cut) and pm.is_complete(DEV) and not pm.is_complete(None)
    for gen in (None, cut):
        row = pm.score_aligned(gen, DEV, PRE)
        assert row["use_dummy"] and row["rel_distance"] == 1.0 and not row["exact_match"]
    assert not pm.score_aligned(DEV, DEV, PRE)["use_dummy"]


def test_aggregate_and_bands():
    rows = [pm.score(DEV, DEV, PRE), pm.score(PRE, DEV, PRE)]
    agg = pm.aggregate(rows)
    assert (agg["n"], agg["exact"], agg["accuracy"], agg["aed"], agg["red"]) == (2, 1, 0.5, 5.5, 0.5)
    assert [pm.red_band(x) for x in (0, 0.1, 0.2, 0.99, 1, 7)] == [
        "RED = 0", "0 < RED < 0.2", "0.2 <= RED < 1", "0.2 <= RED < 1", "RED >= 1", "RED >= 1"]
    assert pm.aggregate([])["accuracy"] is None


def test_paired_tests():
    m = pm.mcnemar_exact([True, False, False, False, True], [True, True, True, True, False])
    assert (m["gain"], m["loss"]) == (3, 1) and m["p"] == pytest.approx(0.625)
    assert pm.mcnemar_exact([True], [True])["p"] == 1.0
    s = pm.sign_test([1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 1])
    assert (s["lower"], s["higher"], s["ties"]) == (5, 0, 1) and s["p"] == pytest.approx(0.0625)
    assert pm.wilcoxon([1.0, 2.0], [1.0, 2.0])["p"] is None


GCC = """\
/src/src/nvim/menu.c: In function 'execute_menu':
/src/src/nvim/menu.c:601:5: error: implicit declaration of function 'msg_attr'; did you mean 'msg_ext'? [-Wimplicit-function-declaration]
  601 |     msg_attr(p, attr);
      |     ^~~~~~~~
      |     msg_ext
/src/src/nvim/menu.c:604:13: warning: unused variable 'x' [-Wunused-variable]
  604 |         int x;
      |             ^
/src/src/nvim/menu.c:610:9: error: 'curbuf_T' undeclared (first use in this function); did you mean 'curbuf'?
  610 |         curbuf_T = 1;
      |         ^~~~~~~~
/src/src/nvim/menu.c:610:9: note: each undeclared identifier is reported only once for each function it appears in
/src/src/nvim/menu.c:612:11: error: 'hashtab_T' {aka 'struct hashtable_S'} has no member named 'ht_flags'
  612 |     ht->ht_flags = 0;
      |           ^~
/src/src/nvim/menu.c:615:3: error: too few arguments to function 'emsg_text'
  615 |   emsg_text(a);
      |   ^~~~~~~~~
In file included from /src/src/nvim/menu.c:20:
/src/build/src/nvim/auto/message.h.generated.h:12:6: note: declared here
/src/src/nvim/other.c:40:1: error: conflicting types for 'execute_menu'
"""


def test_gcc_errors_are_parsed_and_warnings_dropped():
    from gccdiag import missing_names, parse_gcc, render

    errors = parse_gcc(GCC)
    assert [e.line for e in errors] == [601, 610, 612, 615, 40]
    assert errors[0].code_line == "msg_attr(p, attr);"
    assert errors[3].notes == ["declared here"]
    miss = missing_names(errors)
    assert [(m.name, m.kind, m.owner) for m in miss] == [
        ("msg_attr", "function", None), ("curbuf_T", "variable", None),
        ("ht_flags", "member", "hashtab_T"), ("emsg_text", "signature", None),
        ("execute_menu", "signature", None)]
    text, stats = render(errors, first_line=590, last_line=620, rel_path="src/nvim/menu.c")
    assert stats == {"errors": 5, "shown": 5, "inside_function": 4, "outside_function": 1}
    assert "line 12 of the function: error: implicit declaration of function 'msg_attr'" in text
    assert "nvim/other.c:40 (outside the function)" in text and "unused variable" not in text
    short, stats = render(errors, first_line=590, last_line=620, rel_path="src/nvim/menu.c", max_errors=2)
    assert stats["shown"] == 2 and "the first 2 are shown" in short


def test_unit_and_requests_for_a_c_case():
    from common import Unit
    from rcap.reprompt import RepairContext, RepairFeedback, SymbolContext
    from rcap.request import REPAIR_TEMPLATE_ID, REPAIR_V3_TEMPLATE_ID, TEMPLATE_ID, synthesize

    before = ("    static int\nfind_it(buf_T *buf, int n)\n{\n    int i;\n    for (i = 0; i < n; i++)\n    {\n"
              "\tif (buf->b_p_ro)\n\t{\n\t    emsg(_(e_readonly));\n\t    return FAIL;\n\t}\n    }\n    return OK;\n}\n")
    case = {"func_before_source": before, "func_after_source": before.replace("return OK;", "return n;"),
            "func_before_target": "static int find_it(buf_T *buf, int n)\n{\n  return OK;\n}\n",
            "func_after_target": "static int find_it(buf_T *buf, int n)\n{\n  return n;\n}\n",
            "file_path_target": "src/nvim/x.c", "commit_id_target": "0" * 40}
    full, cut = Unit("c1", case, False), Unit("c1", case, True)
    assert full.entity == "find_it" and not full.expected_placeholders
    assert cut.reduction_stats()["placeholders"] == {"source.before": 1, "source.after": 1, "target": 0}
    r = synthesize(full.context)
    assert r.template_id == TEMPLATE_ID and r.language == "c" and "```c\nstatic int find_it" in r.prompt
    assert "RCAP_PH_source.before.1" in synthesize(cut.context).prompt
    fb = RepairFeedback(attempt=2, kind="build", previous_candidate="int x;\n", error_text="1 compile error(s)")
    v2 = synthesize(RepairContext(**dict(full.context), feedback=fb))
    assert v2.template_id == REPAIR_TEMPLATE_ID and "1 compile error(s)" in v2.prompt
    fb.symbol_context = [SymbolContext(symbol="vim_free", origin="predates_change", upstream_signature="void vim_free(void *x)",
                                       fork_heading="The fork does not declare `vim_free`. Its closest declarations",
                                       fork_members=["void xfree(void *ptr);"])]
    v3 = synthesize(RepairContext(**dict(full.context), feedback=fb))
    assert v3.template_id == REPAIR_V3_TEMPLATE_ID and "void xfree(void *ptr);" in v3.prompt
