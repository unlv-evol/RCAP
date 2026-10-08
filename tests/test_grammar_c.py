"""C support (rcap.grammars): the minimum the Vim -> Neovim benchmark needs."""
from rcap import accept, fidelity as fd
from rcap.generate import StubBackend, generate
from rcap.grammars import C, function_name_node, grammar_for, parse, tolerates_parse_errors
from rcap.reduction_program import PLACEHOLDER, _reduce_one, recover
from rcap.request import AdaptationRequest
from rcap.scoring import score_unit
from rcap.unitkind import FUNCTION, find_methods, function_name, shape_mismatch, text_kind

FORK = """\
static int find_it(buf_T *buf, int n)
{
  int i;
#ifdef FEAT_X
  if (n < 0 &&
#else
  if (
#endif
      buf == NULL) {
    return FAIL;
  }
  for (i = 0; i < n; i++) {
    if (buf->b_p_ro) {
      emsg(_(e_readonly));
      return FAIL;
    }
  }
  if (n > 3) {
    n = 3;
    msg_start();
    msg_end();
  }
  return OK;
}
"""
CLEAN = FORK.replace("#ifdef FEAT_X\n  if (n < 0 &&\n#else\n  if (\n#endif\n      buf == NULL) {",
                     "  if (buf == NULL) {")
BEFORE = """\
    static int
find_it(buf_T *buf, int n)
{
    int i;
    for (i = 0; i < n; i++)
    {
	if (buf->b_p_ro)
	{
	    emsg(_(e_readonly));
	    return FAIL;
	}
    }
    if (n > 3)
    {
	n = 3;
	msg_start();
	msg_end();
    }
    return OK;
}
"""
AFTER = BEFORE.replace("    return OK;", "    buf->b_changed = TRUE;\n    return OK;")
PORT = CLEAN.replace("  return OK;", "  buf->b_changed = true;\n  return OK;")


def _request(lang="c"):
    return AdaptationRequest(case_id="t", language=lang, prompt="p", request_hash="h")


def test_grammar_lookup_keeps_salp_and_adds_c():
    assert grammar_for("java").name == "Java" and grammar_for(".scala").name == "Scala"
    assert grammar_for("c") is C and grammar_for(".h") is C
    assert grammar_for("py") is None
    assert tolerates_parse_errors("c") and not tolerates_parse_errors("java")


def test_c_function_is_a_function_and_has_a_name():
    assert text_kind(CLEAN, "c") == FUNCTION
    assert function_name(CLEAN, "c") == ("find_it", 2)
    assert find_methods(CLEAN + "\nint other(void)\n{\n  return 1;\n}\n", "other", "c")[0][2] == 1
    assert shape_mismatch(CLEAN, PORT, "c") is None
    node = parse("char *f(int a) { return 0; }", C).root_node.children[0]
    assert function_name_node(node).text == b"f"


def test_java_function_name_is_unchanged():
    assert function_name("void run(int a, int b) { go(); }", "java") == ("run", 2)


def test_parse_error_of_the_target_is_not_held_against_a_c_answer():
    """A condition split over `#ifdef` branches is an ERROR node for tree-sitter and valid C."""
    assert not parse(CLEAN, C).root_node.has_error
    assert parse(FORK, C).root_node.has_error
    answer = "```c\n" + FORK.replace("return OK;", "return n;") + "```"
    refused = generate(_request(), StubBackend(answer))
    assert refused.outcome == "unparseable"
    kept = generate(_request(), StubBackend(answer), parse_reference=FORK)
    assert kept.outcome == "completion"
    # a clean reference still makes a broken answer unparseable
    broken = generate(_request(), StubBackend("```c\n" + CLEAN.replace("return OK;", "return (;") + "```"),
                      parse_reference=CLEAN)
    assert broken.outcome == "unparseable"
    # and Java is not excused by a reference
    java = generate(_request("java"), StubBackend("```java\nvoid f( {\n```"), parse_reference="void g( {")
    assert java.outcome == "unparseable"


def test_program_cut_in_c_cuts_statements_and_recovers():
    protected = {n for n, ln in enumerate(BEFORE.splitlines()) if "return OK" in ln}
    art = _reduce_one("find_it", "source.before", BEFORE, protected, C, 3, None, "c")
    assert len(art.placeholders) == 2 and art.nodes_after < art.nodes_before
    assert PLACEHOLDER.format(n="source.before.1") in art.reduced_text
    assert recover(art) == BEFORE


def test_program_cut_in_c_never_cuts_a_subtree_with_a_parse_error():
    text = CLEAN.replace("emsg(_(e_readonly));", "emsg(_(e_readonly) FOO BAR;")
    art = _reduce_one("find_it", "target", text, set(), C, 3, None, "c")
    assert all("FOO BAR" not in p.original_text for p in art.placeholders)
    assert recover(art) == text


def test_duplicated_block_is_found_in_c():
    block = "  if (n > 3) {\n    n = 3;\n    msg_start();\n    msg_end();\n  }\n"
    doubled = CLEAN.replace(block, block + block)
    dup = accept.duplicated_blocks(CLEAN, doubled, BEFORE, AFTER, language="c")
    assert dup and dup[0]["copies_candidate"] == 2
    assert accept.duplicated_blocks(CLEAN, PORT, BEFORE, AFTER, language="c") == []
    assert accept.rejection(doubled, CLEAN, BEFORE, AFTER, language="c")[0] == "duplicated_block"


def test_labels_on_c():
    assert score_unit(CLEAN, PORT, BEFORE, AFTER, language="c").label in (fd.FAITHFUL_CODE, fd.PARTIAL)
    assert score_unit(CLEAN, CLEAN, BEFORE, AFTER, language="c").label == fd.NO_CHANGE
    leak = PORT.replace("  int i;", "  int i;\n  /* RCAP_PH_target.1 */")
    assert score_unit(CLEAN, leak, BEFORE, AFTER, language="c").label == fd.PLACEHOLDER_LEAK
    assert score_unit(CLEAN, CLEAN, BEFORE, AFTER, language="c", earlier=[PORT]).label == "REVERTED"
