"""Merging overlapping / nested units into one file (rcap.splice)."""

from rcap.splice import CONFLICT, MERGED, UNSPLICEABLE, UnitEdit, merge_units

FILE = """package p;

class A {
    int a() {
        return 1;
    }

    int b() {
        return 2;
    }

    int c() {
        return 3;
    }
}
"""
CLASS = FILE[FILE.index("class A"):]
A = "    int a() {\n        return 1;\n    }\n"
B = "    int b() {\n        return 2;\n    }\n"


def test_sequential_text_search_loses_the_inner_unit_and_merge_does_not():
    outer = UnitEdit("outer", CLASS, CLASS.replace("return 3", "return 30"))
    inner = UnitEdit("inner", B, B.replace("return 2", "return 20"))
    # the old splice: after the outer replacement the inner original is still there here,
    # but an outer candidate that touches b() removes it; the merge is order-free either way
    r = merge_units(FILE, [outer, inner])
    assert "return 30" in r.text and "return 20" in r.text and "return 1;" in r.text
    assert {k: u.status for k, u in r.units.items()} == {"outer": MERGED, "inner": MERGED}
    assert merge_units(FILE, [inner, outer]).text == r.text


def test_identical_hunk_of_outer_and_inner_counts_once():
    outer = UnitEdit("outer", CLASS, CLASS.replace("return 2", "return 20").replace("return 3", "return 30"))
    inner = UnitEdit("inner", B, B.replace("return 2", "return 20"))
    r = merge_units(FILE, [outer, inner])
    assert r.text == FILE.replace("return 2", "return 20").replace("return 3", "return 30")
    assert r.units["outer"].status == MERGED and r.units["outer"].shared == 1
    assert r.units["inner"].status == MERGED and r.units["inner"].applied == 1
    assert len(r.applied) == 2 and sorted(r.applied[0].owners) == ["inner", "outer"]


def test_true_conflict_prefers_the_inner_unit_and_records_it():
    outer = UnitEdit("outer", CLASS, CLASS.replace("return 2", "return 99").replace("return 3", "return 30"))
    inner = UnitEdit("inner", B, B.replace("return 2", "return 20"))
    r = merge_units(FILE, [outer, inner])
    assert "return 20" in r.text and "return 99" not in r.text and "return 30" in r.text
    o = r.units["outer"]
    assert o.status == CONFLICT and o.hunks == 2 and o.applied == 1
    assert o.lost[0]["lost_to"] == ["inner"] and "return 99" in o.lost[0]["added"]
    assert r.units["inner"].status == MERGED and r.units["inner"].won_over == ["outer"]
    assert "inner" in o.reason


def test_whole_file_unit_next_to_its_functions():
    whole = UnitEdit("file", FILE, FILE.replace("package p;", "package p;\n\nimport q.Z;"))
    fa = UnitEdit("a", A, A.replace("return 1", "return 10"))
    fb = UnitEdit("b", B, B.replace("return 2", "return 20"))
    r = merge_units(FILE, [whole, fa, fb])
    assert all(u.status == MERGED for u in r.units.values())
    assert "import q.Z;" in r.text and "return 10" in r.text and "return 20" in r.text


def test_unspliceable_is_recorded_with_its_reason_and_the_rest_is_applied():
    gone = UnitEdit("gone", "    int zz() {}\n", "    int zz() { return 0; }\n")
    twice = UnitEdit("twice", "        return", "        return  ")
    fa = UnitEdit("a", A, A.replace("return 1", "return 10"))
    r = merge_units(FILE, [gone, twice, fa])
    assert r.units["gone"].status == UNSPLICEABLE and "found 0 times" in r.units["gone"].reason
    assert r.units["twice"].status == UNSPLICEABLE and "found 3 times" in r.units["twice"].reason
    assert r.units["a"].status == MERGED and "return 10" in r.text


def test_echo_is_merged_with_nothing_to_apply_and_single_unit_equals_plain_replace():
    echo = UnitEdit("e", A, A)
    r = merge_units(FILE, [echo])
    assert r.text == FILE and not r.changed and r.units["e"].hunks == 0
    assert "nothing to apply" in r.units["e"].reason
    cand = A.replace("return 1;", "int x = 1;\n        return x;")
    assert merge_units(FILE, [UnitEdit("a", A, cand)]).text == FILE.replace(A, cand, 1)


def test_candidate_trailing_newline_is_not_a_hunk():
    orig = A.rstrip("\n")                      # a unit whose original stops before the newline
    r = merge_units(FILE, [UnitEdit("a", orig, orig.replace("return 1", "return 10") + "\n")])
    assert r.units["a"].hunks == 1 and r.text == FILE.replace("return 1", "return 10")


def test_same_span_units_tie_goes_to_the_first_and_insertions_at_one_point_conflict():
    one = UnitEdit("one", B, B.replace("return 2", "return 21"))
    two = UnitEdit("two", B, B.replace("return 2", "return 22"))
    r = merge_units(FILE, [one, two])
    assert "return 21" in r.text and r.units["two"].status == CONFLICT
    ins_outer = UnitEdit("outer", CLASS, CLASS.replace("    int b() {", "    int x;\n    int b() {"))
    ins_inner = UnitEdit("inner", B + "\n", "    int y;\n" + B + "\n")
    r = merge_units(FILE, [ins_outer, ins_inner])
    assert "int y;" in r.text and "int x;" not in r.text and r.units["outer"].status == CONFLICT


def test_spans_in_the_merged_text_follow_the_edits():
    fa = UnitEdit("a", A, A.replace("return 1;", "int x = 1;\n        return x;"))
    fb = UnitEdit("b", B, B.replace("return 2", "return 20"))
    r = merge_units(FILE, [fa, fb])
    lines = r.text.splitlines()
    first, last = r.lines_of("b")
    assert lines[first - 1].strip() == "int b() {" and lines[last - 1].strip() == "}"
    first, last = r.lines_of("a")
    assert lines[first - 1].strip() == "int a() {" and last - first == 3
    hl = r.hunk_lines()
    assert [o for _, _, o in hl] == [["a"], ["b"]] and lines[hl[1][0] - 1].strip() == "return 20;"
