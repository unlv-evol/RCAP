"""Dependency-ordered application of a PR's change groups (rcap.prorder)."""

from rcap import cochange as cc
from rcap.prorder import (
    Group,
    Rename,
    added_lines,
    build_graph,
    brace_depth,
    find_renames,
    identifiers,
    introduced,
    is_test_of,
    layers,
    revert_rename,
)


def g(gid, file, provides=(), uses=(), kind="unit"):
    return Group(gid, kind, file, set(provides), set(uses), is_test="/src/test/" in file)


def test_added_lines_identifiers_and_introduced_names():
    orig = "int f() {\n  return a();\n}\n"
    cand = "int f() {\n  // uses Old\n  return helper(a(), \"Str\");\n}\n"
    assert added_lines(orig, cand) == ["  // uses Old", "  return helper(a(), \"Str\");"]
    assert identifiers(added_lines(orig, cand)) == {"return", "helper", "a"}
    added = ["  public int logConfig() {", "  private final Map<String, String> cfg;"]
    assert introduced(added, [], "class A { int other; }") == {"logConfig", "cfg"}
    assert introduced(added, ["  public int logConfig(int x) {"], None) == {"cfg"}
    assert introduced(added, [], "class A { int cfg; }") == {"logConfig"}        # already a word of the file


def test_user_depends_on_the_group_that_introduces_the_name():
    decl = g("mech:Buffer@@1", "s/src/main/java/a/Buffer.java", provides={"logConfig"}, kind="mech")
    user = g("u:KTable", "s/src/main/java/a/KTable.java", uses={"logConfig", "size"})
    other = g("u:Other", "s/src/main/java/a/Other.java", uses={"size"})
    graph = build_graph([user, other, decl])
    assert graph.hard == {"u:KTable": {"mech:Buffer@@1": ["logConfig"]}}
    assert layers([user, other, decl], graph) == [["u:Other", "mech:Buffer@@1"], ["u:KTable"]]
    assert graph.hard_dependents("mech:Buffer@@1") == {"u:KTable"}
    assert graph.hard_providers("u:KTable") == {"mech:Buffer@@1"}
    # a name the fork already has elsewhere is no evidence of a dependency
    assert build_graph([user, decl], in_fork=lambda n: n == "logConfig").hard == {}


def test_json_schema_field_provides_its_generated_accessors():
    schema = g("mech:MetadataResponse.json@@2", "c/src/main/resources/common/message/MetadataResponse.json",
               provides={"ErrorCode"}, kind="mech")
    user = g("u:MetadataRequest", "c/src/main/java/r/MetadataRequest.java", uses={"setErrorCode", "responseData"})
    graph = build_graph([user, schema])
    assert graph.hard["u:MetadataRequest"] == {"mech:MetadataResponse.json@@2": ["setErrorCode"]}


def test_a_test_is_a_soft_dependent_of_its_production_file():
    assert is_test_of("c/src/test/java/a/FooTest.java", "c/src/main/java/a/Foo.java")
    assert is_test_of("c/src/test/scala/a/FooIntegrationTest.scala", "c/src/main/scala/a/Foo.scala")
    assert not is_test_of("c/src/test/java/a/FooTest.java", "c/src/main/java/a/FooBar.java")
    assert not is_test_of("c/src/test/java/a/FooTest.java", "c/src/test/java/a/Foo.java")
    prod, test = g("u:Foo", "c/src/main/java/a/Foo.java"), g("u:FooTest", "c/src/test/java/a/FooTest.java")
    graph = build_graph([test, prod])
    assert graph.soft == {"u:FooTest": ["u:Foo"]} and graph.hard == {}
    assert layers([test, prod], graph) == [["u:Foo"], ["u:FooTest"]]
    assert graph.hard_dependents("u:Foo") == set()         # soft: the test is still tried


def test_transitive_chain_and_cycle_share_layers_deterministically():
    a = g("a", "x/src/main/java/A.java", provides={"pa"})
    b = g("b", "x/src/main/java/B.java", provides={"pb"}, uses={"pa"})
    c = g("c", "x/src/main/java/C.java", uses={"pb"})
    graph = build_graph([c, b, a])
    assert layers([c, b, a], graph) == [["a"], ["b"], ["c"]]
    assert graph.hard_dependents("a") == {"b", "c"} and graph.hard_providers("c") == {"a", "b"}
    x = g("x", "x/src/main/java/X.java", provides={"px"}, uses={"py"})
    y = g("y", "x/src/main/java/Y.java", provides={"py"}, uses={"px"})
    z = g("z", "x/src/main/java/Z.java", uses={"px"})
    assert layers([z, x, y], build_graph([z, x, y])) == [["x", "y"], ["z"]]


DIFF = """diff --git a/s/src/main/java/a/Buffer.java b/s/src/main/java/a/Buffer.java
--- a/s/src/main/java/a/Buffer.java
+++ b/s/src/main/java/a/Buffer.java
@@ -10 +10 @@ class Buffer
-    public Map<String, String> logConfig() {
+    public Map<String, String> topicConfig() {
@@ -30 +30,2 @@ class Buffer
-        return 1;
+        int extra = 2;
+        return extra;
diff --git a/s/src/main/java/a/KTable.java b/s/src/main/java/a/KTable.java
--- a/s/src/main/java/a/KTable.java
+++ b/s/src/main/java/a/KTable.java
@@ -5 +5 @@ class KTable
-        builder.with(buffer.logConfig());
+        builder.with(buffer.topicConfig());
@@ -9 +9 @@ class KTable
-        foo(1);
+        foo(2);
"""


def test_rename_is_found_with_its_declaration_and_every_use():
    files = cc.parse_unified_diff(DIFF)
    renames = find_renames(files)
    assert len(renames) == 1
    r = renames[0]
    assert (r.old, r.new) == ("logConfig", "topicConfig") and r.label == "rename logConfig -> topicConfig"
    assert r.decl_hunk == "s/src/main/java/a/Buffer.java@@1" and r.use_hunks == ["s/src/main/java/a/KTable.java@@1"]
    # an added local variable is a declaration, but nothing was renamed into it
    assert all(x.new != "extra" for x in renames)


def test_revert_rename_keeps_the_forks_name_and_reports_it():
    r = Rename("logConfig", "topicConfig", "h1")
    orig = "void f() {\n  use(buffer.logConfig());\n}\n"
    cand = "void f() {\n  use(buffer.topicConfig());\n  more();\n}\n"
    text, done = revert_rename(cand, orig, [r])
    assert text == "void f() {\n  use(buffer.logConfig());\n  more();\n}\n" and done == [r.label]
    pure = "void f() {\n  use(buffer.topicConfig());\n}\n"
    assert revert_rename(pure, orig, [r])[0] == orig                  # a pure rename unit: nothing left to port
    assert revert_rename(orig, orig, [r]) == (orig, [])
    # the fork already uses the new name in this unit: not ours to revert
    both = "void f() {\n  topicConfig();\n}\n"
    assert revert_rename(both, both, [r]) == (both, [])


def test_local_variable_renames_are_not_pr_level_renames():
    before = "class A {\n    void f() {\n        String s = key();\n        use(s);\n    }\n    int logConfig() { return 1; }\n}\n"
    assert brace_depth(before, 3) == 2 and brace_depth(before, 6) == 1 and brace_depth(before, 1) == 0
    diff = ("diff --git a/x/A.java b/x/A.java\n--- a/x/A.java\n+++ b/x/A.java\n"
            "@@ -3,2 +3,2 @@\n-        String s = key();\n-        use(s);\n+        String trimmed = key();\n+        use(trimmed);\n"
            "@@ -6 +6 @@\n-    int logConfig() { return 1; }\n+    int topicConfig() { return 1; }\n")
    files = cc.parse_unified_diff(diff)
    assert {(r.old, r.new) for r in find_renames(files)} == {("s", "trimmed"), ("logConfig", "topicConfig")}
    assert [(r.old, r.new) for r in find_renames(files, old_text=lambda p: before)] == [("logConfig", "topicConfig")]
