"""No-build symbol check of candidate ports (tools/candidates/symbol_check.py, symindex.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "candidates"))

import symbol_check as sc  # noqa: E402
import symindex as si  # noqa: E402

BASE = """package org.fork.core;
import java.util.List;
import static org.fork.util.Checks.requireOpen;
public abstract class Base {
    protected final Logger log = null;
    protected int baseCounter;
    protected void baseHelper(int a) {}
}
"""
TARGET_FILE = """package org.fork.core;
import java.util.List;
import java.util.Map;
public class Worker extends Base implements Runnable {
    private final Map<String, Integer> counts;
    private volatile boolean running;
    enum Mode { FAST, SLOW(2) { void tune() {} }, SAFE; Mode() {} Mode(int x) {} }
    public Worker(Map<String, Integer> counts) { this.counts = counts; }
    public void run() { step(1); }
    void step(int n) { }
    void log(String fmt, Object... args) { }
    private static int compute(int a, int b) { return a + b; }
}
"""
UTIL = """package org.fork.util;
public final class Checks {
    public static void requireOpen(Object o) {}
    public static int retries = 3;
}
"""
OTHER = """package org.fork.other;
public class Elsewhere extends com.lib.Remote {
    private long elsewhereOnly;
    public void farAway(String s) {}
}
"""
LIBSUB = """package org.fork.core;
public class Plugin extends AbstractLibraryPlugin {
    void own() {}
}
"""
TARGET_FN = """public void run() {
    step(1);
}
"""


def fork():
    files = {p: si.file_record(p, t) for p, t in [
        ("core/Base.java", BASE), ("core/Worker.java", TARGET_FILE), ("util/Checks.java", UTIL),
        ("other/Elsewhere.java", OTHER), ("core/Plugin.java", LIBSUB)]}
    jdk = si.NameIndex({"<jdk>": {"types": ["String", "Object", "Runnable", "Math", "Thread", "Duration", "TimeUnit",
                                            "RuntimeException"],
                                  "fields": ["MAX_VALUE", "ZERO", "SECONDS"],
                                  "methods": {"max": [2], "size": [0], "get": [1], "interrupt": [0], "info": [1],
                                              "toMillis": [1], "valueOf": [1], "keySet": [0], "forEach": [1]},
                                  "ctors": {"Thread": [0, 1]}}})
    return sc.ForkContext({"files": files}, jdk), jdk


def run(cand, pr=None, target=TARGET_FN, path="core/Worker.java"):
    f, j = fork()
    return sc.check(cand, target, f, j, pr or {}, path)


def test_names_declared_in_the_fork_file_and_its_supertypes_resolve():
    r = run("public void run() {\n  if (running) { step(compute(1, 2)); baseHelper(baseCounter); }\n"
            "  log.info(\"x\"); counts.size();\n}\n")
    assert r["verdict"] == sc.V_OK, r


def test_an_invented_method_is_unresolved():
    r = run("public void run() {\n  stepWithRetry(1);\n}\n")
    assert r["verdict"] == sc.V_UNRES and r["unresolved"] == ["stepWithRetry:call"]


def test_a_method_the_pr_adds_needs_pr_edits():
    pr = {"types": [], "fields": [], "methods": {"stepWithRetry": [1]}, "ctors": {}}
    r = run("public void run() {\n  stepWithRetry(1);\n}\n", pr=pr)
    assert r["verdict"] == sc.V_PR and r["pr_only"] == ["stepWithRetry:call:method"]


def test_locals_parameters_lambda_and_catch_names_are_not_references():
    r = run("public void run() {\n  int total = 0; for (String key : counts.keySet()) { total++; }\n"
            "  counts.forEach((k, v) -> step(v)); Runnable q = () -> step(total);\n"
            "  try { step(2); } catch (RuntimeException err) { step(err.hashCode()); }\n}\n")
    assert r["verdict"] == sc.V_OK, r
    assert not any(x.split(":")[0] in {"total", "key", "k", "v", "err", "q"} for x in r["unresolved"])


def test_bare_field_elsewhere_in_the_fork_does_not_count_strict():
    r = run("public void run() {\n  elsewhereOnly = 1;\n}\n")
    assert r["verdict"] == sc.V_UNRES and r["unresolved"] == ["elsewhereOnly:bare"]


def test_qualified_member_anywhere_in_the_fork_resolves():
    r = run("public void run() {\n  Elsewhere other = null; other.elsewhereOnly = 1; other.farAway(\"s\");\n}\n")
    assert r["verdict"] == sc.V_OK, r


def test_this_qualified_names_are_checked_against_the_enclosing_classes():
    assert run("public void run() {\n  this.elsewhereOnly = 1;\n}\n")["verdict"] == sc.V_UNRES
    assert run("public void run() {\n  this.running = false;\n}\n")["verdict"] == sc.V_OK


def test_static_import_somewhere_in_the_fork_makes_a_name_importable():
    r = run("public void run() {\n  requireOpen(this);\n}\n")
    assert r["verdict"] == sc.V_OK


def test_arity_mismatch_on_a_fork_only_method():
    r = run("public void run() {\n  Elsewhere other = null; other.farAway(\"a\", \"b\");\n}\n")
    assert r["verdict"] == sc.V_ARITY and r["arity_mismatch"] == ["farAway(2)"]


def test_varargs_accept_any_count_from_n_minus_one():
    assert run("public void run() {\n  log(\"a\");\n log(\"a\", 1, 2);\n}\n")["verdict"] == sc.V_OK


def test_known_name_called_with_a_new_count_is_checked_for_arity_only():
    # step(1) is in the fork function; step(1, 2) is not a declared overload
    r = run("public void run() {\n  step(1, 2);\n}\n")
    assert r["verdict"] == sc.V_ARITY and r["arity_mismatch"] == ["step(2)"]
    # ... unless the PR changes step's signature
    pr = {"types": [], "fields": [], "methods": {"step": [2]}, "ctors": {}}
    assert run("public void run() {\n  step(1, 2);\n}\n", pr=pr)["verdict"] == sc.V_PR


def test_constructor_arity():
    r = run("public void run() {\n  Worker w = new Worker(counts, 3);\n}\n")
    assert r["verdict"] == sc.V_ARITY and r["arity_mismatch"] == ["Worker(2)"]
    assert run("public void run() {\n  Worker w = new Worker(counts);\n}\n")["verdict"] == sc.V_OK


def test_jdk_names_resolve_and_bare_jdk_methods_need_a_jdk_supertype():
    assert run("public void run() {\n  long m = Math.max(1, 2); Duration d = Duration.ZERO;\n}\n")["verdict"] == sc.V_OK
    # Worker implements Runnable (a JDK type): bare interrupt() may be inherited
    assert run("public void run() {\n  interrupt();\n}\n")["verdict"] == sc.V_OK
    # Elsewhere extends a library class only, Checks extends nothing: a bare JDK name is not inherited there
    assert run("public void x() {\n  interrupt();\n}\n", target="public void x() {}\n",
               path="util/Checks.java")["verdict"] == sc.V_UNRES


def test_library_superclass_lets_bare_names_through():
    r = run("void own() {\n  registerWithHost(name);\n}\n", target="void own() {}\n", path="core/Plugin.java")
    assert r["verdict"] == sc.V_OK


def test_comments_strings_and_package_qualified_names_are_ignored():
    r = run("public void run() {\n  // callTheInventedThing()\n  String s = \"inventedName()\";\n"
            "  java.util.concurrent.TimeUnit.SECONDS.toMillis(1);\n}\n")
    assert r["verdict"] == sc.V_OK, r


def test_enum_constants_are_fields():
    assert run("public void run() {\n  Mode m = Mode.SAFE; Mode f = Mode.FAST;\n}\n")["verdict"] == sc.V_OK


def test_only_names_new_relative_to_the_fork_function_are_checked():
    target = "public void run() {\n  legacyThing();\n}\n"
    assert run("public void run() {\n  legacyThing();\n  step(1);\n}\n", target=target)["verdict"] == sc.V_OK


def test_java_decls_members_locals_records_and_supertypes():
    d = si.java_decls(si.lex(TARGET_FILE))
    assert {"counts", "running", "FAST", "SLOW", "SAFE"} <= d["fields"]
    assert d["methods"]["step"] == {1} and d["methods"]["log"] == {-2} and d["methods"]["tune"] == {0}
    assert d["ctors"]["Worker"] == {1} and d["ctors"]["Mode"] == {0, 1}
    assert d["extends"]["Worker"] == ["Base", "Runnable"]
    r = si.java_decls(si.lex("record Point(int x, int y) { }"))
    assert r["fields"] == {"x", "y"} and r["ctors"]["Point"] == {2} and r["methods"]["x"] == {0}


def test_lombok_accessors():
    rec = si.file_record("A.java", "import lombok.Getter;\n@Getter class A { private int size; }\n")
    assert 0 in rec["methods"]["getSize"]


def test_scala_declarations():
    d = si.scala_decls(si.lex("object TestUtils { def createTopic(a: Int, b: Int = 1): Unit = {}\n"
                              "  val brokers = 3 }\ncase class Cfg(host: String, port: Int)\n"))
    assert {"TestUtils", "Cfg"} <= d["types"] and None in d["methods"]["createTopic"]
    assert {"brokers", "host", "port"} <= d["fields"]


def test_pr_additions_are_per_file():
    # `stopping` already existed in another file at the base; the PR adds it to this one: still added
    a = si.added_names_per_file({
        "A.java": ([("A.java", "class A { boolean stopping; }")], [("A.java", "class A { boolean stopping; }")]),
        "B.java": ([("B.java", "class B { int x; }")],
                   [("B.java", "class B { int x; boolean stopping; void stop(int a) {} }")])})
    assert a["fields"] == ["stopping"] and a["methods"] == {"stop": [1]}


def test_member_only_fragment_counts_as_fields():
    a = si.added_names([("u", "")], [("u", "private volatile boolean stopping;\n")])
    assert "stopping" in a["fields"]
