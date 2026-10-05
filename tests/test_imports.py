"""Deterministic import post-pass (rcap.imports)."""

from rcap.imports import fix_imports, parse_imports, unused_import_fails_build

FORK = """package org.x;

import java.util.List;
import org.x.util.Old;

import static org.junit.Assert.assertEquals;

public class A {
    List<String> f(Old o) {
        return o.names();
    }
}
"""
UP = """package org.x;

import java.util.List;
import java.util.Optional;
import org.x.util.Helper;
import org.x.util.Old;
import org.lib.Missing;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThrows;

public class A {}
"""


def yes(imp):
    return None if imp.fqn.startswith("org.lib.") else "test: exists"


def test_adds_the_upstream_import_of_a_name_the_candidate_introduces():
    merged = FORK.replace("return o.names();", "return Optional.of(Helper.wrap(o.names())).get();")
    fix = fix_imports(FORK, merged, UP, "java", exists=yes)
    assert [a["import"] for a in fix.added] == ["import java.util.Optional;", "import org.x.util.Helper;"]
    lines = fix.text.splitlines()
    assert lines.index("import org.x.util.Helper;") == lines.index("import org.x.util.Old;") + 2
    assert all(a["why"] == "test: exists" and a["kind"] == "type" for a in fix.added)
    assert fix.removed == []


def test_static_import_goes_after_the_static_block():
    merged = FORK.replace("return o.names();", "assertThrows(null, null);\n        return o.names();")
    fix = fix_imports(FORK, merged, UP, "java", exists=yes)
    assert fix.added[0]["import"] == "import static org.junit.Assert.assertThrows;"
    lines = fix.text.splitlines()
    assert lines.index("import static org.junit.Assert.assertThrows;") == \
        lines.index("import static org.junit.Assert.assertEquals;") + 1


def test_conservative_skips_are_recorded_not_guessed():
    merged = FORK.replace("return o.names();", "Missing.x(); Unknown.y(); return o.names();")
    fix = fix_imports(FORK, merged, UP, "java", exists=yes)
    assert fix.added == [] and fix.text == merged          # nothing provable -> file untouched
    assert fix.skipped == [{"name": "Missing", "import": "import org.lib.Missing;",
                            "why": "import target not shown to exist for the fork"}]


def test_no_edit_when_the_name_is_already_resolved():
    # already imported from another package, covered by a wildcard, used before, or only in a comment
    fork = FORK.replace("import org.x.util.Old;", "import org.x.util.Old;\nimport other.pkg.Helper;\nimport java.util.*;")
    merged = fork.replace("return o.names();", "Optional.empty(); Helper.z(); // Missing\n        return o.names();")
    assert fix_imports(fork, merged, UP, "java", exists=yes).added == []
    assert fix_imports(FORK, FORK, UP, "java", exists=yes).changed is False


def test_scala_selectors_renames_and_top_level_only():
    up = ("package kafka\n\nimport kafka.utils.{Logging, TestUtils => TU}\nimport scala.collection._\n"
          "import org.apache.kafka.common.Uuid\n\nclass B {\n  import inner.Scoped\n}\n")
    fork = "package kafka\n\nimport kafka.utils.Logging\n\nclass B extends Logging {\n  def f = 1\n}\n"
    assert [(i.simple, i.fqn, i.wildcard) for i in parse_imports(up, "scala")] == [
        ("Logging", "kafka.utils.Logging", False), ("TU", "kafka.utils.TestUtils", False),
        ("*", "scala.collection", True), ("Uuid", "org.apache.kafka.common.Uuid", False)]
    merged = fork.replace("def f = 1", "def f = TU.x(Uuid.randomUuid(), Scoped)")
    fix = fix_imports(fork, merged, up, "scala", exists=lambda i: "ok")
    assert [a["import"] for a in fix.added] == ["import kafka.utils.{TestUtils => TU}",
                                                "import org.apache.kafka.common.Uuid"]
    assert fix.text.splitlines()[3:5] == ["import kafka.utils.{TestUtils => TU}", "import org.apache.kafka.common.Uuid"]


def test_unused_imports_are_removed_only_when_the_build_would_fail_on_them():
    merged = FORK.replace("return o.names();", "return null;").replace("Old o", "Object o")
    assert fix_imports(FORK, merged, UP, "java", exists=yes).removed == []         # default: leave them
    fix = fix_imports(FORK, merged, UP, "java", exists=yes, remove_unused=True)
    assert [r["import"] for r in fix.removed] == ["import org.x.util.Old;"]
    assert "import org.x.util.Old;" not in fix.text and "import java.util.List;" in fix.text
    # the policy: javac never fails on an unused import; checkstyle / fatal scalac warnings do
    assert unused_import_fails_build("java", checkstyle=False, scala_fatal_warnings=True) is False
    assert unused_import_fails_build("java", checkstyle=True, scala_fatal_warnings=False) is True
    assert unused_import_fails_build("scala", checkstyle=True, scala_fatal_warnings=False) is False
    assert unused_import_fails_build("scala", checkstyle=False, scala_fatal_warnings=True) is True
