"""Context-aware single re-prompt: what a missing name IS (rcap.adaptctx), how the
v3 repair request shows it (rcap.request), how it is cut to fit (never code), and
the one-re-prompt loop with a stub backend."""

from __future__ import annotations

import pytest

from rcap import cochange as cc
from rcap.adaptctx import (
    Sources,
    build_symbol_context,
    declarations,
    definition_at,
    import_symbols,
    infer_owner,
    inlining_needs,
)
from rcap.feedback import missing_symbols, parse_compile_errors
from rcap.prorder import find_renames
from rcap.reprompt import RepairFeedback, SymbolContext, fit_feedback, run_reprompt_loop, Validation
from rcap.request import (
    REPAIR_TEMPLATE_ID,
    REPAIR_V3_TEMPLATE_ID,
    untraceable_lines,
)
from tests.test_reprompt import FIRST, ScriptedBackend, _fn, _units

UP_HERDER = """package org.apache.kafka.connect.runtime;

import org.apache.kafka.connect.runtime.rest.entities.ConnectorType;
import org.apache.kafka.connect.errors.ConnectException;
import java.util.Map;

public abstract class AbstractHerder {
    /**
     * Retrieves ConnectorType for the class specified in the connector config
     * @param connConfig the connector config
     */
    public ConnectorType connectorType(Map<String, String> connConfig) {
        if (connConfig == null) {
            return ConnectorType.UNKNOWN;
        }
        String connClass = connConfig.get(ConnectorConfig.CONNECTOR_CLASS_CONFIG);
        try {
            return ConnectorType.from(getConnector(connClass).getClass());
        } catch (ConnectException e) {
            log.warn("Unable to retrieve connector type", e);
            return ConnectorType.UNKNOWN;
        }
    }

    public static final long DEFAULT_TIMEOUT_MS = TimeUnit.SECONDS.toMillis(90);

    public short latestVersion(boolean unstable) {
        return 3;
    }
}
"""
FORK_UNIT = """package org.apache.kafka.connect.runtime.standalone;

import java.util.Map;

public class StandaloneHerder extends AbstractHerder {
    private ConnectorInfo createConnectorInfo(String connector) {
        Map<String, String> config = configState.rawConnectorConfig(connector);
        return new ConnectorInfo(connector, config, connectorType(config));
    }
}
"""
FORK_APIKEYS = """package org.apache.kafka.common.protocol;

public enum ApiKeys {
    PRODUCE;

    public short latestVersion() {
        return 2;
    }

    public short oldestVersion() {
        return 0;
    }

    public boolean isVersionSupported(short apiVersion) {
        return true;
    }
}
"""
UP_APIKEYS = FORK_APIKEYS.replace("public short latestVersion() {",
                                  "public short latestVersion(boolean enableUnstableLastVersion) {")
DIFF = """diff --git a/connect/runtime/src/main/java/org/apache/kafka/connect/runtime/AbstractHerder.java b/connect/runtime/src/main/java/org/apache/kafka/connect/runtime/AbstractHerder.java
--- a/connect/runtime/src/main/java/org/apache/kafka/connect/runtime/AbstractHerder.java
+++ b/connect/runtime/src/main/java/org/apache/kafka/connect/runtime/AbstractHerder.java
@@ -11,0 +12,12 @@
+    public ConnectorType connectorType(Map<String, String> connConfig) {
+        if (connConfig == null) {
+            return ConnectorType.UNKNOWN;
+        }
+        String connClass = connConfig.get(ConnectorConfig.CONNECTOR_CLASS_CONFIG);
+        try {
+            return ConnectorType.from(getConnector(connClass).getClass());
+        } catch (ConnectException e) {
+            log.warn("Unable to retrieve connector type", e);
+            return ConnectorType.UNKNOWN;
+        }
+    }
@@ -12,0 +25 @@
+    public static final long DEFAULT_TIMEOUT_MS = TimeUnit.SECONDS.toMillis(90);
diff --git a/streams/src/main/java/org/apache/kafka/streams/Task.java b/streams/src/main/java/org/apache/kafka/streams/Task.java
--- a/streams/src/main/java/org/apache/kafka/streams/Task.java
+++ b/streams/src/main/java/org/apache/kafka/streams/Task.java
@@ -5 +5 @@
-    StateStore getStore(final String name);
+    StateStore store(final String name);
diff --git a/core/src/main/java/kafka/log/LogConfig.java b/storage/src/main/java/org/apache/kafka/storage/LogConfig.java
similarity index 90%
rename from core/src/main/java/kafka/log/LogConfig.java
rename to storage/src/main/java/org/apache/kafka/storage/LogConfig.java
--- a/core/src/main/java/kafka/log/LogConfig.java
+++ b/storage/src/main/java/org/apache/kafka/storage/LogConfig.java
@@ -1 +1 @@
-package kafka.log;
+package org.apache.kafka.storage;
"""
HERDER = "connect/runtime/src/main/java/org/apache/kafka/connect/runtime/AbstractHerder.java"
UNIT = "connect/runtime/src/main/java/org/apache/kafka/connect/runtime/standalone/StandaloneHerder.java"
APIKEYS = "clients/src/main/java/org/apache/kafka/common/protocol/ApiKeys.java"
TASK = "streams/src/main/java/org/apache/kafka/streams/Task.java"
FORK_TASK = "package org.apache.kafka.streams;\n\npublic interface Task {\n    StateStore getStore(final String name);\n}\n"
FORK_WORDS = {"ConnectorType", "UNKNOWN", "connConfig", "get", "ConnectorConfig", "CONNECTOR_CLASS_CONFIG",
              "from", "getConnector", "getClass", "ConnectException", "log", "warn", "Map", "TimeUnit",
              "SECONDS", "toMillis"}


def _sources(**over) -> Sources:
    up_after = {HERDER: UP_HERDER, APIKEYS: UP_APIKEYS}
    up_before = {HERDER: UP_HERDER.replace("connectorType", "gone"), APIKEYS: UP_APIKEYS}
    fork = {UNIT: FORK_UNIT, APIKEYS: FORK_APIKEYS, TASK: FORK_TASK,
            "core/src/main/java/kafka/log/LogConfig.java": "package kafka.log;\npublic class LogConfig {}\n",
            "connect/runtime/src/main/java/org/apache/kafka/connect/runtime/rest/entities/ConnectorType.java":
                "package org.apache.kafka.connect.runtime.rest.entities;\npublic enum ConnectorType { UNKNOWN }\n"}
    stems = lambda files: (lambda n: [p for p in files if p.rsplit("/", 1)[-1].split(".")[0] == n])  # noqa: E731
    kw = dict(upstream_after=up_after.get, upstream_before=up_before.get, fork=fork.get,
              upstream_type_files=stems(up_before), fork_type_files=stems(fork),
              fork_has_word=lambda w: w in FORK_WORDS)
    kw.update(over)
    return Sources(**kw)


def _javac(path, line, message, *body):
    log = f"/fork/{path}:{line}: error: {message}\n" + "".join(b + "\n" for b in body)
    errs = parse_compile_errors(log)
    for e in errs:
        e.rel = e.path[len("/fork/"):]
    return errs


def _build(errs, **kw):
    files = cc.parse_unified_diff(DIFF)
    args = dict(files=files, region=None, unit_file=UNIT, source_file=UNIT.replace("standalone/", ""),
                sources=_sources(), scope_text=FORK_UNIT, renames=find_renames(files))
    args.update(kw)
    return build_symbol_context(missing_symbols(errs) + import_symbols(errs), errs, **args)


# ---------------------------------------------------------------- declarations
def test_declarations_signature_doc_and_body():
    d = definition_at(UP_HERDER, "java", "connectorType", 15)
    assert d.kind == "method" and d.params == 1 and d.lines == 12
    assert d.signature == "public ConnectorType connectorType(Map<String, String> connConfig)"
    assert d.text.startswith("public ConnectorType connectorType(") and d.text.endswith("}")
    assert d.text.splitlines()[1] == "    if (connConfig == null) {"       # dedented
    assert d.doc.startswith("/**") and "Retrieves ConnectorType" in d.doc
    const = declarations(UP_HERDER, "java", "DEFAULT_TIMEOUT_MS")[0]
    assert const.kind == "field" and const.signature.endswith("TimeUnit.SECONDS.toMillis(90);")
    scala = declarations("class A {\n  def poll(timeout: Long, max: Int): Seq[R] = {\n    Nil\n  }\n}\n",
                         "scala", "poll")[0]
    assert scala.signature.startswith("def poll(timeout: Long, max: Int): Seq[R]") and scala.params == 2


@pytest.mark.parametrize("usage,name,owner", [
    ("SaslExtensions.empty();", "empty", "SaslExtensions"),
    ("return connectorType(config);", "connectorType", "Enclosing"),
    ("this.reset(now);", "reset", "Enclosing"),
    ("queue.poll(wallClockTime);", "poll", "RecordQueue"),
    ("builder.foo().bar();", "bar", None),              # a receiver that cannot be typed
    ("new ErrorHandlingMetrics(id);", "ErrorHandlingMetrics", None),
])
def test_infer_owner(usage, name, owner):
    scope = "void f() {\n  final RecordQueue queue = queues.get(tp);\n}\n"
    assert infer_owner(usage, name, scope, "Enclosing") == owner


# ---------------------------------------------------------------- what a name is
def test_helper_added_by_the_same_change_is_shown_with_its_body():
    errs = _javac(UNIT, 8, "cannot find symbol", "        return new ConnectorInfo(connector, config, connectorType(config));",
                  "  symbol:   method connectorType(Map<String,String>)", "  location: class StandaloneHerder")
    (sc,), omitted = _build(errs)
    assert omitted == 0 and sc.origin == "added_by_change" and sc.upstream_file == HERDER
    assert sc.upstream_complete and sc.upstream_lines == 12
    assert "return ConnectorType.from(getConnector(connClass).getClass());" in sc.upstream_text
    assert sc.usage.startswith("return new ConnectorInfo(")
    # what inlining it into ANOTHER file needs
    assert sc.names_checked and sc.missing_in_fork == []
    assert sc.qualify == ["org.apache.kafka.connect.runtime.rest.entities.ConnectorType",
                          "org.apache.kafka.connect.errors.ConnectException"]


def test_helper_using_something_the_fork_lacks_says_so():
    missing, _ = inlining_needs("int f() { return Brand.newThing(x); }", "f", UP_HERDER, FORK_UNIT,
                                _sources(fork_has_word=lambda w: w in {"x"}))
    assert missing == ["Brand", "newThing"]


def test_long_definition_is_shown_as_doc_and_signature_only(monkeypatch):
    monkeypatch.setattr("rcap.adaptctx.MAX_BODY_LINES", 5)
    errs = _javac(UNIT, 8, "cannot find symbol", "        connectorType(config);",
                  "  symbol:   method connectorType(Map<String,String>)", "  location: class StandaloneHerder")
    (sc,), _ = _build(errs)
    assert sc.origin == "added_by_change" and not sc.upstream_complete and sc.upstream_lines == 12
    assert sc.upstream_text.startswith("/**") and sc.upstream_text.rstrip().endswith("connConfig)")
    assert "getConnector" not in sc.upstream_text and not sc.names_checked


def test_constant_added_by_the_change():
    errs = _javac(UNIT, 8, "cannot find symbol", "        wait(AbstractHerder.DEFAULT_TIMEOUT_MS);",
                  "  symbol:   variable DEFAULT_TIMEOUT_MS", "  location: class AbstractHerder")
    (sc,), _ = _build(errs)
    assert sc.origin == "added_by_change" and sc.owner == "AbstractHerder"
    assert sc.upstream_text == "public static final long DEFAULT_TIMEOUT_MS = TimeUnit.SECONDS.toMillis(90);"


def test_symbol_older_than_the_change_lists_the_fork_members():
    errs = _javac(UNIT, 8, "method latestVersion in enum ApiKeys cannot be applied to given types;",
                  "        short v = ApiKeys.PRODUCE.latestVersion(true);",
                  "  required: no arguments", "  found:    boolean")
    (sc,), _ = _build(errs, hints={"latestVersion": ["fork equivalent found by the screen (arity): x"]})
    assert sc.origin == "predates_change" and sc.owner == "ApiKeys" and sc.upstream_file == APIKEYS
    assert sc.upstream_signature == "public short latestVersion(boolean enableUnstableLastVersion)"
    assert sc.fork_heading == f"The fork's ApiKeys ({APIKEYS}) declares"
    assert sc.fork_members[0] == "public short latestVersion()"            # same name first
    assert "public short oldestVersion()" in sc.fork_members               # then close names
    assert sc.hints == ["fork equivalent found by the screen (arity): x"]


def test_rename_by_the_same_change_keeps_the_fork_name():
    errs = _javac(UNIT, 8, "cannot find symbol", "        return task.store(name);",
                  "  symbol:   method store(String)", "  location: variable task of type Task")
    (sc,), _ = _build(errs)
    assert sc.origin == "renamed_by_change" and sc.old_name == "getStore" and sc.owner == "Task"
    assert sc.upstream_file == TASK
    assert sc.fork_members == ["StateStore getStore(final String name);"]


def test_type_moved_by_the_same_change_is_an_import_to_keep():
    errs = _javac(UNIT, 3, "package org.apache.kafka.storage does not exist",
                  "import org.apache.kafka.storage.LogConfig;")
    assert [m.name for m in import_symbols(errs)] == ["LogConfig"]
    (sc,), _ = _build(errs)
    assert sc.origin == "moved_by_change" and sc.old_name == "kafka.log.LogConfig"
    assert sc.upstream_file == "storage/src/main/java/org/apache/kafka/storage/LogConfig.java"


def test_unknown_name_is_said_to_be_unknown():
    errs = _javac(UNIT, 8, "cannot find symbol", "        invented(config);",
                  "  symbol:   method invented(Map<String,String>)", "  location: class StandaloneHerder")
    (sc,), _ = _build(errs)
    assert sc.origin == "unknown" and sc.fork_members == []
    assert "declares no member named `invented`" in sc.fork_heading


def test_symbol_count_is_bounded_and_the_rest_counted():
    errs = []
    for i in range(9):
        errs += _javac(UNIT, 10 + i, "cannot find symbol", f"        m{i}();",
                       f"  symbol:   method m{i}()", "  location: class StandaloneHerder")
    scs, omitted = _build(errs)
    assert len(scs) == 6 and omitted == 3


# ---------------------------------------------------------------- the v3 request
def _contexts():
    errs = _javac(UNIT, 8, "cannot find symbol", "        connectorType(config);",
                  "  symbol:   method connectorType(Map<String,String>)", "  location: class StandaloneHerder")
    errs += _javac(UNIT, 9, "method latestVersion in enum ApiKeys cannot be applied to given types;",
                   "        short v = ApiKeys.PRODUCE.latestVersion(true);")
    errs += _javac(UNIT, 10, "cannot find symbol", "        return task.store(name);",
                   "  symbol:   method store(String)", "  location: variable task of type Task")
    return _build(errs)[0]


def test_v3_request_shows_each_name_and_is_traceable(sap_dir, pr_manifest):
    unit = _units(sap_dir, pr_manifest, ScriptedBackend([FIRST]))[0]
    fb = RepairFeedback(attempt=2, kind="build", previous_candidate=unit.generation.candidate,
                        error_text="- F.java:8: cannot find symbol\n    symbol: method connectorType",
                        symbol_context=_contexts())
    ctx, req = fit_feedback(unit.context, fb, None)
    p = req.prompt
    assert req.template_id == REPAIR_V3_TEMPLATE_ID
    order = [p.index(h) for h in (
        "## Source function BEFORE", "## Source function AFTER", "## The change to port",
        "## ORIGINAL function in the fork, from BEFORE the port (reference only; not an acceptable "
        "answer unless this function has nothing to port)",
        "## PREVIOUS ATTEMPT (attempt 1): the code to fix",
        "## Compiler diagnostics for the previous attempt",
        "## The names the previous attempt uses that the fork does not have", "## Output")]
    assert order == sorted(order)
    # (a) added by the same change: definition, stated absent, inline-or-decline
    assert "### `connectorType` (on `StandaloneHerder`)" in p
    assert f"ADDED BY THE SAME CHANGE in {HERDER}. It is NOT PRESENT in the fork" in p
    assert "Source definition:\n```java\npublic ConnectorType connectorType(" in p
    assert "Every name this definition uses occurs in the fork." in p
    assert "write them fully qualified when inlining): org.apache.kafka.connect.runtime.rest.entities.ConnectorType" in p
    assert f"otherwise answer CANNOT_ADAPT: needs connectorType from {HERDER}" in p
    # (b) older than the change: the fork's own API
    assert "OLDER THAN THE CHANGE: the source project already had it before this change" in p
    assert "Source declaration: `public short latestVersion(boolean enableUnstableLastVersion)`" in p
    assert f"The fork's ApiKeys ({APIKEYS}) declares:\n  public short latestVersion()" in p
    assert "express the source change with the fork's API as declared above" in p
    # rename: keep the fork's name; the original is acceptable only when nothing is left to port
    assert "RENAMED BY THE SAME CHANGE" in p and "keep the fork's name `getStore`" in p
    out = p.split("## Output")[1]
    assert "The ORIGINAL function unchanged is acceptable only if nothing of the change is left" in out
    assert "CANNOT_ADAPT: needs <symbol or change> from <file>" in out
    assert "Co-changes of the same upstream change" not in p      # v3 states it in the task
    assert untraceable_lines(p, ctx) == []


def test_without_symbol_context_the_v2_request_is_unchanged(sap_dir, pr_manifest):
    unit = _units(sap_dir, pr_manifest, ScriptedBackend([FIRST]))[0]
    fb = RepairFeedback(attempt=2, kind="build", previous_candidate=unit.generation.candidate,
                        error_text="- F.java:8: cannot find symbol")
    _, req = fit_feedback(unit.context, fb, None)
    assert req.template_id == REPAIR_TEMPLATE_ID
    assert "The names the previous attempt uses" not in req.prompt
    assert "returning it unchanged is NOT an acceptable answer" in req.prompt
    assert "None applied: only this function is changed in the fork for this build." in req.prompt


def test_context_is_cut_to_fit_and_code_never_is(sap_dir, pr_manifest):
    unit = _units(sap_dir, pr_manifest, ScriptedBackend([FIRST]))[0]
    fb = RepairFeedback(attempt=2, kind="build", previous_candidate=unit.generation.candidate,
                        error_text="- F.java:8: cannot find symbol", symbol_context=_contexts())
    full_ctx, full = fit_feedback(unit.context, fb, None)
    assert full_ctx.feedback.context_truncated == []
    # 1. bodies go first: the definition shrinks to its declaration
    ctx, req = fit_feedback(unit.context, fb, len(full.prompt) - 200)
    assert ctx.feedback.context_truncated == ["symbol_bodies"]
    assert "Source declaration (body not shown: 12 lines)" in req.prompt
    assert "getConnector(connClass)" not in req.prompt and len(ctx.feedback.symbol_context) == 3
    assert len(req.prompt) <= len(full.prompt) - 200
    # 2. then whole entries from the end, counted
    ctx, req = fit_feedback(unit.context, fb, len(req.prompt) - 300)
    assert "symbol_context" in ctx.feedback.context_truncated
    assert ctx.feedback.symbol_context_omitted >= 1
    assert "more missing name(s) not described: cut to fit the request limit" in req.prompt
    # the code is intact at every size
    for r in (req, full):
        assert unit.generation.candidate.strip() in r.prompt
        assert unit.context.transformation["target"] in r.prompt
        assert unit.context.transformation["source.after"] in r.prompt
    assert untraceable_lines(req.prompt, ctx) == []


# ---------------------------------------------------------------- exactly one re-prompt
def _extras(calls):
    def repair_extras(i, validation):
        calls.append(i)
        return {"symbol_context": [SymbolContext(
            symbol="missing", owner="Validator", origin="added_by_change", upstream_file="V.java",
            upstream_text="void missing(Config c) {\n    audit(c);\n}", upstream_signature="void missing(Config c)",
            upstream_complete=True, upstream_lines=3)]}
    return repair_extras


def _validator(seen, ok):
    def validate(attempt, expanded):
        seen.append(attempt)
        good = ok(expanded[0])
        return Validation(verdict="VALID" if good else "BUILD_FAIL", kind="build",
                          feedback="- V.java:3: cannot find symbol\n    symbol: method missing(Config)")
    return validate


def test_one_context_aware_reprompt_inlines_the_helper(sap_dir, pr_manifest):
    backend = ScriptedBackend([FIRST, _fn("check(c); audit(c);"), _fn("check(c); never(c);")])
    units = _units(sap_dir, pr_manifest, backend)
    seen, calls = [], []
    rec = run_reprompt_loop(units, backend, _validator(seen, lambda t: "audit(c)" in t), k=2,
                            repair_extras=_extras(calls))
    assert rec["stop_reason"] == "success" and rec["first_success_attempt"] == 2
    assert seen == [1, 2] and calls == [0]
    assert len(backend.requests) == 2                      # attempt 1 + exactly one re-prompt
    repair = backend.requests[1]
    assert repair.template_id == REPAIR_V3_TEMPLATE_ID
    assert "void missing(Config c) {\n    audit(c);\n}" in repair.prompt
    assert rec["attempts"][1]["units"][0]["template_id"] == REPAIR_V3_TEMPLATE_ID


def test_k2_never_issues_a_second_reprompt(sap_dir, pr_manifest):
    backend = ScriptedBackend([FIRST, _fn("check(c); stillMissing(c);"), _fn("check(c); audit(c);")])
    units = _units(sap_dir, pr_manifest, backend)
    seen = []
    rec = run_reprompt_loop(units, backend, _validator(seen, lambda t: "audit(c)" in t), k=2,
                            repair_extras=_extras([]))
    assert rec["stop_reason"] == "k_exhausted" and rec["final_verdict"] == "BUILD_FAIL"
    assert seen == [1, 2] and len(backend.requests) == 2 and rec["first_success_attempt"] is None


def test_reverting_to_the_fork_code_is_an_echo_not_a_success(sap_dir, pr_manifest):
    backend = ScriptedBackend([FIRST])
    units = _units(sap_dir, pr_manifest, backend)
    reduced = units[0].context.transformation["target"]
    backend._responses.append("```java\n" + reduced + "```")
    seen = []
    rec = run_reprompt_loop(units, backend, _validator(seen, lambda t: True if len(seen) > 1 else False),
                            k=2, repair_extras=_extras([]))
    assert seen == [1]                                     # the echo is never built
    assert rec["final_verdict"] == "ECHO" and rec["first_success_attempt"] is None
    assert rec["attempts"][1]["echo"] is True and rec["stop_reason"] == "k_exhausted"


def test_cannot_adapt_is_a_typed_refusal(sap_dir, pr_manifest):
    backend = ScriptedBackend([FIRST, "CANNOT_ADAPT: needs missing from V.java"])
    units = _units(sap_dir, pr_manifest, backend)
    seen = []
    rec = run_reprompt_loop(units, backend, _validator(seen, lambda t: False), k=2,
                            repair_extras=_extras([]))
    assert rec["stop_reason"] == "needs_cochange" and rec["needs_cochange_source"] == "model"
    assert rec["needs_cochange"][0]["symbol"] == "missing" and rec["needs_cochange"][0]["file"] == "V.java"
    assert rec["final_verdict"] == "BUILD_FAIL" and seen == [1]


def test_placeholder_leak_in_the_repair_answer_is_rejected(sap_dir, pr_manifest):
    backend = ScriptedBackend([FIRST, "```java\nvoid validate(Config c) { /* RCAP_PH_target.1 */ "
                                      "/* RCAP_PH_target.9 */ audit(c); }\n```"])
    units = _units(sap_dir, pr_manifest, backend)
    for u in units:
        u.expected_placeholders = {"/* RCAP_PH_target.1 */"}
    seen = []
    rec = run_reprompt_loop(units, backend, _validator(seen, lambda t: True if len(seen) > 1 else False),
                            k=2, repair_extras=_extras([]))
    assert seen == [1] and rec["stop_reason"] == "unparseable"
    assert rec["stop_detail"] == "placeholder_violation" and rec["first_success_attempt"] is None


def test_jdk_type_the_file_does_not_import_is_to_be_qualified():
    errs = _javac(UNIT, 8, "cannot find symbol", "        Optional<String> x = Optional.empty();",
                  "  symbol:   class Optional", "  location: class StandaloneHerder")
    up = UP_HERDER.replace("import java.util.Map;", "import java.util.Map;\nimport java.util.Optional;")
    (sc,), _ = _build(errs, source_file=HERDER, sources=_sources(upstream_after={HERDER: up}.get))
    assert sc.owner is None and sc.fork_members == []
    assert "imports it as `java.util.Optional`" in sc.fork_heading
    assert "write it fully qualified" in sc.fork_heading


def test_a_package_segment_is_not_an_owner_type():
    errs = _javac(UNIT, 3, "cannot find symbol", "import org.apache.kafka.connect.runtime.rest.entities.ConnectorType;",
                  "  symbol:   class ConnectorType", "  location: package org.apache.kafka.connect.runtime.rest.entities")
    (sc,), _ = _build(errs)
    assert sc.owner is None and sc.fork_heading.startswith("The fork declares a type `ConnectorType` in")
