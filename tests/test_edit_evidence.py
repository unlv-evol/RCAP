"""Edit evidence (rcap.evidence, ExecutionConfig.edit_evidence) and the unit allowlist.

Off (the default), every prompt is byte-identical to before. On, a function unit's prompt shows
the same SAP's non-function edits (imports, fields, ...) as removed/added lines, those units leave
the "signatures only" list, and the PR's functions in other files are listed by signature.
"""

import json
from pathlib import Path

import pytest
from conftest import F_S, F_S_PRIME, F_T, _hunk, _prov
from salp.characterization import Characterizer
from salp.models import SAP, Category, FunctionPayload, PRGroup, SAPReference
from salp.packaging import write_pr_group, write_sap

from rcap import evidence, runoverrides
from rcap.config import ExecutionConfig
from rcap.generate import StubBackend
from rcap.pipeline import run_case

IMPORTS_BEFORE = "import java.util.List;\nimport java.util.Map;\n"
IMPORTS_AFTER = "import java.util.List;\nimport java.util.Map;\nimport java.util.Optional;\n"
IMPORTS_TARGET = "import java.util.List;\n"
OTHER_BEFORE = "public int size(String key) {\n    return 0;\n}\n"
OTHER_AFTER = "public long size(String key, boolean exact) {\n    return 0L;\n}\n"
SAME_BEFORE = "public void close() {\n    flush();\n}\n"
SAME_AFTER = "public void close() {\n    flush(true);\n}\n"


def _write_sap(pr_dir, sap_id, src_file, fns, hunks):
    sap = SAP(sap_id=sap_id, change_id="PR-7", schema_version="1.1",
              source_file=src_file, target_file=src_file,
              functions={fid: f for fid, (f, *_) in fns.items()},
              hunks=hunks, hunk_order=[h.hunk_id for h in hunks],
              provenance=_prov("fixture", repo="apache/kafka", commit="a" * 40))
    for fn, before, after, target in fns.values():
        sap.add_payload(fn.source_before_ref, before)
        sap.add_payload(fn.source_after_ref, after)
        sap.add_payload(fn.target_ref, target)
    for h in hunks:
        sap.add_payload(f"hunks/{h.hunk_id}/hunk.diff", "@@ -1,1 +1,1 @@\n-old\n+new\n")
    ch = Characterizer()
    write_sap(sap, pr_dir / sap_id, profiles={h.hunk_id: ch.characterize(
        {Category(c): ce for c, ce in h.categories.items()}) for h in hunks})


def _payload(fid):
    return FunctionPayload(fn_id=fid, ext="java", has_source_before=True,
                           has_source_after=True, has_target=True)


@pytest.fixture(scope="module")
def ev_pr(tmp_path_factory):
    """PR-7: sap-A edits fn-1 (a function) and A_imports (imports); sap-B edits
    size() (signature change) and close() (body only)."""
    pr_dir = tmp_path_factory.mktemp("sapout-ev") / "linkedinKafka-apacheKafka" / "PR-7"
    _write_sap(pr_dir, "sap-A", "src/main/java/kafka/A.java",
               {"fn-1": (_payload("fn-1"), F_S, F_S_PRIME, F_T),
                "A_imports": (_payload("A_imports"), IMPORTS_BEFORE, IMPORTS_AFTER, IMPORTS_TARGET)},
               [_hunk("H-1", "fn-1"), _hunk("H-2", "A_imports")])
    _write_sap(pr_dir, "sap-B", "src/main/java/kafka/B.java",
               {"B_close": (_payload("B_close"), SAME_BEFORE, SAME_AFTER, SAME_BEFORE),
                "B_size": (_payload("B_size"), OTHER_BEFORE, OTHER_AFTER, OTHER_BEFORE)},
               [_hunk("H-1", "B_close"), _hunk("H-2", "B_size")])
    write_pr_group(PRGroup(pr_id="PR-7", variant_pair="linkedinKafka-apacheKafka",
                           source_repo="apache/kafka", target_repo="linkedin/kafka",
                           saps=[SAPReference(sap_id=s, gacpd_classification="MO", path=s, hunk_count=2)
                                 for s in ("sap-A", "sap-B")]), pr_dir)
    return pr_dir, json.loads((pr_dir / "pr.json").read_text())


class _Echo(StubBackend):
    """Answers every request with its own target (so every unit completes) and records prompts."""

    def __init__(self):
        super().__init__("")
        self.prompts = []

    def generate(self, request):
        self.prompts.append(request.prompt)
        body = request.prompt.split("(adapt this)\n```java\n", 1)[1].split("```", 1)[0]
        return f"```java\n{body}```"


def _unit(res, entity):
    return next(u for u in res.units if u.unit.entity == entity)


def test_off_by_default_and_prompts_unchanged(ev_pr):
    pr_dir, man = ev_pr
    assert ExecutionConfig().edit_evidence is False
    res = run_case(pr_dir / "sap-A", man, _Echo())
    p = _unit(res, "functions/fn-1").request.prompt
    assert "Other edits made by this same change" not in p
    assert "in other files" not in p
    # the old behaviour: the imports unit is a sibling shown as `?`
    assert "- functions/A_imports: `?`" in p


def test_evidence_section_and_sibling_list(ev_pr):
    pr_dir, man = ev_pr
    res = run_case(pr_dir / "sap-A", man, _Echo(), ExecutionConfig(edit_evidence=True))
    fn1 = _unit(res, "functions/fn-1")
    p = fn1.request.prompt
    assert "## Other edits made by this same change (outside functions)" in p
    assert "# A_imports (imports)\n+ import java.util.Optional;" in p
    assert "A_imports: `?`" not in p and "## Other functions edited by this same change" not in p
    assert [e.entity for e in fn1.context.other_edits] == ["A_imports"]
    # other files of the PR: signatures only, and only those the change alters
    sec = p.split("## Functions this same change adds, removes or re-signs in other files "
                  "(signatures only)\n", 1)[1]
    assert ("- B.java: `public long size(String key, boolean exact)` (changed from "
            "`public int size(String key)`)") in sec
    assert "close()" not in p, "a body-only change in another file is not listed"
    assert "flush(true)" not in p and "return 0L" not in p, "never bodies of other files"
    # nothing of sap-A itself in the other-files list
    assert "A.java" not in sec.split("\n\n", 1)[0]


def test_both_modes_carry_the_same_evidence(ev_pr):
    from rcap.config import EvaluationMode
    pr_dir, man = ev_pr
    a = run_case(pr_dir / "sap-A", man, _Echo(), ExecutionConfig(edit_evidence=True))
    b = run_case(pr_dir / "sap-A", man, _Echo(),
                 ExecutionConfig(edit_evidence=True, mode=EvaluationMode.SEMANTIC_ONLY))
    assert _unit(a, "functions/fn-1").context.other_edits == _unit(b, "functions/fn-1").context.other_edits
    assert _unit(a, "functions/fn-1").context.pr_functions == _unit(b, "functions/fn-1").context.pr_functions


def test_deterministic(ev_pr):
    pr_dir, man = ev_pr
    cfg = ExecutionConfig(edit_evidence=True)
    p1 = [u.request.prompt for u in run_case(pr_dir / "sap-A", man, _Echo(), cfg).units]
    evidence._sap_functions.cache_clear()
    p2 = [u.request.prompt for u in run_case(pr_dir / "sap-A", man, _Echo(), cfg).units]
    assert p1 == p2


def test_allowlist_builds_all_sends_listed(ev_pr):
    pr_dir, man = ev_pr
    cfg = ExecutionConfig(edit_evidence=True)
    full_be, part_be = _Echo(), _Echo()
    full = run_case(pr_dir / "sap-A", man, full_be, cfg)
    part = run_case(pr_dir / "sap-A", man, part_be, cfg, generate_units={"functions/fn-1"})
    assert len(full_be.prompts) == 2 and len(part_be.prompts) == 1
    assert _unit(part, "functions/A_imports").generation.outcome == "not_in_allowlist"
    assert _unit(part, "functions/fn-1").generation.outcome == "completion"
    # the listed unit's prompt is the same as without an allowlist (the other unit is still evidence)
    assert part_be.prompts[0] == _unit(full, "functions/fn-1").request.prompt
    assert part.package is None


def test_change_lines_and_kinds():
    assert evidence.change_lines(IMPORTS_BEFORE, IMPORTS_AFTER) == ["+ import java.util.Optional;"]
    assert evidence.change_lines("@@ -1,2 +1,2 @@\n int a;\n", "@@ -1,3 +1,3 @@\n int a;\n int b;\n") \
        == ["+ int b;"]
    assert evidence.is_function_payload(SAME_BEFORE, SAME_AFTER, "java")
    assert evidence.is_function_payload("", SAME_AFTER, "java"), "an added function"
    assert not evidence.is_function_payload(IMPORTS_BEFORE, IMPORTS_AFTER, "java")


def test_cap_truncates_with_note(monkeypatch):
    from types import SimpleNamespace
    monkeypatch.setattr(evidence, "OTHER_EDITS_CHARS", 60)
    ctx = SimpleNamespace(other_edits=[evidence.OtherEdit(
        entity="X_imports", kind="imports", lines=[f"+ import a.b.C{i};" for i in range(20)])],
        pr_functions=[])
    [sec] = evidence.render_sections(ctx, "java")
    assert "+ import a.b.C0;" in sec and "+ import a.b.C19;" not in sec
    assert "more lines not shown: cut at 60 characters]" in sec


def test_overrides_file_and_off_switch(tmp_path, monkeypatch):
    tsv = tmp_path / "units.tsv"
    tsv.write_text("pair\tpr\tsap\tunit\np\tPR-1\tsap-A\tfn-1\n")
    f = tmp_path / "ov.json"
    f.write_text(json.dumps({"name": "t", "edit_evidence": True, "unit_allowlist": str(tsv)}))
    monkeypatch.delenv("RCAP_EDIT_EVIDENCE", raising=False)
    ov = runoverrides.load(f)
    assert ov.active and ov.apply(ExecutionConfig()).edit_evidence is True
    assert ov.units_for("p", "PR-1", "sap-A") == {"functions/fn-1"}
    assert ov.units_for("p", "PR-1", "sap-B") == set()
    monkeypatch.setenv("RCAP_EDIT_EVIDENCE", "0")
    off = runoverrides.load(f)
    assert off.apply(ExecutionConfig()).edit_evidence is False and off.variant != ov.variant
    assert off.units_for("p", "PR-1", "sap-A") == {"functions/fn-1"}, "the ablation keeps the allowlist"
    monkeypatch.delenv("RCAP_EDIT_EVIDENCE")
    monkeypatch.setenv("RCAP_OVERRIDES", "none")
    assert not runoverrides.load().active
    monkeypatch.setenv("RCAP_OVERRIDES", str(tmp_path / "missing.json"))
    assert not runoverrides.load().active


def test_pr_function_signatures_ignore_comments_and_take_bodyless(tmp_path):
    d = tmp_path / "sap-I"
    (d / "functions" / "I_count").mkdir(parents=True)
    (d / "functions" / "I_doc").mkdir(parents=True)
    (d / "sap.json").write_text(json.dumps({"source_file": "src/I.java", "functions": ["I_count", "I_doc"]}))
    (d / "functions" / "I_count" / "source.before.java").write_text("int countItems(Context c);\n")
    (d / "functions" / "I_count" / "source.after.java").write_text("int countItems(Context c, boolean all);\n")
    (d / "functions" / "I_doc" / "source.before.java").write_text("/** Old {doc}; */\nvoid run() {\n  a();\n}\n")
    (d / "functions" / "I_doc" / "source.after.java").write_text("/** New doc */\nvoid run() {\n  b();\n}\n")
    evidence._sap_functions.cache_clear()
    got = {f.entity: f for f in evidence._sap_functions(str(d))}
    assert got["I_count"].status == "signature changed"
    assert got["I_count"].signature_after == "int countItems(Context c, boolean all)"
    assert got["I_doc"].status == "body changed" and got["I_doc"].signature_after == "void run()"


def test_declaration_of():
    assert evidence.declaration_of('@SuppressWarnings({"a", "b"})\n<K> K build(final Map<K, ?> m) {\n  x();\n}\n') \
        == '@SuppressWarnings({"a", "b"}) <K> K build(final Map<K, ?> m)'
    assert evidence.declaration_of("// c\nint f(int a);\n") == "int f(int a)"
    assert evidence.declaration_of("import a.b;\n") is None


# --- relevance filter (ExecutionConfig.evidence_filter = "names") -----------------------------------
# Modelled on apache/kafka PR-16346: SaslChannelBuilder.validateReconfiguration gains
# `throws ConfigException` and a try/catch around sslFactory.validateReconfiguration(configs); the
# imports unit adds ConfigException; SslFactory.validateReconfiguration is re-signed in another file.

SASL_BEFORE = """@Override
    public void validateReconfiguration(Map<String, ?> configs) {
        if (this.securityProtocol == SecurityProtocol.SASL_SSL)
            sslFactory.validateReconfiguration(configs);
    }
"""
SASL_AFTER = """@Override
    public void validateReconfiguration(Map<String, ?> configs) throws ConfigException {
        if (this.securityProtocol == SecurityProtocol.SASL_SSL)
            try {
                sslFactory.validateReconfiguration(configs);
            } catch (IllegalStateException e) {
                throw new ConfigException("SASL reconfiguration failed due to " + e);
            }
    }
"""
SASL_IMPORTS_BEFORE = "import java.util.Map;\nimport org.apache.kafka.common.utils.Utils;\n"
SASL_IMPORTS_AFTER = ("import java.util.Map;\nimport org.apache.kafka.common.config.ConfigException;\n"
                      "import org.apache.kafka.common.utils.Time;\n")
SASL_CLOSE_BEFORE = "@Override\npublic void close() {\n    Utils.closeQuietly(loginManager);\n}\n"
SASL_CLOSE_AFTER = "@Override\npublic void close() {\n    Utils.closeQuietly(loginManager, \"login\");\n}\n"
SSL_VR_BEFORE = "@Override\npublic void validateReconfiguration(Map<String, ?> newConfigs) {\n    check(newConfigs);\n}\n"
SSL_VR_AFTER = ("@Override\npublic void validateReconfiguration(Map<String, ?> newConfigs) throws ConfigException {\n"
                "    check(newConfigs);\n}\n")
SSL_OTHER_BEFORE = "public void reset(int a) {\n    a++;\n}\n"
SSL_OTHER_AFTER = "public void reset(int a, int b) {\n    a++;\n}\n"
SSL_TEST_AFTER = "@Test\npublic void testReconfiguration() throws Exception {\n    sslFactory.validateReconfiguration(cfg);\n}\n"


@pytest.fixture(scope="module")
def pr16346(tmp_path_factory):
    pr_dir = tmp_path_factory.mktemp("sapout-filter") / "linkedinKafka-apacheKafka" / "PR-16346"
    _write_sap(pr_dir, "sap-S", "clients/src/main/java/SaslChannelBuilder.java",
               {"S_validateReconfiguration": (_payload("S_validateReconfiguration"), SASL_BEFORE, SASL_AFTER, SASL_BEFORE),
                "S_imports": (_payload("S_imports"), SASL_IMPORTS_BEFORE, SASL_IMPORTS_AFTER, SASL_IMPORTS_BEFORE),
                "S_close": (_payload("S_close"), SASL_CLOSE_BEFORE, SASL_CLOSE_AFTER, SASL_CLOSE_BEFORE)},
               [_hunk("H-1", "S_validateReconfiguration"), _hunk("H-2", "S_imports"), _hunk("H-3", "S_close")])
    _write_sap(pr_dir, "sap-F", "clients/src/main/java/SslFactory.java",
               {"F_validateReconfiguration": (_payload("F_validateReconfiguration"), SSL_VR_BEFORE, SSL_VR_AFTER, SSL_VR_BEFORE),
                "F_reset": (_payload("F_reset"), SSL_OTHER_BEFORE, SSL_OTHER_AFTER, SSL_OTHER_BEFORE)},
               [_hunk("H-1", "F_validateReconfiguration"), _hunk("H-2", "F_reset")])
    _write_sap(pr_dir, "sap-T", "clients/src/test/java/SslFactoryTest.java",
               {"T_testReconfiguration": (_payload("T_testReconfiguration"), "", SSL_TEST_AFTER, "")},
               [_hunk("H-1", "T_testReconfiguration")])
    write_pr_group(PRGroup(pr_id="PR-16346", variant_pair="linkedinKafka-apacheKafka",
                           source_repo="apache/kafka", target_repo="linkedin/kafka",
                           saps=[SAPReference(sap_id=s, gacpd_classification="MO", path=s, hunk_count=2)
                                 for s in ("sap-S", "sap-F", "sap-T")]), pr_dir)
    return pr_dir, json.loads((pr_dir / "pr.json").read_text())


def _prompt(pr, filt):
    pr_dir, man = pr
    evidence._sap_functions.cache_clear()
    res = run_case(pr_dir / "sap-S", man, _Echo(), ExecutionConfig(edit_evidence=True, evidence_filter=filt))
    return _unit(res, "functions/S_validateReconfiguration")


def test_filter_pr16346_keeps_the_import_and_the_callee(pr16346):
    u = _prompt(pr16346, "names")
    p = u.request.prompt
    assert "# S_imports (imports)\n+ import org.apache.kafka.common.config.ConfigException;\n```" in p
    assert "Time;" not in p and "Utils;" not in p, "imports of names the function does not use are dropped"
    assert ("- SslFactory.java: `@Override public void validateReconfiguration(Map<String, ?> newConfigs) "
            "throws ConfigException` (changed from") in p
    assert "reset(" not in p, "an unrelated re-signed function is dropped"
    assert "testReconfiguration" not in p, "a test method is never listed"
    assert "## Other functions edited by this same change" not in p, "close() is not called"
    assert "Only the edits that mention a name this function uses are shown." in p
    assert u.context.evidence_filtered


def test_filter_none_is_the_unfiltered_evidence(pr16346):
    u = _prompt(pr16346, "none")
    p = u.request.prompt
    assert "Time;" in p and "reset(int a, int b)" in p and "functions/S_close" in p
    assert "Only the edits" not in p and "Only functions whose name" not in p
    assert ExecutionConfig().evidence_filter == "none"
    # the filter has no effect without edit evidence
    pr_dir, man = pr16346
    off = run_case(pr_dir / "sap-S", man, _Echo(), ExecutionConfig(evidence_filter="names"))
    base = run_case(pr_dir / "sap-S", man, _Echo(), ExecutionConfig())
    assert [x.request.prompt for x in off.units] == [x.request.prompt for x in base.units]


def test_filter_omits_empty_sections(ev_pr):
    # fn-1 (conftest F_S) uses none of Optional / size / close: every section goes
    pr_dir, man = ev_pr
    res = run_case(pr_dir / "sap-A", man, _Echo(), ExecutionConfig(edit_evidence=True, evidence_filter="names"))
    p = _unit(res, "functions/fn-1").request.prompt
    assert "## Other edits" not in p and "in other files" not in p and "## Other functions edited" not in p


def test_filter_deterministic_and_mode_independent(pr16346):
    from rcap.config import EvaluationMode
    pr_dir, man = pr16346
    cfg = ExecutionConfig(edit_evidence=True, evidence_filter="names")
    a = [u.request.prompt for u in run_case(pr_dir / "sap-S", man, _Echo(), cfg).units]
    evidence._sap_functions.cache_clear()
    b = [u.request.prompt for u in run_case(pr_dir / "sap-S", man, _Echo(), cfg).units]
    assert a == b
    s = run_case(pr_dir / "sap-S", man, _Echo(), cfg.model_copy(update={"mode": EvaluationMode.SEMANTIC_ONLY}))
    f = run_case(pr_dir / "sap-S", man, _Echo(), cfg)
    assert _unit(s, "functions/S_validateReconfiguration").context.other_edits \
        == _unit(f, "functions/S_validateReconfiguration").context.other_edits


def _names(before, after, target=None):
    return evidence.function_names(before, after, target if target is not None else before)


def test_statement_kinds_and_keys():
    sts = evidence.statements([
        "+ import org.a.b.ConfigException;", "+ import static org.junit.Assert.assertEquals;",
        "+ /**", "+ * The fields.", "+ */", "+ @Mock", "+ private final Map<String, List<Integer>> sizes = new HashMap<>(",
        "+ 16);", "+ RETRY_LATER(3),", "+ public static class Builder extends Base<Foo> {", "+ int count = 1;", "+ }",
        "+ startApp(streams1);", "+ return value;"])
    got = [(s.kind, s.keys) for s in sts]
    assert got == [("import", ["ConfigException"]), ("import", ["assertEquals"]),
                   ("declaration", ["sizes"]), ("declaration", ["RETRY_LATER"]),
                   ("declaration", ["Base", "Builder", "Foo", "count"]),
                   ("fragment", ["startApp", "streams1"]), ("fragment", ["value"])]
    assert sts[2].lines[0] == "+ /**", "comment lines go with the statement after them"


def test_imports_keep_on_type_or_field_convention():
    names = _names("void f() { collectionService.count(); }", "void f() { collectionService.count(ctx); }")
    kept, dropped = evidence.filter_statements(
        ["+ import a.CollectionService;", "+ import a.Autowired;"], names)
    assert [s.keys for s in kept] == [["CollectionService"]] and [s.keys for s in dropped] == [["Autowired"]]


def test_fragments_need_a_name_the_change_introduces_or_removes():
    before = "void t() { startOld(s); andExpect(jsonPath(1)); }"
    after = "void t() { startNew(s); andExpect(jsonPath(2)); }"
    kept, dropped = evidence.filter_statements(
        ["- startOld(s2);", "+ startNew(s2);", "- andExpect(jsonPath(3));", "+ andExpect(jsonPath(4));"],
        _names(before, after))
    assert [ln for s in kept for ln in s.lines] == ["- startOld(s2);", "+ startNew(s2);"]
    assert len(dropped) == 2


def test_reformatted_declaration_dropped_and_rename_kept_in_pairs():
    names = _names("void f() { plugins.x(); }", "void f() { plugins.x(); newSensor.record(); }")
    kept, dropped = evidence.filter_statements(
        ["- @Mock Plugins plugins;", "+ @Mock", "+ Plugins plugins;"], names)
    assert kept == [] and len(dropped) == 2, "moved or reformatted only"
    kept, _ = evidence.filter_statements(["- private final Sensor oldSensor;", "+ private final Sensor newSensor;"], names)
    assert [ln for s in kept for ln in s.lines] == ["- private final Sensor oldSensor;", "+ private final Sensor newSensor;"]


def test_signature_relevance_rules():
    names = _names("void f() { wallet.decrypt(k); create(); }", "void f() { wallet.decrypt(k); create(); }",
                   "void f() { wallet.decrypt(k); create(); }")
    rel = evidence.signature_relevant
    assert rel("public void decrypt(Key k)", "public void decrypt(Key k) throws BadKey", names), \
        "a callee in an unchanged line still counts"
    assert not rel("public void other(Key k)", "public void other(Key k, int n)", names)
    assert not rel(None, "@Test public void decrypt()", names), "test methods are never listed"
    assert not rel("public Integer decrypt()", "@Override public Integer decrypt()", names), \
        "an annotation-only change does not affect callers"
    own = _names("void flush(int a) { sync(); }", "void flush(int a) { sync(a); }")
    assert not rel("void flush(String s)", "void flush(String s)", own, same_file=True), "an overload, not called"
    assert rel("void sync()", "void sync()", own, same_file=True), "a called sibling"
    assert rel("void flush(String s)", "void flush(String s, int n)", own, same_file=True), "a re-signed overload"
    assert evidence.method_name('@SuppressWarnings({"a"}) <K> K build(final Map<K, ?> m)') == "build"


def test_overrides_evidence_filter(tmp_path, monkeypatch):
    f = tmp_path / "ov.json"
    f.write_text(json.dumps({"name": "t", "edit_evidence": True, "evidence_filter": "names"}))
    monkeypatch.delenv("RCAP_EDIT_EVIDENCE", raising=False)
    monkeypatch.delenv("RCAP_EVIDENCE_FILTER", raising=False)
    ov = runoverrides.load(f)
    cfg = ov.apply(ExecutionConfig())
    assert cfg.edit_evidence and cfg.evidence_filter == "names" and ov.variant == "t:ev=1:allow=0:filter=names"
    monkeypatch.setenv("RCAP_EVIDENCE_FILTER", "none")
    ov2 = runoverrides.load(f)
    assert ov2.apply(ExecutionConfig()).evidence_filter == "none" and ov2.variant != ov.variant
    monkeypatch.setenv("RCAP_EVIDENCE_FILTER", "bogus")
    with pytest.raises(ValueError):
        runoverrides.load(f)


REAL_PR16346 = Path("/home/adam/Documents/work/rcap-class-project/1-sap-inputs/sap-2026-11/out-a/"
                    "linkedinKafka-apacheKafka/PR-16346")


@pytest.mark.skipif(not REAL_PR16346.is_dir(), reason="the study's SAP set is not on this machine")
def test_real_pr16346_worked_example():
    man = json.loads((REAL_PR16346 / "pr.json").read_text())
    evidence._sap_functions.cache_clear()
    res = run_case(REAL_PR16346 / "sap-clients-src-main-java-org-apache-kafka-common-network-SaslChannelBuilder-java",
                   man, _Echo(), ExecutionConfig(edit_evidence=True, evidence_filter="names"),
                   generate_units={"functions/SaslChannelBuilder_validateReconfiguration"})
    p = _unit(res, "functions/SaslChannelBuilder_validateReconfiguration").request.prompt
    assert "+ import org.apache.kafka.common.config.ConfigException;" in p
    assert "SslFactory.java: `@Override public void validateReconfiguration(Map<String, ?> newConfigs) throws ConfigException`" in p
    assert "SslFactoryTest" not in p and "testReconfiguration" not in p
