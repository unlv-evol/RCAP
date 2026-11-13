"""Edit evidence (rcap.evidence, ExecutionConfig.edit_evidence) and the unit allowlist.

Off (the default), every prompt is byte-identical to before. On, a function unit's prompt shows
the same SAP's non-function edits (imports, fields, ...) as removed/added lines, those units leave
the "signatures only" list, and the PR's functions in other files are listed by signature.
"""

import json

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
