"""Section-15 measurement completeness: tokens, characterization columns,
reference-configuration identifier, and the aggregation over rows."""

import importlib.util
import json
from pathlib import Path

from rcap.eval_harness import run_configs
from rcap.generate import StubBackend
from rcap.intake import load_case

CANDIDATE = "```java\nvoid validate(Config c) { }\n```"


def _load_aggregate_module():
    path = Path(__file__).parent.parent / "tools" / "aggregate_rows.py"
    spec = importlib.util.spec_from_file_location("aggregate_rows", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_characterization_scores_from_salp_profile(sap_dir, pr_manifest):
    """Coverage/Fidelity are min-over-hunks scores, Readiness the aggregate
    level — the same aggregation rule SALP itself records."""
    case = load_case(sap_dir, pr_manifest=pr_manifest)
    scores = case.characterization_scores()
    assert 0.0 < scores["coverage"] <= 1.0
    assert 0.0 < scores["fidelity"] <= 1.0
    assert scores["readiness"] in ("HIGH", "MEDIUM", "LOW")


def test_rows_carry_reference_config_and_characterization(sap_dir, pr_manifest):
    rows = run_configs(sap_dir, pr_manifest, StubBackend(CANDIDATE))
    raw_row = next(r for r in rows if r.mode == "raw_context")
    for row in rows:
        assert row.ref_config_id == raw_row.config_id
        assert row.coverage is not None
        assert row.fidelity is not None
        assert row.readiness in ("HIGH", "MEDIUM", "LOW")


class UsageBackend(StubBackend):
    """Backend that reports runtime token counts, the way OllamaBackend does."""

    def generate(self, request):
        self.last_usage = {"input_tokens": 111, "output_tokens": 22}
        return super().generate(request)


def test_tokens_recorded_when_reported_and_absent_otherwise(sap_dir, pr_manifest):
    reported = run_configs(sap_dir, pr_manifest, UsageBackend(CANDIDATE))
    assert all(r.input_tokens == 111 and r.output_tokens == 22 for r in reported)

    # No tokenizer -> honestly absent, never an estimate.
    plain = run_configs(sap_dir, pr_manifest, StubBackend(CANDIDATE))
    assert all(r.input_tokens is None and r.output_tokens is None for r in plain)


def test_failure_rows_keep_characterization(sap_dir, pr_manifest, tmp_path):
    import shutil

    null_sap = tmp_path / "sap-NullTau"
    shutil.copytree(sap_dir, null_sap)
    fn = null_sap / "functions" / "fn-1"
    (fn / "source.after.java").write_text((fn / "source.before.java").read_text())

    rows = run_configs(null_sap, pr_manifest, StubBackend(CANDIDATE))
    for row in rows:
        assert row.outcome == "failure:materialization"
        assert row.coverage is not None
        assert row.readiness in ("HIGH", "MEDIUM", "LOW")
        assert row.input_tokens is None  # no generation happened


def _row(case, mode, *, nodes=100, tokens=None, outcome="completion", red_mode="off"):
    return {"case_id": case, "config_id": f"dev:{mode}", "mode": mode,
            "reduction_mode": {"H-1": red_mode}, "outcome": outcome,
            "nodes_after": nodes, "context_chars": nodes * 10,
            "prompt_chars": nodes * 8, "input_tokens": tokens,
            "output_tokens": 5 if tokens else None}


def test_aggregate_pct_vs_raw_and_degraded_separation():
    agg_mod = _load_aggregate_module()
    rows = [
        # guided case: full_rcap halves the nodes
        _row("c-guided", "raw_context", nodes=200, tokens=400),
        _row("c-guided", "full_rcap", nodes=100, tokens=200, red_mode="relationship_guided"),
        # degraded case must not contaminate the guided population
        _row("c-degraded", "raw_context", nodes=1000, tokens=None),
        _row("c-degraded", "full_rcap", nodes=900, red_mode="degraded"),
    ]
    pops = {name: agg_mod.aggregate(subset)
            for name, subset in agg_mod.split_by_reduction_mode(rows).items()}

    guided = pops["guided"]["full_rcap"]["metrics"]
    assert guided["nodes_after"] == {"n": 1, "total": 100, "ref_total": 200,
                                     "pct_vs_raw": -50.0}
    assert guided["input_tokens"]["pct_vs_raw"] == -50.0

    degraded = pops["degraded"]["full_rcap"]["metrics"]
    assert degraded["nodes_after"]["ref_total"] == 1000
    # token count unreported for the degraded case's raw row -> excluded pairwise
    assert "input_tokens" not in degraded

    text = agg_mod.render(pops)
    assert "guided" in text and "degraded" in text and "-50.0%" in text


def test_aggregate_excludes_failure_rows_from_size_metrics():
    agg_mod = _load_aggregate_module()
    rows = [
        _row("c1", "raw_context", nodes=200),
        {**_row("c1", "full_rcap", nodes=0), "outcome": "failure:materialization"},
    ]
    agg = agg_mod.aggregate(rows)
    assert "nodes_after" not in agg["full_rcap"]["metrics"]
    assert agg["full_rcap"]["outcomes"] == {"failure:materialization": 1}


def test_package_characterization_carries_scores_and_timestamp(sap_dir, pr_manifest):
    """Section 12: the package's characterization holds coverage, fidelity and
    readiness; section 11: the execution record carries a timestamp alongside
    the deterministic exec_id."""
    from rcap.pipeline import run_case

    result = run_case(sap_dir, pr_manifest,
                      StubBackend("```java\nvoid v() { /* RCAP_PH_target.1 */ }\n```"))
    pkg = result.package
    assert pkg is not None
    for key in ("coverage", "fidelity", "readiness"):
        assert key in pkg.characterization
    assert pkg.generation_meta["executed_at"].startswith("20")
    assert result.generation.executed_at == pkg.generation_meta["executed_at"]


def test_rows_roundtrip_through_jsonl(sap_dir, pr_manifest, tmp_path):
    out = tmp_path / "rows.jsonl"
    rows = run_configs(sap_dir, pr_manifest, UsageBackend(CANDIDATE), out_path=out)
    written = [json.loads(line) for line in out.read_text().splitlines()]
    assert len(written) == len(rows)
    for w in written:
        for field in ("ref_config_id", "input_tokens", "output_tokens",
                      "coverage", "fidelity", "readiness"):
            assert field in w
