#!/usr/bin/env python
"""Aggregate section-15 eval rows across configurations.

Reads eval-rows.jsonl files and reports, per configuration, absolute values
and percentage reduction relative to Raw Context (RCAP ref section 15:
"Absolute values and percentage reduction relative to Raw Context are both
reported"). Cases whose semantic reduction ran degraded are reported
separately, never folded into a relationship-guided claim.

Percentages are paired: for each metric, only cases with the metric recorded
in BOTH the configuration row and that case's Raw Context row contribute, so
a missing backend-reported token count shrinks n instead of skewing the sum.

Usage:
    .venv/bin/python tools/aggregate_rows.py [rows.jsonl ...] [--out FILE]
Defaults to out/PR-*/eval-rows.jsonl and writes out/aggregate.md.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REF_MODE = "raw_context"
MODES = ("raw_context", "program_reduction_only", "semantic_reduction_only", "full_rcap")
# (metric, only meaningful on non-failure rows)
METRICS = ("nodes_after", "context_chars", "prompt_chars", "input_tokens", "output_tokens")


def load_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def split_by_reduction_mode(rows: list[dict]) -> dict[str, list[dict]]:
    """Partition CASES (all four rows together) by semantic-reduction mode:
    'degraded' if any hunk of any row ran degraded, 'guided' if any hunk ran
    relationship-guided (and none degraded), else 'unmeasured' — typically a
    case whose rows all failed before a reduction mode was recorded. A case
    never appears in more than one population (section 15: degraded cases are
    excluded from, or reported separately in, any semantic-reduction claim)."""
    modes_by_case: dict[str, set[str]] = {}
    for r in rows:
        modes_by_case.setdefault(r["case_id"], set()).update(
            r.get("reduction_mode", {}).values())

    def bucket(case_id: str) -> str:
        modes = modes_by_case[case_id]
        if "degraded" in modes:
            return "degraded"
        if "relationship_guided" in modes:
            return "guided"
        return "unmeasured"

    out: dict[str, list[dict]] = {"guided": [], "degraded": [], "unmeasured": []}
    for r in rows:
        out[bucket(r["case_id"])].append(r)
    return out


def aggregate(rows: list[dict]) -> dict[str, dict]:
    """Per-mode aggregates: outcome counts, and per-metric paired sums with
    percent change vs the same cases' Raw Context rows."""
    ref_by_case = {r["case_id"]: r for r in rows if r["mode"] == REF_MODE}
    out: dict[str, dict] = {}
    for mode in MODES:
        mode_rows = [r for r in rows if r["mode"] == mode]
        if not mode_rows:
            continue
        outcomes: dict[str, int] = {}
        for r in mode_rows:
            outcomes[r["outcome"]] = outcomes.get(r["outcome"], 0) + 1
        metrics: dict[str, dict] = {}
        for metric in METRICS:
            pairs = []
            for r in mode_rows:
                ref = ref_by_case.get(r["case_id"])
                if ref is None:
                    continue
                v, rv = r.get(metric), ref.get(metric)
                # failure rows carry structural zeros, not measurements
                if r["outcome"].startswith("failure:") or ref["outcome"].startswith("failure:"):
                    continue
                if isinstance(v, (int, float)) and isinstance(rv, (int, float)) and rv:
                    pairs.append((v, rv))
            if not pairs:
                continue
            total = sum(v for v, _ in pairs)
            ref_total = sum(rv for _, rv in pairs)
            metrics[metric] = {
                "n": len(pairs),
                "total": total,
                "ref_total": ref_total,
                "pct_vs_raw": 100.0 * (total - ref_total) / ref_total,
            }
        out[mode] = {"rows": len(mode_rows), "outcomes": outcomes, "metrics": metrics}
    return out


def render(populations: dict[str, dict[str, dict]]) -> str:
    lines = ["# Section-15 aggregation: absolute values and % vs Raw Context", ""]
    for pop_name, agg in populations.items():
        if not agg:
            continue
        n_cases = max((a["rows"] for a in agg.values()), default=0)
        lines.append(f"## Population: {pop_name} reduction mode ({n_cases} cases)")
        lines.append("")
        lines.append("| configuration | completions | other outcomes | "
                     + " | ".join(f"{m} (Δ% vs raw)" for m in METRICS) + " |")
        lines.append("|" + "---|" * (3 + len(METRICS)))
        for mode, a in agg.items():
            completions = a["outcomes"].get("completion", 0)
            others = ", ".join(f"{k}:{v}" for k, v in sorted(a["outcomes"].items())
                               if k != "completion") or "—"
            cells = []
            for metric in METRICS:
                m = a["metrics"].get(metric)
                if m is None:
                    cells.append("not recorded")
                elif mode == REF_MODE:
                    cells.append(f"{m['total']} (ref, n={m['n']})")
                else:
                    cells.append(f"{m['total']} ({m['pct_vs_raw']:+.1f}%, n={m['n']})")
            lines.append(f"| {mode} | {completions} | {others} | " + " | ".join(cells) + " |")
        lines.append("")
    lines.append("Token columns are backend-reported counts; rows without a reporting "
                 "backend are excluded pairwise (n shows how many cases contribute). "
                 "Failure rows contribute to outcome counts only. Completion is not "
                 "correctness (I11).")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rows", nargs="*", type=Path,
                        help="eval-rows.jsonl files (default: out/PR-*/eval-rows.jsonl)")
    parser.add_argument("--out", type=Path, default=None,
                        help="also write the table here (default: out/aggregate.md)")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    paths = args.rows or sorted((root / "out").glob("PR-*/eval-rows.jsonl"))
    if not paths:
        raise SystemExit("no eval-rows.jsonl found; run tools/run_harness.py first")

    rows = load_rows(paths)
    populations = {name: aggregate(subset)
                   for name, subset in split_by_reduction_mode(rows).items()}
    text = render(populations)
    print(text)
    out = args.out or (root / "out" / "aggregate.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"[written] {out}")


if __name__ == "__main__":
    main()
