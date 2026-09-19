#!/usr/bin/env python
"""Compare the harness rows against PPatHF (Pan et al., ISSTA'24) — RCAP ref [1].

The Program-Reduction-Only configuration mirrors PPatHF's reduction by design
(section 15: protected set = changed regions only), so the paired comparison
against Raw Context measures the same quantity PPatHF reports for its
reduction module: the retained fraction of the input (their average: 0.70).

What is comparable and what is not:
- COMPARABLE: retained-length ratio after reduction; how often the input fits
  the model window; the population taxonomy against PPatHF's assumptions
  (one-to-one function correspondence, non-null transformation, whole
  function fits the window).
- NOT COMPARABLE: PPatHF's 42.3% accuracy is exact-match against the
  DEVELOPER-PORTED patch on Vim->Neovim C code with a fine-tuned
  StarCoder-15.5B. Our population (apache/kafka -> linkedin/kafka) has no
  developer ports for most changes, RCAP is zero-shot by design, and
  correctness is SVRP's, downstream (I11). No accuracy number is produced
  here, deliberately.

Usage: .venv/bin/python tools/compare_ppathf.py [rows.jsonl ...] [--out FILE]
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

PPATHF = {
    "retained_ratio": 0.70,     # "average length after applying the reduction module"
    "accuracy": "42.3% (131/310) exact vs developer port — Vim->Neovim, fine-tuned StarCoder",
    "context_limits": "even at 8k tokens StarCoder could not fit all 310 patches",
}


def load_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def retained_ratios(rows: list[dict]) -> dict[str, list[tuple[str, float]]]:
    """Per-case retained fraction under Program-Reduction-Only vs Raw Context,
    for prompt characters and AST nodes. Only cases measurable in both modes."""
    by_case: dict[str, dict[str, dict]] = {}
    for r in rows:
        by_case.setdefault(r["case_id"], {})[r["mode"]] = r
    out: dict[str, list[tuple[str, float]]] = {"prompt_chars": [], "nodes_after": []}
    for case_id, modes in sorted(by_case.items()):
        raw, prog = modes.get("raw_context"), modes.get("program_reduction_only")
        if not raw or not prog:
            continue
        if raw["outcome"].startswith("failure:") or prog["outcome"].startswith("failure:"):
            continue
        for metric, pairs in out.items():
            if raw.get(metric) and prog.get(metric):
                pairs.append((case_id, prog[metric] / raw[metric]))
    return out


def taxonomy(rows: list[dict]) -> dict[str, object]:
    """The population against PPatHF's assumptions."""
    by_case: dict[str, list[dict]] = {}
    for r in rows:
        by_case.setdefault(r["case_id"], []).append(r)
    n = len(by_case)
    null_tau = sum(1 for rs in by_case.values()
                   if all(r["outcome"] == "failure:materialization" for r in rs))
    composite = sum(1 for rs in by_case.values() if any(r["units"] > 1 for r in rs))
    over_limit = sum(1 for rs in by_case.values()
                     if any(r["outcome"] == "limits_exceeded" for r in rs))
    completions = sum(1 for rs in by_case.values()
                      if any(r["mode"] == "full_rcap" and r["outcome"] == "completion"
                             for r in rs))
    return {"cases": n, "null_tau": null_tau, "composite": composite,
            "limits_exceeded": over_limit, "full_rcap_completions": completions}


def render(ratios, tax) -> str:
    lines = ["# PPatHF comparison (RCAP ref [1], section 15 config 2)", ""]
    lines.append("## Retained fraction after reduction (paired vs Raw Context)")
    lines.append("")
    lines.append(f"PPatHF reports its reduction keeps **{PPATHF['retained_ratio']:.2f}** "
                 "of the original length on Vim->Neovim C functions. "
                 "Ours, on the kafka population (Program-Reduction-Only, the "
                 "PPatHF-mirror configuration):")
    lines.append("")
    for metric, pairs in ratios.items():
        if not pairs:
            lines.append(f"- {metric}: no measurable pairs")
            continue
        values = [v for _, v in pairs]
        reduced = [(c, v) for c, v in pairs if v < 0.999]
        lines.append(f"- {metric}: mean {statistics.mean(values):.2f} over "
                     f"{len(values)} cases (median {statistics.median(values):.2f}); "
                     f"{len(reduced)} cases actually reduced, their mean "
                     f"{statistics.mean([v for _, v in reduced]):.2f}"
                     if reduced else
                     f"- {metric}: mean {statistics.mean(values):.2f} over "
                     f"{len(values)} cases; none reduced")
        top = sorted(pairs, key=lambda cv: cv[1])[:5]
        for case, v in top:
            lines.append(f"    - {case}: {v:.2f}")
    lines.append("")
    lines.append("## Population vs PPatHF's assumptions")
    lines.append("")
    lines.append(f"{tax['cases']} cases. PPatHF assumes a one-to-one function "
                 "correspondence, a non-null transformation, and a function "
                 "that fits the window; on this real fork population:")
    lines.append(f"- null transformation (refused at materialization): "
                 f"{tax['null_tau']} ({100 * tax['null_tau'] / tax['cases']:.0f}%)")
    lines.append(f"- composite (multiple processing units): {tax['composite']} "
                 f"({100 * tax['composite'] / tax['cases']:.0f}%)")
    lines.append(f"- exceeds the declared request limit even reduced: "
                 f"{tax['limits_exceeded']}")
    lines.append(f"- full-RCAP completions: {tax['full_rcap_completions']}")
    lines.append("")
    lines.append("## Not compared, deliberately")
    lines.append("")
    lines.append(f"- Accuracy: PPatHF reports {PPATHF['accuracy']}. That metric "
                 "needs a developer-ported ground truth, which this population "
                 "mostly lacks, and correctness is SVRP's concern downstream "
                 "(I11); RCAP records no correctness metric in any row.")
    lines.append(f"- Context limits: PPatHF observes that {PPATHF['context_limits']}; "
                 "RCAP's analogue is the typed limits_exceeded refusal rather "
                 "than truncation.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rows", nargs="*", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    paths = args.rows or sorted((root / "out").glob("PR-*/eval-rows.jsonl"))
    rows = load_rows(paths)
    text = render(retained_ratios(rows), taxonomy(rows))
    print(text)
    out = args.out or (root / "out" / "ppathf-comparison.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    print(f"[written] {out}")


if __name__ == "__main__":
    main()
