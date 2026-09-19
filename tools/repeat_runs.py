#!/usr/bin/env python
"""Repeatability study: the same PR run N times against the live backend.

Section 11: repeated generations are separate executions, each with its own
generation metadata — this script measures how backend-reported token counts,
runtime, and the candidate itself vary across identical requests
(temperature 0, fixed seed, identical request_hash).

Known caveat, measured rather than hidden: ollama's prompt_eval_count is the
number of prompt tokens the runtime actually EVALUATED — a warm prompt-prefix
cache makes later repetitions report fewer input tokens for the same prompt.

Usage:
  .venv/bin/python tools/repeat_runs.py <pr-dir> [--reps 5] [--modes raw_context,full_rcap]
Writes out/repeatability/<PR>/rows.jsonl and prints a variance table.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path

from rcap.backend_ollama import OllamaBackend
from rcap.config import EvaluationMode
from rcap.eval_harness import run_configs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pr_dir", type=Path)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--modes", default="raw_context,full_rcap")
    args = parser.parse_args()

    modes = tuple(EvaluationMode(m) for m in args.modes.split(","))
    manifest = json.loads((args.pr_dir / "pr.json").read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parent.parent
    out_dir = root / "out" / "repeatability" / args.pr_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = out_dir / "rows.jsonl"
    rows_path.unlink(missing_ok=True)

    backend = OllamaBackend()
    records: list[dict] = []
    for rep in range(1, args.reps + 1):
        for sap in sorted(args.pr_dir.glob("sap-*")):
            artifacts = out_dir / f"rep{rep}" / sap.name
            for row in run_configs(sap, manifest, backend, modes=modes,
                                   artifacts_dir=artifacts):
                rec = row.model_dump()
                rec["rep"] = rep
                rec["candidates"] = {
                    p.name: _sha(p.read_text(encoding="utf-8"))
                    for p in sorted((artifacts / row.mode).glob("candidate*.java"))
                } if (artifacts / row.mode).is_dir() else {}
                records.append(rec)
                print(f"rep {rep}  {row.case_id.split('/')[-1]:<50s} {row.mode:<24s}"
                      f" {row.outcome:<24s} in={row.input_tokens} out={row.output_tokens}"
                      f" {row.runtime_ms}ms")
    with rows_path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")

    print("\n=== variance across repetitions (same case, same mode) ===")
    keyed: dict[tuple[str, str], list[dict]] = {}
    for rec in records:
        keyed.setdefault((rec["case_id"], rec["mode"]), []).append(rec)
    for (case_id, mode), recs in sorted(keyed.items()):
        if all(r["outcome"].startswith("failure:") for r in recs):
            continue
        for metric in ("input_tokens", "output_tokens", "runtime_ms"):
            values = [r[metric] for r in recs if isinstance(r[metric], int)]
            if not values:
                continue
            spread = (f"min={min(values)} max={max(values)} "
                      f"mean={statistics.mean(values):.0f}")
            if len(values) > 1:
                spread += f" stdev={statistics.stdev(values):.1f}"
            print(f"{case_id} [{mode}] {metric}: {spread}")
        hashes = {r["request_hash"] for r in recs}
        print(f"{case_id} [{mode}] request_hash: "
              f"{'IDENTICAL across reps' if len(hashes) == 1 else f'{len(hashes)} DISTINCT'}")
        cand = {json.dumps(r["candidates"], sort_keys=True) for r in recs
                if r.get("candidates")}
        if cand:
            print(f"{case_id} [{mode}] candidate code: "
                  f"{'IDENTICAL across reps' if len(cand) == 1 else f'{len(cand)} DISTINCT versions'}")
    print(f"\n[written] {rows_path}")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


if __name__ == "__main__":
    main()
