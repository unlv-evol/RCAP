"""Run the §15 four-config evaluation harness over every minted SAP (live LLM)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rcap.backend_ollama import OllamaBackend
from rcap.eval_harness import run_configs

OUT_ROOT = Path(__file__).resolve().parent.parent / "out"
SALP_OUT = Path("/home/adam/Documents/work/SALP/data/out/linkedinKafka-apacheKafka")


def main() -> None:
    backend = OllamaBackend()
    print(f"backend: {backend.name}/{backend.model} digest={backend.model_digest}")

    for pr_dir in sorted(SALP_OUT.glob("PR-*")):
        manifest = json.loads((pr_dir / "pr.json").read_text())
        out_path = OUT_ROOT / pr_dir.name / "eval-rows.jsonl"
        out_path.unlink(missing_ok=True)

        for sap in sorted(p.name for p in pr_dir.glob("sap-*")):
            print(f"\n===== {pr_dir.name} / {sap} =====", flush=True)
            rows = run_configs(
                pr_dir / sap, manifest, backend, out_path=out_path,
                artifacts_dir=OUT_ROOT / pr_dir.name / sap)
            for r in rows:
                print(f"  {r.mode:28s} nodes {r.nodes_before}->{r.nodes_after}"
                      f"  ph={r.placeholders}  prompt={r.prompt_chars}ch"
                      f"  outcome={r.outcome}  {r.runtime_ms}ms", flush=True)


if __name__ == "__main__":
    main()
