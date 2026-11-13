"""Every request RCAP would send for every SAP of a SALP out root, without a model.

Runs the real rcap.pipeline.run_case (so region requests, target narrowing and the echo fix are exactly as
in a live run) with a backend that records each request and then fails it (a region request is answered with its own\ntext instead, so the unit's later regions are reached too). One row per request sent, plus
one row per unit the pipeline refused before any request (e.g. over the request limit) and per SAP refused.

    python dump_rcap_prompts.py <SALP out root> <out.jsonl> [--mode full_rcap|semantic_reduction_only]
                                [--num-ctx 32768] [--pairs a,b]

--mode semantic_reduction_only is "RCAP without program reduction": no placeholders on any payload.
(Semantic reduction has no admissible edge on these SAPs, so full_rcap is in effect program reduction only.)
"""
import argparse, hashlib, json, os, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from rcap.backend_ollama import OllamaBackend  # noqa: E402
from rcap.config import EvaluationMode, ExecutionConfig  # noqa: E402
from rcap.pipeline import run_case  # noqa: E402


class Recorder:
    """Looks like an OllamaBackend of the given num_ctx; records the request and refuses to answer."""

    def __init__(self, num_ctx):
        inner = OllamaBackend(url="http://127.0.0.1:9", num_ctx=num_ctx)   # port 9: never a model
        for k in ("name", "model", "params", "request_limit_chars", "request_limit_chars_is_explicit",
                  "limits", "accepts_budget", "num_ctx"):
            setattr(self, k, getattr(inner, k))
        self.model_digest = "dry-run"; self.last_usage = None; self.last_done_reason = None
        self.last_timing = None; self.last_source = None; self.sent = []

    def generate(self, request, **budget):
        self.sent.append((request, budget))
        region = getattr(request, "region", None)
        if region and region.get("text") is not None:
            # a unit sent as several region requests stops at the first failed one, so answer each region
            # with its own text (an echo) and every region request of the unit is reached and recorded
            return f"```{request.language}\n{region['text']}\n```"
        raise RuntimeError("dry-run: prompt recorded, no model")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root"); ap.add_argument("out")
    ap.add_argument("--mode", default="full_rcap", choices=[m.value for m in EvaluationMode])
    ap.add_argument("--num-ctx", type=int, default=32768)
    ap.add_argument("--pairs")
    a = ap.parse_args()
    cfg = ExecutionConfig(mode=EvaluationMode(a.mode))
    pairs = set(a.pairs.split(",")) if a.pairs else None
    n = 0; t0 = time.time()
    with open(a.out, "w") as out:
        for pair in sorted(p for p in Path(a.root).iterdir() if p.is_dir()):
            if pairs and pair.name not in pairs:
                continue
            for pr_dir in sorted(pair.glob("PR-*"), key=lambda p: int(p.name[3:])):
                man = pr_dir / "pr.json"
                if not man.is_file():
                    continue
                manifest = json.loads(man.read_text())
                for sap in sorted(pr_dir.glob("sap-*")):
                    be = Recorder(a.num_ctx)
                    base = {"pair": pair.name, "pr": pr_dir.name, "sap": sap.name, "mode": a.mode, "num_ctx": a.num_ctx}
                    try:
                        res = run_case(sap, manifest, be, cfg)
                    except Exception as exc:
                        out.write(json.dumps({**base, "kind": "sap_refused", "outcome": f"REFUSED:{type(exc).__name__}",
                                              "stage": getattr(exc, "stage", None),
                                              "diag": str((getattr(exc, "diagnostics", None) or [exc])[0])[:300]}) + "\n")
                        continue
                    sent = {id(r): b for r, b in be.sent}
                    for u in res.units:
                        uid = getattr(u.unit, "entity", None) or getattr(u.unit, "unit_id", None)
                        reqs = list(getattr(u, "region_requests", None) or []) or [u.request]
                        reqs = [r for r in reqs if r is not None]
                        if not any(id(r) in sent for r in reqs):
                            # refused before the backend (limits, packaging): no request left the pipeline
                            out.write(json.dumps({**base, "kind": "unit_refused", "unit": str(uid),
                                                  "outcome": u.generation.outcome,
                                                  "diag": str((u.generation.diagnostics or [""])[0])[:300],
                                                  "prompt_chars": len(u.request.prompt) if u.request else None}) + "\n")
                            continue
                        for i, r in enumerate(reqs):
                            if id(r) not in sent:
                                continue
                            out.write(json.dumps({**base, "kind": "request", "unit": str(uid), "part": i, "parts": len(reqs),
                                                  "answer_kind": r.answer_kind, "language": r.language,
                                                  "budget": sent[id(r)], "prompt_chars": len(r.prompt),
                                                  "prompt_sha256": hashlib.sha256(r.prompt.encode()).hexdigest(),
                                                  "prompt": r.prompt}) + "\n")
                    # anything the backend saw that no unit claims (retries etc.) is still recorded
                    claimed = {id(r) for u in res.units for r in ([*(getattr(u, "region_requests", None) or []), u.request])}
                    for r, b in be.sent:
                        if id(r) not in claimed:
                            out.write(json.dumps({**base, "kind": "request", "unit": "?unclaimed", "part": 0, "parts": 1,
                                                  "answer_kind": r.answer_kind, "language": r.language, "budget": b,
                                                  "prompt_chars": len(r.prompt),
                                                  "prompt_sha256": hashlib.sha256(r.prompt.encode()).hexdigest(),
                                                  "prompt": r.prompt}) + "\n")
                    n += 1
                    if n % 100 == 0:
                        print(f"{n} SAPs, {time.time() - t0:.0f}s", flush=True)
    print(f"done: {n} SAPs, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
