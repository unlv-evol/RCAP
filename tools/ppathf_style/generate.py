"""PPatHF-style treatment, generation (roadmap item 8): send each eligible prompt of build_inputs.py to the model
with RCAP's settings (temperature 0, seed 12535, num_ctx 16384, k=1, thinking off as in RCAP's backend), take
the answer and put PPatHF's removed pieces back (Reducer.do_recovering, Java profile).

The user message is PPatHF's prompt byte for byte. A zero-shot chat model needs to be told the answer format
(PPatHF's model was fine-tuned on it), so that, and the placeholder rule when the prompt has placeholders, go
in a system message; both are the wording of our Vim->Neovim replication harness.

Answers are streamed and the request ends as soon as the first complete code block has arrived. recover() only
ever reads that first block, so the score cannot change; a model that keeps writing after it (starcoder2) simply
stops wasting GPU time. Such records have done_reason "closed_block" and output_tokens counted from the stream.

Resumable: units already in the output are skipped, except backend errors, which are retried (the old rows are
moved to <out>.backend-errors). --no-recover only stores the model's answers (the cluster
run, which needs nothing beyond the standard library); --recover-only then fills in `candidate` for such a file.

    python generate.py <inputs.jsonl> <out.jsonl> [--prs PR-1,PR-2] [--limit N] [--no-recover | --recover-only]
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

OLLAMA_BASE = os.environ.get("RCAP_OLLAMA_URL", "http://127.0.0.1:11435")
MODEL = os.environ.get("RCAP_MODEL", "qwen3-coder:30b")
NUM_CTX = 16384
TIMEOUT_S = 600
MAX_OUTPUT_TOKENS = 4096  # above every answer of the local qwen3-coder run; stops a run-on answer that never closes a block

FORMAT_NOTE = "Return ONLY the complete function after, in one ```java code block, with no commentary."
PLACEHOLDER_NOTE = ("Placeholder comments of the form /* placeholder_N */ stand for elided code: keep each one exactly "
                    "where it is, character for character, and do not invent code for it.")
FENCE_RE = re.compile(r"```[a-zA-Z]*[ \t]*\n(.*?)```", re.DOTALL)


def post(path, body, timeout):
    req = urllib.request.Request(OLLAMA_BASE + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


class PartialAnswer(TimeoutError):
    """Timed out while streaming; carries what the model had written so the record shows why."""
    def __init__(self, msg, text):
        super().__init__(msg); self.text = text


def chat_until_block(body, timeout):
    """Stream a chat answer; stop at the first closed code block or at `timeout` seconds in total. Returns the same
    fields a non-streamed answer gives. urlopen's timeout only limits each read, hence the explicit clock."""
    req = urllib.request.Request(OLLAMA_BASE + "/api/chat", data=json.dumps({**body, "stream": True}).encode(),
                                 headers={"Content-Type": "application/json"})
    text, chunks, t0 = "", 0, time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for line in resp:
            if not line.strip():
                continue
            d = json.loads(line)
            text += d.get("message", {}).get("content", "")
            chunks += 1
            if d.get("done"):
                return {"text": text, "input_tokens": d.get("prompt_eval_count"), "output_tokens": d.get("eval_count"),
                        "done_reason": d.get("done_reason")}
            if "```" in text and FENCE_RE.search(text):
                return {"text": text, "input_tokens": None, "output_tokens": chunks, "done_reason": "closed_block"}
            if time.time() - t0 > timeout:
                raise PartialAnswer(f"no closed code block after {timeout} s ({chunks} chunks)", text)
    raise OSError("stream ended without a final record")


def model_think():
    """RCAP's backend rule: a model that can think gets think=False unless RCAP_THINK=1; others get no field."""
    caps = post("/api/show", {"model": MODEL}, 60).get("capabilities") or []
    return (os.environ.get("RCAP_THINK") == "1") if "thinking" in caps else None


def strip_trailing_prose(text):
    """An unfenced answer up to its last line starting with '}' (PPatHF's own completeness mark): models append
    an explanation, which is not code. No such line: the whole answer."""
    lines = text.strip().splitlines()
    last = max((i for i, l in enumerate(lines) if l.lstrip().startswith("}")), default=len(lines) - 1)
    return "\n".join(lines[:last + 1]) + "\n"


def recover(rec, removed, reducer):
    """Put PPatHF's removed pieces back into the fenced answer; sets outcome, candidate, placeholders_lost."""
    text = rec.get("response") or ""
    m = FENCE_RE.search(text)
    if m is not None:
        reduced = m.group(1)
    elif text.strip() and "```" not in text and rec.get("done_reason") == "stop":
        reduced = strip_trailing_prose(text)  # RCAP's rule: the first fenced block, else the whole answer
        rec["unfenced"] = True
    else:
        rec.update(outcome="no_output"); return
    rec["candidate_reduced"] = reduced
    if reducer is None:
        rec.update(outcome="answered"); return
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        try:
            recovered = reducer.do_recovering(reduced, removed)
        except Exception as e:  # recorded, never dropped
            recovered = None
            rec["recover_error"] = f"{type(e).__name__}: {e}"[:300]
    rec.update(outcome="completion" if recovered is not None else "recover_failed", candidate=recovered,
               placeholders_lost=sum("not found" in str(x.message) for x in w))


def make_reducer():
    import reducerConfig
    reducerConfig.set_language("java")
    from reducer import Reducer
    return Reducer(tolerant=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs"); ap.add_argument("out")
    ap.add_argument("--prs", help="comma-separated PR ids (default: all)")
    ap.add_argument("--limit", type=int)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--no-recover", action="store_true"); g.add_argument("--recover-only", action="store_true")
    a = ap.parse_args()
    if a.recover_only:
        removed = {r["pr"] + "/" + r["unit"]: r["removed_pieces_target"]
                   for r in map(json.loads, open(a.inputs)) if r.get("eligible")}
        reducer = make_reducer()
        recs = [json.loads(l) for l in open(a.out) if l.strip()]
        for rec in recs:
            if rec.get("outcome") in ("answered", "no_output") or rec.get("unfenced"):
                recover(rec, removed[rec["unit_id"]], reducer)
        Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in recs))
        print(f"recovered {len(recs)} records", flush=True)
        return
    only = set(a.prs.split(",")) if a.prs else None
    done = set()
    if Path(a.out).exists():
        recs = [json.loads(l) for l in open(a.out) if l.strip()]
        errs = [r for r in recs if r.get("outcome") == "backend_error"]
        if errs:  # retried below; kept aside, never lost
            with open(a.out + ".backend-errors", "a") as f:
                f.writelines(json.dumps(r) + "\n" for r in errs)
            recs = [r for r in recs if r.get("outcome") != "backend_error"]
            Path(a.out).write_text("".join(json.dumps(r) + "\n" for r in recs))
        done = {r["unit_id"] for r in recs}
    rows = [r for r in map(json.loads, open(a.inputs)) if r.get("eligible") and (not only or r["pr"] in only)]
    if a.limit:
        rows = rows[:a.limit]
    think = model_think()
    digest = post("/api/show", {"model": MODEL}, 60).get("digest")
    reducer = None if a.no_recover else make_reducer()
    print(f"{MODEL} think={think} {len(rows)} units, {len(done & {r['pr'] + '/' + r['unit'] for r in rows})} done",
          flush=True)
    with open(a.out, "a") as out:
        for i, r in enumerate(rows, 1):
            uid = r["pr"] + "/" + r["unit"]
            if uid in done:
                continue
            system = FORMAT_NOTE + ("\n" + PLACEHOLDER_NOTE if r["placeholders"] else "")
            body = {"model": MODEL,
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": r["prompt"]}],
                    **({"think": think} if think is not None else {}),
                    "options": {"temperature": 0, "seed": 12535, "num_ctx": NUM_CTX, "num_predict": MAX_OUTPUT_TOKENS}}
            rec = {"unit_id": uid, "pr": r["pr"], "sap": r["sap"], "fn": r["fn"], "model": MODEL, "model_digest": digest,
                   "think": think, "prompt_sha256": r["prompt_sha256"], "system": system,
                   "placeholders": r["placeholders"]}
            t0 = time.time()
            try:
                resp = chat_until_block(body, TIMEOUT_S)
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                rec.update(outcome="backend_error", error=f"{type(e).__name__}: {e}"[:300],
                           partial=getattr(e, "text", None))
            else:
                rec.update(input_tokens=resp["input_tokens"], output_tokens=resp["output_tokens"],
                           done_reason=resp["done_reason"], response=resp["text"])
                recover(rec, r["removed_pieces_target"], reducer)
            rec["seconds"] = round(time.time() - t0, 1)
            out.write(json.dumps(rec) + "\n"); out.flush()
            print(f"[{i}/{len(rows)}] {uid}: {rec['outcome']} {rec['seconds']}s", flush=True)


if __name__ == "__main__":
    main()
