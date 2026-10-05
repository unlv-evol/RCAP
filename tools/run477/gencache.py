"""Prompt-hash generation cache shared by the drivers of this branch.

Generation is deterministic (temperature 0, fixed seed), so an identical prompt may be answered
from (a) candidates stored by the 477 run (`replay`), (b) this branch's own cache directory or
(c) read-only cache directories of earlier rounds. `sources` counts where each answer came from.
Only the first cache directory is ever written.
"""
import collections, glob, hashlib, json
from pathlib import Path


def sha(t): return hashlib.sha256(t.encode()).hexdigest()


class CachedBackend:
    def __init__(self, inner, cache_dir, ro_dirs=(), regenerate=()):
        self.inner = inner; self.dir = Path(cache_dir); self.dir.mkdir(parents=True, exist_ok=True)
        self.ro = [Path(d) for d in ro_dirs]; self.regenerate = set(regenerate)
        self.replay = {}; self.expected = []; self.sources = collections.Counter()
        for k in ("name", "model", "params", "model_digest", "request_limit_chars", "request_limit_chars_is_explicit"):
            setattr(self, k, getattr(inner, k))
        self.last_usage = None; self.last_source = None

    def _cached(self, h):
        for d in [self.dir, *self.ro]:
            f = d / f"{h}.json"
            if f.exists(): return json.loads(f.read_text()), d is self.dir
        return None, False

    def generate(self, request):
        h = sha(request.prompt); d, own = self._cached(h)
        if self.expected:
            # outcomes the 477 run recorded for this SAP's units, in order (it kept no prompt hash
            # for failures, so identity is checked by prompt length)
            outcome, chars = self.expected.pop(0)
            if outcome != "completion" and outcome not in self.regenerate and chars == len(request.prompt) \
                    and h not in self.replay and d is None:
                self.sources["stored-477-failure"] += 1
                raise RuntimeError("stored-477 outcome: " + outcome)
        rp = self.replay.get(h)
        if rp and rp.get("failed") and rp["failed"] not in self.regenerate:
            self.sources["stored-477-failure"] += 1
            raise RuntimeError("stored-477 outcome: " + rp["failed"])
        if rp and not rp.get("failed"):
            raw, usage, src = rp["raw"], rp["usage"], "stored-477"
        elif d is not None:
            if d.get("error"):
                self.sources["cache-error"] += 1; raise RuntimeError(d["error"])
            raw, usage, src = d["raw"], d.get("usage"), "cache" if own else "cache-earlier-round"
        else:
            f = self.dir / f"{h}.json"
            try: raw = self.inner.generate(request)
            except Exception as exc:      # a timeout costs 10 minutes: pay for it once
                f.write_text(json.dumps({"error": f"{type(exc).__name__}: {exc}"[:300], "case_id": request.case_id}))
                self.sources["live-error"] += 1; raise
            usage = self.inner.last_usage; src = "live"
            f.write_text(json.dumps({"raw": raw, "usage": usage, "model": self.model, "params": self.params,
                                     "template_id": request.template_id, "case_id": request.case_id}))
        self.last_usage = usage or None; self.last_source = src; self.sources[src] += 1
        return raw


def load_replay(candidates_dir, pr):
    """Stored 477 answers of one PR by prompt hash (completions as raw text, failures as outcome)."""
    out = {}
    for f in glob.glob(f"{candidates_dir}/*__{pr}__*.json"):
        r = json.loads(Path(f).read_text())
        for u in r.get("units", []):
            if u.get("outcome") == "completion" and u.get("candidate_raw") and u.get("prompt_sha256"):
                out[u["prompt_sha256"]] = {"raw": "```\n" + u["candidate_raw"] + "```", "usage": u.get("usage") or None}
            elif u.get("prompt_sha256") and u.get("outcome") in ("placeholder_violation", "unparseable", "no_output", "limits_exceeded", "backend_error"):
                out[u["prompt_sha256"]] = {"failed": u["outcome"]}
    return out
