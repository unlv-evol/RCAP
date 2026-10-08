"""Who produced a result row: model, settings and code version.

Every result row a driver writes carries this record, so that when a different
model is tried the rows say so themselves and nothing has to be inferred from
file dates. `model_digest` is the registry digest of the weights (what
`ollama list` shows), not the family/size/quantization string that
`OllamaBackend.model_digest` has always held (kept there unchanged, because it
is part of stored generation configs)."""

from __future__ import annotations

import json
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]


def code_version(repo: Path | None = None) -> dict[str, object]:
    repo = repo or _REPO

    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", "-C", str(repo), *args], capture_output=True,
                                  text=True, timeout=20).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return ""
    return {"rcap_commit": git("rev-parse", "HEAD") or None,
            "rcap_branch": git("rev-parse", "--abbrev-ref", "HEAD") or None,
            "rcap_dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def registry_digest(url: str, model: str, timeout: int = 20) -> str | None:
    """The digest `ollama list` reports for `model`, or None."""
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=timeout) as resp:
            for m in json.loads(resp.read()).get("models", []):
                if m.get("name") == model or m.get("model") == model:
                    return m.get("digest")
    except (urllib.error.URLError, OSError, ValueError):
        return None
    return None


def run_provenance(backend: object | None = None, **extra: object) -> dict[str, object]:
    """The record for one driver run. `backend` may be None (a driver that
    calls no model, e.g. a re-validation of stored candidates)."""
    out: dict[str, object] = {"backend": None, "model": None}
    if backend is not None:
        inner = getattr(backend, "inner", backend)          # a caching wrapper
        params = dict(getattr(backend, "params", None) or {})
        model = getattr(backend, "model", None)
        url = getattr(inner, "url", None)
        out.update({
            "backend": getattr(backend, "name", None), "model": model,
            "model_tag": model.split(":", 1)[1] if isinstance(model, str) and ":" in model else None,
            "model_digest": registry_digest(url, model) if url and model else None,
            "model_details": getattr(backend, "model_digest", None),
            "num_ctx": params.get("num_ctx"), "temperature": params.get("temperature"),
            "seed": params.get("seed"), "params": params, "url": url,
        })
        limits = getattr(backend, "limits", None)
        if limits is not None and hasattr(limits, "as_dict"):
            # the capacity configuration: what decides a capacity refusal
            out["limits"] = limits.as_dict()
    out.update(code_version())
    out.update(extra)
    return out
