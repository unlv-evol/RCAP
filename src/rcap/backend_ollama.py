"""Local LLM backend via ollama (RCAP ref section 11).

The backend sits behind RCAP's request interface: which model answers the
request is recorded in the generation configuration (theta), never part of
RCAP's contract. Deterministic settings where the runtime allows (temperature
0, fixed seed); the model digest is recorded so a weight change is always
distinguishable from a change in the engineered evidence.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from rcap.request import AdaptationRequest

DEFAULT_MODEL = "qwen3-coder:30b"
DEFAULT_URL = "http://localhost:11435"


class OllamaBackend:
    def __init__(self, model: str = DEFAULT_MODEL, url: str = DEFAULT_URL,
                 *, num_ctx: int = 16384, timeout: int = 600):
        self.name = "ollama"
        self.model = model
        self.url = url
        self.num_ctx = num_ctx
        self.timeout = timeout
        self.params = {"temperature": 0, "seed": 12535, "num_ctx": num_ctx}
        self.model_digest = self._digest()

    def _api(self, path: str, payload: dict | None = None) -> dict:
        req = urllib.request.Request(
            f"{self.url}{path}",
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Content-Type": "application/json"},
            method="POST" if payload is not None else "GET",
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def _digest(self) -> str | None:
        try:
            data = self._api("/api/show", {"model": self.model})
            details = data.get("details", {})
            return f"{details.get('family')}/{details.get('parameter_size')}/" \
                   f"{details.get('quantization_level')}"
        except (urllib.error.URLError, OSError, KeyError):
            return None

    def generate(self, request: AdaptationRequest) -> str:
        data = self._api("/api/chat", {
            "model": self.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "stream": False,
            "options": self.params,
        })
        return data.get("message", {}).get("content", "")
