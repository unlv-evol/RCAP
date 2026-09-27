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

# Lowest chars-per-token MEASURED per language on this backend, used to convert
# the token window into a character bound. A floor, not an average: the bound has
# to under-estimate how many tokens a prompt costs, or it admits prompts that
# cannot fit and the runtime truncates them silently.
#
# Measured (n=295 C prompts, n=25 Scala): C 2.66-3.79, Scala 3.66-4.93. One
# global floor cannot serve both -- 3.2 is unsafe for C (a 52k-char C prompt is
# ~19.7k tokens against a 16,384 window) while 2.6 needlessly refuses Scala
# prompts that fit comfortably. So it is per language, defaulting to the most
# conservative value for anything unmeasured.
CHARS_PER_TOKEN_FLOOR_BY_LANG = {
    "c": 2.6, "h": 2.6,
    "java": 3.6, "scala": 3.6, "sc": 3.6,
}
CHARS_PER_TOKEN_FLOOR_DEFAULT = 2.6


def chars_per_token_floor(language: str | None) -> float:
    return CHARS_PER_TOKEN_FLOOR_BY_LANG.get(
        (language or "").lstrip(".").lower(), CHARS_PER_TOKEN_FLOOR_DEFAULT)


def admissible_chars(num_ctx: int, language: str | None = None) -> int:
    """Longest prompt in characters that provably fits `num_ctx` tokens."""
    return int(num_ctx * chars_per_token_floor(language))
# Above this, the reported token count is too small for the prompt's length to be
# real code in any language we handle (C peaks at 3.4, Scala/Java at 4.9).
CHARS_PER_TOKEN_IMPLAUSIBLE = 6.0


class OllamaBackend:
    def __init__(self, model: str = DEFAULT_MODEL, url: str = DEFAULT_URL,
                 *, num_ctx: int = 16384, timeout: int = 600,
                 request_limit_chars: int | None = None):
        self.name = "ollama"
        self.model = model
        self.url = url
        self.num_ctx = num_ctx
        self.timeout = timeout
        self.params = {"temperature": 0, "seed": 12535, "num_ctx": num_ctx}
        # This is the language-agnostic fallback; `generate` recomputes it from
        # the request's own language, which is the tighter and correct bound.
        self.request_limit_chars = (request_limit_chars if request_limit_chars is not None
                                    else admissible_chars(num_ctx))
        # An operator-supplied limit is honoured verbatim; a derived one is
        # refined per language at generation time.
        self.request_limit_chars_is_explicit = request_limit_chars is not None
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
        self.last_usage: dict[str, int] | None = None
        data = self._api("/api/chat", {
            "model": self.model,
            "messages": [{"role": "user", "content": request.prompt}],
            "stream": False,
            "options": self.params,
        })
        # Runtime-reported token counts (section 15). prompt_eval_count is
        # what the runtime actually evaluated — a warm prompt-prefix cache can
        # make it lower than the full prompt length; recorded as reported.
        usage = {}
        if isinstance(data.get("prompt_eval_count"), int):
            usage["input_tokens"] = data["prompt_eval_count"]
        if isinstance(data.get("eval_count"), int):
            usage["output_tokens"] = data["eval_count"]
        self.last_usage = usage or None
        return data.get("message", {}).get("content", "")
