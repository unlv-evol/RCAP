"""Local LLM backend via ollama (RCAP ref section 11).

The backend sits behind RCAP's request interface: which model answers the
request is recorded in the generation configuration (theta), never part of
RCAP's contract. Deterministic settings where the runtime allows (temperature
0, fixed seed); the model digest is recorded so a weight change is always
distinguishable from a change in the engineered evidence.
"""

from __future__ import annotations

import json
import math
import os
import socket
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field

from rcap.request import AdaptationRequest

# RCAP_MODEL / RCAP_OLLAMA_URL select another model or server (e.g. on the cluster)
DEFAULT_MODEL = os.environ.get("RCAP_MODEL", "qwen3-coder:30b")
DEFAULT_URL = os.environ.get("RCAP_OLLAMA_URL", "http://localhost:11435")

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


# Chars per token EXPECTED on this backend (the median measured on the 30-PR
# evaluation set: Java prompts 4.8 and answers 5.0, Scala 4.4 / 4.4). An
# estimate, not a bound: it sizes the output budget and the up-front "prompt +
# answer cannot fit the window" refusal. The floor above stays the hard bound
# on the request alone. A wrong estimate cannot run away: the budget is passed
# to the runtime as `num_predict`, and an answer that exhausts it is reported
# as a capacity refusal, never as a model error.
CHARS_PER_TOKEN_EXPECTED_BY_LANG = {
    "c": 3.0, "h": 3.0,
    "java": 4.8, "scala": 4.4, "sc": 4.4,
}
CHARS_PER_TOKEN_EXPECTED_DEFAULT = 3.0


@dataclass
class GenerationLimits:
    """Everything about this backend + configuration that decides whether a
    request can be attempted, how long its answer may get and how long it may
    take. Model- and runtime-specific by nature: another model behind the same
    interface declares its own, and a unit refused under these limits is a
    CAPACITY refusal of this configuration, not a failure of the model.

    All of it is recorded in every generation record (`limits`)."""

    num_ctx: int = 16384
    # hard bound on the request alone, in characters; None = derive from
    # num_ctx and the per-language floor (admissible_chars)
    request_limit_chars: int | None = None
    chars_per_token_floor: dict[str, float] = field(
        default_factory=lambda: dict(CHARS_PER_TOKEN_FLOOR_BY_LANG))
    chars_per_token_floor_default: float = CHARS_PER_TOKEN_FLOOR_DEFAULT
    chars_per_token_expected: dict[str, float] = field(
        default_factory=lambda: dict(CHARS_PER_TOKEN_EXPECTED_BY_LANG))
    chars_per_token_expected_default: float = CHARS_PER_TOKEN_EXPECTED_DEFAULT
    # the answer may be this much longer than expected before generation is cut
    output_slack: float = 1.5
    output_floor_tokens: int = 512
    # generation timeout = base + expected prompt tokens / prompt rate
    #                           + output budget / minimum generation rate
    # RCAP_TIMEOUT_BASE_S: a slower local GPU/CPU can wait longer; never changes the answer, only when it is abandoned
    timeout_base_s: float = float(os.environ.get("RCAP_TIMEOUT_BASE_S", "180"))
    prompt_tokens_per_second: float = 100.0
    min_output_tokens_per_second: float = 8.0
    timeout_max_s: float = float(os.environ.get("RCAP_TIMEOUT_MAX_S", "3600"))

    def expected_cpt(self, language: str | None) -> float:
        return self.chars_per_token_expected.get(
            (language or "").lstrip(".").lower(), self.chars_per_token_expected_default)

    def floor_cpt(self, language: str | None) -> float:
        return self.chars_per_token_floor.get(
            (language or "").lstrip(".").lower(), self.chars_per_token_floor_default)

    def request_limit(self, language: str | None) -> int:
        if self.request_limit_chars is not None:
            return self.request_limit_chars
        return int(self.num_ctx * self.floor_cpt(language))

    def plan(self, prompt_chars: int, expected_answer_chars: int | None,
             language: str | None) -> dict[str, object]:
        """The budget of one request: what is expected, what is allowed."""
        cpt = self.expected_cpt(language)
        out: dict[str, object] = {
            "num_ctx": self.num_ctx, "request_chars": prompt_chars,
            "request_limit_chars": self.request_limit(language),
            "chars_per_token_floor": self.floor_cpt(language),
            "chars_per_token_expected": cpt,
            "expected_prompt_tokens": math.ceil(prompt_chars / cpt),
        }
        if expected_answer_chars is None:
            # nothing known about the answer: the window minus the prompt
            budget = max(self.output_floor_tokens, self.num_ctx - out["expected_prompt_tokens"])
            out.update(expected_answer_chars=None, expected_output_tokens=None)
        else:
            exp_out = math.ceil(expected_answer_chars / cpt)
            budget = max(self.output_floor_tokens, math.ceil(exp_out * self.output_slack))
            out.update(expected_answer_chars=expected_answer_chars, expected_output_tokens=exp_out,
                       fits_window=out["expected_prompt_tokens"] + exp_out <= self.num_ctx)
        budget = min(budget, self.num_ctx)
        timeout = (self.timeout_base_s + out["expected_prompt_tokens"] / self.prompt_tokens_per_second
                   + budget / self.min_output_tokens_per_second)
        out.update(output_budget_tokens=budget, timeout_s=round(min(timeout, self.timeout_max_s), 1))
        return out

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


class OllamaBackend:
    def __init__(self, model: str = DEFAULT_MODEL, url: str = DEFAULT_URL,
                 *, num_ctx: int | None = None, timeout: int = 600,
                 request_limit_chars: int | None = None,
                 limits: GenerationLimits | None = None):
        self.name = "ollama"
        self.model = model
        self.url = url
        # RCAP_NUM_CTX sets the window for a whole run (default 16384, as before)
        num_ctx = num_ctx if num_ctx is not None else int(os.environ.get("RCAP_NUM_CTX", "16384"))
        self.num_ctx = num_ctx
        # `timeout` is the fallback for calls without a budget (metadata calls,
        # requests of unknown answer size); budgeted calls use limits.plan().
        self.timeout = timeout
        self.limits = limits or GenerationLimits(num_ctx=num_ctx,
                                                 request_limit_chars=request_limit_chars)
        self.accepts_budget = True
        self.params = {"temperature": 0, "seed": 12535, "num_ctx": num_ctx}
        # This is the language-agnostic fallback; `generate` recomputes it from
        # the request's own language, which is the tighter and correct bound.
        self.request_limit_chars = (request_limit_chars if request_limit_chars is not None
                                    else admissible_chars(num_ctx))
        # An operator-supplied limit is honoured verbatim; a derived one is
        # refined per language at generation time.
        self.request_limit_chars_is_explicit = request_limit_chars is not None
        # Thinking models (qwen3.5+, ...) would spend the shared window on hidden
        # reasoning first; ask for the answer directly unless RCAP_THINK=1.
        # Models without the capability get no `think` field, as before.
        self.think: bool | None = None
        self.model_digest = self._digest()

    def _api(self, path: str, payload: dict | None = None, timeout: float | None = None) -> dict:
        req = urllib.request.Request(
            f"{self.url}{path}",
            data=json.dumps(payload).encode() if payload is not None else None,
            headers={"Content-Type": "application/json"},
            method="POST" if payload is not None else "GET",
        )
        with urllib.request.urlopen(req, timeout=timeout or self.timeout) as resp:
            return json.loads(resp.read())

    def _digest(self) -> str | None:
        try:
            data = self._api("/api/show", {"model": self.model})
            details = data.get("details", {})
            digest = f"{details.get('family')}/{details.get('parameter_size')}/" \
                     f"{details.get('quantization_level')}"
            if "thinking" in (data.get("capabilities") or []):
                self.think = os.environ.get("RCAP_THINK") == "1"
                digest += "/think=" + ("on" if self.think else "off")
            return digest
        except (urllib.error.URLError, OSError, KeyError):
            return None

    def generate(self, request: AdaptationRequest, *, num_predict: int | None = None,
                 timeout: float | None = None) -> str:
        """`num_predict` caps the answer in tokens (the runtime stops there and
        says so: `last_done_reason == "length"`); `timeout` is this call's
        client timeout in seconds. Neither changes what the model writes up to
        the cap, so a cached answer of the same prompt stays valid."""
        self.last_usage: dict[str, int] | None = None
        self.last_done_reason: str | None = None
        self.last_timing: dict[str, float] | None = None
        options = dict(self.params)
        if num_predict:
            options["num_predict"] = int(num_predict)
        try:
            data = self._api("/api/chat", {
                "model": self.model,
                # dialogue mode: the earlier turns, then this one
                "messages": [*(getattr(request, "history", None) or []),
                             {"role": "user", "content": request.prompt}],
                "stream": False,
                "options": options,
                **({"think": self.think} if self.think is not None else {}),
            }, timeout=timeout)
        except (TimeoutError, socket.timeout) as exc:
            raise TimeoutError(f"generation timed out after {timeout or self.timeout:.0f}s") from exc
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                raise TimeoutError(
                    f"generation timed out after {timeout or self.timeout:.0f}s") from exc
            raise
        # Runtime-reported token counts (section 15). prompt_eval_count is
        # what the runtime actually evaluated — a warm prompt-prefix cache can
        # make it lower than the full prompt length; recorded as reported.
        usage = {}
        if isinstance(data.get("prompt_eval_count"), int):
            usage["input_tokens"] = data["prompt_eval_count"]
        if isinstance(data.get("eval_count"), int):
            usage["output_tokens"] = data["eval_count"]
        self.last_usage = usage or None
        self.last_done_reason = data.get("done_reason")
        self.last_timing = {k[:-len("_duration")] + "_s": round(data[k] / 1e9, 2)
                            for k in ("total_duration", "load_duration", "prompt_eval_duration",
                                      "eval_duration") if isinstance(data.get(k), int)} or None
        return data.get("message", {}).get("content", "")
