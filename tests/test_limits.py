"""Section-11 limits_exceeded: a request or context exceeding backend limits
is a distinct recorded outcome, never a silently truncated completion."""

import typing

from rcap.eval_harness import run_configs
from rcap.generate import StubBackend, generate

CANDIDATE = "```java\nvoid validate(Config c) { }\n```"


class CountingBackend(StubBackend):
    """Stub that counts invocations and declares an acceptance limit."""

    def __init__(self, response, *, limit=None, usage=None):
        super().__init__(response)
        if limit is not None:
            self.request_limit_chars = limit
        self._usage = usage
        self.calls = 0

    def generate(self, request):
        self.calls += 1
        if self._usage is not None:
            self.last_usage = dict(self._usage)
        return super().generate(request)


def _request(sap_dir, pr_manifest):
    from rcap.context import build_context
    from rcap.intake import load_case
    from rcap.materialize import materialize
    from rcap.reduction_program import reduce_program
    from rcap.reduction_semantic import reduce_semantic
    from rcap.request import synthesize
    from rcap.selection import select

    case = load_case(sap_dir, pr_manifest=pr_manifest)
    red = reduce_semantic(case, select(case))
    mat = materialize(case, red)
    prog = reduce_program(case, red, mat)
    return synthesize(build_context(case, red, mat, prog))


def test_oversized_request_is_refused_before_invocation(sap_dir, pr_manifest):
    backend = CountingBackend(CANDIDATE, limit=10)
    req = _request(sap_dir, pr_manifest)
    gen = generate(req, backend)
    assert gen.outcome == "limits_exceeded"
    assert backend.calls == 0, "the backend must never see an oversized request"
    assert any("declares an acceptance limit" in d for d in gen.diagnostics)
    assert gen.generation_config["request_limit_chars"] == 10
    assert gen.candidate is None


def test_runtime_window_saturation_is_limits_exceeded(sap_dir, pr_manifest):
    """ollama truncates silently at num_ctx; a reported input-token count at
    the window is runtime evidence of truncation, so no completion may be
    recorded (the model saw a different prompt than the request)."""

    class SaturatedBackend(CountingBackend):
        params: typing.ClassVar[dict] = {"temperature": 0, "num_ctx": 128}

    backend = SaturatedBackend(CANDIDATE, usage={"input_tokens": 128, "output_tokens": 5})
    gen = generate(_request(sap_dir, pr_manifest), backend)
    assert gen.outcome == "limits_exceeded"
    assert backend.calls == 1
    assert any("saturating the declared context window" in d for d in gen.diagnostics)
    assert gen.usage["input_tokens"] == 128, "the runtime evidence itself is kept"


def test_within_limits_completes_unchanged(sap_dir, pr_manifest):
    backend = CountingBackend(CANDIDATE, limit=10_000_000,
                              usage={"input_tokens": 500, "output_tokens": 5})
    gen = generate(_request(sap_dir, pr_manifest), backend)
    assert gen.outcome == "completion"


def test_harness_records_limits_exceeded_rows(sap_dir, pr_manifest):
    rows = run_configs(sap_dir, pr_manifest, CountingBackend(CANDIDATE, limit=10))
    assert all(r.outcome == "limits_exceeded" for r in rows)
    # Not a pipeline failure: the sizes measured up to the refusal are real.
    assert all(r.prompt_chars > 10 for r in rows)


def test_ollama_backend_declares_default_limit():
    from rcap.backend_ollama import OllamaBackend

    backend = OllamaBackend(url="http://127.0.0.1:9", num_ctx=16384)  # no server
    assert backend.request_limit_chars == 16384 * 4
    assert OllamaBackend(url="http://127.0.0.1:9",
                         request_limit_chars=99).request_limit_chars == 99
