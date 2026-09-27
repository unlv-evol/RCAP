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


def _ctx_backend(num_ctx, in_tok):
    class Ctx(CountingBackend):
        params: typing.ClassVar[dict] = {"temperature": 0, "num_ctx": num_ctx}

    return Ctx(CANDIDATE, usage={"input_tokens": in_tok, "output_tokens": 5})


def test_runtime_window_saturation_is_limits_exceeded(sap_dir, pr_manifest):
    """A reported input-token count at the window is runtime evidence of
    truncation, so no completion may be recorded (the model saw a different
    prompt than the request).

    The window has to be large enough for the request to clear the pre-flight
    bound, or this exercises that check instead of the runtime one."""
    num_ctx = 16384
    backend = _ctx_backend(num_ctx, num_ctx)
    gen = generate(_request(sap_dir, pr_manifest), backend)
    assert gen.outcome == "limits_exceeded"
    assert backend.calls == 1, "the runtime check happens after invocation"
    assert any("truncated by the runtime" in d for d in gen.diagnostics)
    assert gen.usage["input_tokens"] == num_ctx, "the runtime evidence itself is kept"


def test_truncation_pinned_token_count_is_detected(sap_dir, pr_manifest):
    """The real signature. When the runtime truncates it does NOT report a
    saturated count -- it reports one pinned at num_ctx // 2 + 2, far below the
    window, so a `>= num_ctx` test can never fire. This is the case that went
    undetected in ~1,550 records."""
    num_ctx = 16384
    backend = _ctx_backend(num_ctx, num_ctx // 2 + 2)
    gen = generate(_request(sap_dir, pr_manifest), backend)
    assert gen.outcome == "limits_exceeded"
    assert any("pinned at the truncation value" in d for d in gen.diagnostics)


def test_implausible_chars_per_token_is_detected(sap_dir, pr_manifest):
    """A pinned count at some OTHER value still shows up as a chars/token ratio no
    real code reaches -- provided the count is near the window, since truncation
    is impossible far below it (see the precondition test below)."""
    req = _request(sap_dir, pr_manifest)
    num_ctx = max(4, len(req.prompt) // 12)      # ~12 chars/token fills this window
    backend = _ctx_backend(num_ctx, num_ctx - 1)
    gen = generate(req, backend)
    assert gen.outcome == "limits_exceeded"


def test_ordinary_ratio_is_not_flagged(sap_dir, pr_manifest):
    """Regression guard. The ratio test must stay language-independent: Scala and
    Java run 3.7-4.9 chars/token where C peaks at 3.4, so a threshold calibrated
    on C alone reports ordinary Scala prompts as truncated."""
    req = _request(sap_dir, pr_manifest)
    for chars_per_token in (3.4, 4.5, 4.9):
        in_tok = max(1, int(len(req.prompt) / chars_per_token))
        gen = generate(req, _ctx_backend(16384, in_tok))
        assert gen.outcome == "completion", (
            f"{chars_per_token} chars/token is normal code, not truncation")


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
    from rcap.backend_ollama import OllamaBackend, admissible_chars, chars_per_token_floor

    backend = OllamaBackend(url="http://127.0.0.1:9", num_ctx=16384)  # no server
    assert backend.request_limit_chars == admissible_chars(16384)
    # The bound must UNDER-estimate token cost, or it admits prompts that cannot
    # fit and are then truncated silently. At the old 4 chars/token it was 65,536
    # chars ~= 20.5k tokens against a 16,384 window, so ~52k-65k chars was
    # accepted and cut.
    assert backend.request_limit_chars < 16384 * 4
    # Floors are per language and must not exceed that language's measured
    # minimum (C 2.66, Scala/Java 3.66); an unmeasured language gets the
    # most conservative value.
    assert chars_per_token_floor("c") <= 2.66
    assert chars_per_token_floor("scala") <= 3.66
    assert chars_per_token_floor("java") <= 3.66
    assert chars_per_token_floor("rust") == chars_per_token_floor(None)
    assert admissible_chars(16384, "scala") > admissible_chars(16384, "c")
    assert OllamaBackend(url="http://127.0.0.1:9",
                         request_limit_chars=99).request_limit_chars == 99


def test_short_prompt_is_never_flagged_as_truncated(sap_dir, pr_manifest):
    """Truncation is structurally impossible well below the window.

    The runtime pins a truncated prompt's reported count at num_ctx // 2 + 2, so
    a count far under that cannot be a truncation no matter how unusual its
    chars-per-token ratio. Without this precondition, ordinary short Java prompts
    at 6.0-6.8 chars/token were labelled truncated: 10 of 96 limits_exceeded rows
    in the full 477 run were this false positive.
    """
    req = _request(sap_dir, pr_manifest)
    num_ctx = 16384
    in_tok = max(1, len(req.prompt) // 7)       # ~7 chars/token, past the ratio bound
    assert in_tok < num_ctx * 0.4, "fixture must sit well below the window"
    gen = generate(req, _ctx_backend(num_ctx, in_tok))
    assert gen.outcome == "completion"


def test_high_ratio_near_the_window_is_still_flagged(sap_dir, pr_manifest):
    """The precondition must not disable the check where it does apply."""
    req = _request(sap_dir, pr_manifest)
    num_ctx = 200
    in_tok = int(num_ctx * 0.6)                 # near the window, implausible ratio
    gen = generate(req, _ctx_backend(num_ctx, in_tok))
    assert gen.outcome == "limits_exceeded"
