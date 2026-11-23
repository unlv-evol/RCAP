"""Accuracy, AED and RED as PPatHF's replication package computes them.

Pan et al., "Automating Zero-Shot Patch Porting for Hard Forks" (ISSTA'24),
section 5 (eq. 4) and the code of their package, which is what is matched here
line for line (file:line of the package in brackets):

tokens      `cleaned_tokens`: empty lines removed, the text parsed with the
            tree-sitter C grammar, the leaves taken in order, leaves of type
            `comment` dropped [reduction/fcu.py:75-101, 218-221]. Layout and
            comments therefore never count; string literals are several leaves
            (the quotes and the content), as tree-sitter has them.
Accuracy    the generated function's token list EQUALS the developer's
            [metrics.py:77, `np.array_equal`]. Identity is decided on tokens,
            not on text.
AED         token-level Levenshtein distance generated <-> developer
            [metrics.py:78, `editdistance.eval`], averaged over samples.
RED         per sample gen_distance / src_distance, where src_distance is the
            distance pre-patch fork function <-> developer; averaged over
            samples [metrics.py:79-82]. When the pre-patch function already
            equals the developer's on tokens (src_distance == 0, a comment-only
            port) the package DIVIDES BY 1, i.e. RED = gen_distance: such a
            sample scores 0 when the output is right and its raw distance
            otherwise. `src_zero` marks them so a report can count them.
aligned     over the full test set a sample the model could not take, or whose
            generation is INCOMPLETE (the last non-empty line of the output
            does not start with `}` [test.py:92]), is scored as the DUMMY
            output = the unchanged pre-patch fork function
            [test.py:27 `generate_dummy`, :119-154 `align_test_metrics`; paper
            sec. 6.1]. "Origin" in the paper's tables is the all-dummy row.

Not from the package: `red_band`, the paired tests and the bootstrap. They are
ours and are reported as such.

Needs `tree-sitter-c` and `editdistance` (the optional extra `vimnvim`).
"""

from __future__ import annotations

import math
import sys
from collections.abc import Sequence
from functools import lru_cache

RED_BANDS = ("RED = 0", "0 < RED < 0.2", "0.2 <= RED < 1", "RED >= 1")


@lru_cache(maxsize=1)
def _parser():
    import tree_sitter
    import tree_sitter_c

    sys.setrecursionlimit(max(sys.getrecursionlimit(), 100000))
    return tree_sitter.Parser(tree_sitter.Language(tree_sitter_c.language()))


def remove_empty_lines(text: str) -> str:
    return "".join(line for line in text.splitlines(keepends=True) if line.strip())


def cleaned_tokens(code: str) -> list[str]:
    """Leaf tokens of `code` in order, comments dropped (fcu.get_cleaned_tokens)."""
    raw = remove_empty_lines(code or "").encode("utf-8")
    out: list[str] = []
    stack = [_parser().parse(raw).root_node]
    while stack:
        node = stack.pop()
        if node.children:
            stack.extend(reversed(node.children))
        elif node.type != "comment":
            out.append(raw[node.start_byte:node.end_byte].decode("utf-8", "replace"))
    return out


def edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    import editdistance

    return int(editdistance.eval(list(a), list(b)))


def is_complete(generation: str | None) -> bool:
    """The package's completeness rule: the last line starts with `}`."""
    lines = (generation or "").strip().splitlines()
    return bool(lines) and lines[-1][:1] == "}"


def score(generated: str, developer: str, pre_patch: str) -> dict[str, object]:
    """One sample, exactly as MetricsSampleWise.get_sample_wise_test_results."""
    gen, ref, before = cleaned_tokens(generated), cleaned_tokens(developer), cleaned_tokens(pre_patch)
    gen_distance = edit_distance(gen, ref)
    src_distance = edit_distance(before, ref)
    return {
        "exact_match": gen == ref,
        "gen_distance": gen_distance,
        "src_distance": src_distance,
        "rel_distance": gen_distance / (src_distance if src_distance != 0 else 1),
        "src_zero": src_distance == 0,
        "echo": gen == before,          # ours: the output is the pre-patch function on tokens
    }


def score_aligned(generated: str | None, developer: str, pre_patch: str) -> dict[str, object]:
    """`score`, with the dummy (pre-patch function) standing in for an output
    that is missing or incomplete. `use_dummy` and `complete` say which."""
    complete = generated is not None and is_complete(generated)
    row = score(generated if complete else pre_patch, developer, pre_patch)
    row.update(complete=complete, use_dummy=not complete)
    return row


def aggregate(rows: Sequence[dict[str, object]]) -> dict[str, object]:
    """n, exact, Accuracy, AED, RED (means over samples, as `cal_metrics`)."""
    n = len(rows)
    if not n:
        return {"n": 0, "exact": 0, "accuracy": None, "aed": None, "red": None, "red_median": None,
                "src_zero": 0}
    reds = sorted(float(r["rel_distance"]) for r in rows)
    mid = n // 2
    return {
        "n": n,
        "exact": sum(1 for r in rows if r["exact_match"]),
        "accuracy": sum(1 for r in rows if r["exact_match"]) / n,
        "aed": sum(int(r["gen_distance"]) for r in rows) / n,
        "red": sum(reds) / n,
        "red_median": reds[mid] if n % 2 else (reds[mid - 1] + reds[mid]) / 2,   # ours
        "src_zero": sum(1 for r in rows if r.get("src_zero")),
    }


def red_band(rel_distance: float) -> str:
    if rel_distance == 0:
        return RED_BANDS[0]
    if rel_distance < 0.2:
        return RED_BANDS[1]
    if rel_distance < 1:
        return RED_BANDS[2]
    return RED_BANDS[3]


# ------------------------------------------------------------------ paired tests (ours)

def mcnemar_exact(a: Sequence[bool], b: Sequence[bool]) -> dict[str, object]:
    """Exact two-sided McNemar test on paired booleans. `gain`: b right where a
    is wrong; `loss`: the reverse."""
    if len(a) != len(b):
        raise ValueError("paired samples must have the same length")
    gain = sum(1 for x, y in zip(a, b) if y and not x)
    loss = sum(1 for x, y in zip(a, b) if x and not y)
    n = gain + loss
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(gain, loss) + 1)) / 2 ** n)
    return {"n": len(a), "gain": gain, "loss": loss, "p": p}


def sign_test(a: Sequence[float], b: Sequence[float]) -> dict[str, object]:
    """Exact two-sided sign test on paired values (ties dropped). `lower`:
    pairs where b < a."""
    lower = sum(1 for x, y in zip(a, b) if y < x)
    higher = sum(1 for x, y in zip(a, b) if y > x)
    n = lower + higher
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(lower, higher) + 1)) / 2 ** n)
    return {"n": len(a), "lower": lower, "higher": higher, "ties": len(a) - n, "p": p}


def wilcoxon(a: Sequence[float], b: Sequence[float]) -> dict[str, object]:
    """Two-sided Wilcoxon signed-rank test on paired values (zero differences
    dropped); p is None when nothing differs or scipy is not installed."""
    diffs = [y - x for x, y in zip(a, b) if y != x]
    out: dict[str, object] = {"n": len(a), "nonzero": len(diffs), "p": None, "statistic": None}
    if not diffs:
        return out
    try:
        from scipy.stats import wilcoxon as _w
    except ImportError:
        return out
    res = _w(diffs)
    out.update(p=float(res.pvalue), statistic=float(res.statistic))
    return out
