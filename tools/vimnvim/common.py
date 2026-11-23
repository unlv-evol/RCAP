"""Shared by the Vim -> Neovim drivers: config, cases, contexts, backend, rows.

The benchmark supplies the three function texts of a case itself, so there is
no SAP, no intake, no SALP stage. What IS reused from the branch, unchanged:
the request templates (rcap.request.synthesize), generation with its limits and
acceptance checks (rcap.generate, rcap.accept), the program cut
(rcap.reduction_program._reduce_one), the re-prompt loop (rcap.reprompt), the
labels (rcap.scoring) and the generation cache of tools/run477.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
sys.path.insert(0, str(HERE.parent / "run477"))

from gencache import CachedBackend  # noqa: E402
from rcap.backend_ollama import OllamaBackend  # noqa: E402
from rcap.context import AdaptationContext  # noqa: E402
from rcap.grammars import C  # noqa: E402
from rcap.pipeline import duplicate_check  # noqa: E402
from rcap.provenance import run_provenance  # noqa: E402
from rcap.reduction_program import (PLACEHOLDER, ReducedArtifact, _changed_lines,  # noqa: E402
                                    _count_nodes, _normalize, _reduce_one, recover)
from rcap.grammars import parse  # noqa: E402
from rcap.unitkind import function_name  # noqa: E402

LANG = "c"
_FENCE = re.compile(r"```[A-Za-z0-9_+-]*\s*\n(.*?)```", re.DOTALL)   # rcap.generate._CODE_BLOCK


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def load_config(out_dir: str | Path) -> dict:
    """The FROZEN config of a results directory (freeze.py wrote it)."""
    cfg = json.loads((Path(out_dir) / "config.json").read_text())
    cfg["_out"] = str(Path(out_dir).resolve())
    # arms added AFTER results were seen live in their own file and are marked
    # exploratory; the frozen config is never edited
    extra = Path(out_dir) / "config.exploratory.json"
    if extra.exists():
        for name, arm in json.loads(extra.read_text())["arms"].items():
            assert name not in cfg["arms"] and arm.get("exploratory") is True
            cfg["arms"][name] = arm
    return cfg


def case_id(index: int, case: dict) -> str:
    return f"{index:03d}_{case['commit_id_target'][:10]}"


def load_cases(cfg: dict) -> dict[str, dict]:
    """case_id -> case, in benchmark order; asserted against the frozen list."""
    path = Path(cfg["benchmark"]["test_set"])
    raw = path.read_bytes()
    want = cfg["benchmark"].get("test_set_sha256")
    if want and not want.startswith("FILLED") and hashlib.sha256(raw).hexdigest() != want:
        raise SystemExit(f"test set changed: {path}")
    cases = {case_id(i, c): c for i, c in enumerate(json.loads(raw))}
    frozen = Path(cfg["_out"]) / "cases.txt"
    if frozen.exists() and frozen.read_text().split() != list(cases):
        raise SystemExit("case list differs from the frozen cases.txt")
    return cases


def nl(text: str) -> str:
    return text if text.endswith("\n") else text + "\n"


def load_jsonl(path: str | Path, key: str = "case_id") -> dict[str, dict]:
    out: dict[str, dict] = {}
    p = Path(path)
    if p.exists():
        for line in p.read_text().splitlines():
            if line.strip():
                try:
                    row = json.loads(line)
                except ValueError:
                    continue            # a line cut by a killed run
                out[row[key]] = row
    return out


def extract_code(raw: str | None) -> str | None:
    """The answer's code as rcap.generate takes it: the first fenced block,
    else the whole answer; None when there is nothing."""
    if not raw or not raw.strip():
        return None
    m = _FENCE.search(raw)
    return (m.group(1) if m else raw).strip() + "\n"


class Unit:
    """One case as a single RCAP unit: context, request inputs, recovery."""

    def __init__(self, cid: str, case: dict, reduced: bool, min_lines: int = 3):
        self.case_id, self.case, self.reduced = cid, case, reduced
        f_s, f_sp, f_t = nl(case["func_before_source"]), nl(case["func_after_source"]), nl(case["func_before_target"])
        self.target_original = f_t
        self.entity = (function_name(f_t, LANG) or function_name(nl(case["func_after_target"]), LANG) or ("function", 0))[0]
        ch_s, ch_sp = _changed_lines(f_s, f_sp)
        if reduced:
            art_s = _reduce_one(self.entity, "source.before", f_s, ch_s, C, min_lines, None, LANG)
            art_sp = _reduce_one(self.entity, "source.after", f_sp, ch_sp, C, min_lines, None, LANG)
            norm = {_normalize(p.original_text) for p in art_s.placeholders + art_sp.placeholders}
            # "only segments the source also removes": with nothing removed on the
            # source side, nothing is removed from the target (an empty set, NOT None)
            art_t = _reduce_one(self.entity, "target", f_t, set(), C, min_lines, norm, LANG)
        else:
            def whole(role, text, protected):
                n = _count_nodes(parse(text, C).root_node)
                return ReducedArtifact(entity=self.entity, role=role, reduced_text=text,
                                       protected_lines=sorted(protected), nodes_before=n, nodes_after=n,
                                       reduction_reason="program_reduction_disabled")
            art_s, art_sp, art_t = whole("source.before", f_s, ch_s), whole("source.after", f_sp, ch_sp), whole("target", f_t, set())
        self.artifacts = {"source.before": art_s, "source.after": art_sp, "target": art_t}
        assert recover(art_t) == f_t and recover(art_s) == f_s and recover(art_sp) == f_sp
        self.expected_placeholders = {PLACEHOLDER.format(n=p.ph_id) for p in art_t.placeholders}
        self.context = AdaptationContext(
            case_id=cid, language=LANG,
            unit={"entity": self.entity},
            transformation={r: a.reduced_text for r, a in self.artifacts.items()},
            edit_regions=[],
            target_localization={"function": self.entity, "file": case["file_path_target"]},
            program_context=[art_s, art_sp, art_t],
            reduction_mode={"program": "reduced" if reduced else "disabled", "semantic": "none (no SAP evidence)"})
        self.reject = duplicate_check(self.context, LANG)

    def expand(self, candidate: str) -> str:
        """Placeholder form -> full text (the recovery map of the target)."""
        out = candidate
        for p in self.artifacts["target"].placeholders:
            out = out.replace(PLACEHOLDER.format(n=p.ph_id), p.original_text, 1)
        return out

    def reduction_stats(self) -> dict[str, object]:
        a = self.artifacts
        return {"nodes_before": sum(x.nodes_before for x in a.values()),
                "nodes_after": sum(x.nodes_after for x in a.values()),
                "placeholders": {r: len(x.placeholders) for r, x in a.items()},
                "chars_before": sum(len(recover(x)) for x in a.values()),
                "chars_after": sum(len(x.reduced_text) for x in a.values())}


def make_backend(cfg: dict, *, live: bool = True) -> CachedBackend:
    b = cfg["backend"]
    inner = OllamaBackend(b["model"], b["url"], num_ctx=b["num_ctx"])
    assert inner.params == {"temperature": b["temperature"], "seed": b["seed"], "num_ctx": b["num_ctx"]}
    backend = CachedBackend(inner, Path(cfg["_out"]) / "gen-cache")
    if not live:
        def refuse(request, **_):
            raise RuntimeError("offline: no cached answer for this prompt")
        inner.generate = refuse      # type: ignore[method-assign]
    return backend


def provenance(cfg: dict, backend=None, **extra) -> dict:
    return run_provenance(backend, config_id=cfg["config_id"], language=LANG, **extra)


def raw_answer(cfg: dict, prompt: str) -> dict | None:
    """The cached raw answer of a prompt (also of one whose candidate was rejected)."""
    f = Path(cfg["_out"]) / "gen-cache" / f"{sha(prompt)}.json"
    return json.loads(f.read_text()) if f.exists() else None


def flagged_baseline(cfg: dict, stored: dict[str, dict]) -> dict[str, str]:
    """case_id -> why, for stored baseline records the erratum flags."""
    limit = cfg["erratum"]["truncation_chars_per_input_token"]
    out = {}
    for cid, r in stored.items():
        tok = r.get("input_tokens")
        if tok and r.get("prompt_chars", 0) / tok > limit:
            out[cid] = f"truncated ({r['prompt_chars']} chars / {tok} input tokens)"
        elif r.get("outcome") in cfg["erratum"]["flag_outcomes"]:
            out[cid] = f"{r['outcome']} (timeout)"
    return out
