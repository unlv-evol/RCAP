"""Run-level overrides read from a file next to the code, for drivers whose command lines are fixed.

A batch job's command line is frozen when it is submitted, so a study that must change how
already-queued jobs run RCAP does it with a file the drivers read at start-up:

    <repo root>/cand32k-overrides.json        (or the path in RCAP_OVERRIDES; RCAP_OVERRIDES=none: ignore)
    {"name": "cand32k-ev1",
     "edit_evidence": true,                                # ExecutionConfig.edit_evidence
     "unit_allowlist": "tools/candidates/shared-units.tsv"} # relative to the repo root; TSV pair, pr, sap, unit

RCAP_EDIT_EVIDENCE=0 / =1 forces edit evidence off / on whatever the file says (the ablation).
No file and no environment variable: nothing changes (`Overrides.active` is False).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FILE = REPO_ROOT / "cand32k-overrides.json"


@dataclass
class Overrides:
    path: str | None = None
    name: str | None = None
    edit_evidence: bool | None = None          # None: leave the config as it is
    allowlist_path: str | None = None
    allowlist: dict[tuple[str, str, str], set[str]] | None = None   # (pair, pr, sap) -> {"functions/<fn>"}
    notes: list[str] = field(default_factory=list)

    @property
    def active(self) -> bool:
        return self.path is not None or self.edit_evidence is not None

    @property
    def variant(self) -> str | None:
        """A tag for every result row, so a resumed run never mixes rows written under other settings."""
        if not self.active:
            return None
        return f"{self.name or 'overrides'}:ev={int(bool(self.edit_evidence))}:allow={int(self.allowlist is not None)}"

    def units_for(self, pair: str, pr: str, sap: str) -> set[str] | None:
        """The units of this SAP that may be sent (None: all of them)."""
        if self.allowlist is None:
            return None
        return self.allowlist.get((pair, pr, sap), set())

    def apply(self, config):
        """The ExecutionConfig with the overrides applied (a copy)."""
        if self.edit_evidence is None:
            return config
        return config.model_copy(update={"edit_evidence": self.edit_evidence})


def load_allowlist(path: str | Path) -> dict[tuple[str, str, str], set[str]]:
    out: dict[tuple[str, str, str], set[str]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith(("#", "pair\t")):
            continue
        pair, pr, sap, unit = line.split("\t")[:4]
        out.setdefault((pair, pr, sap), set()).add(unit if unit.startswith("functions/") else f"functions/{unit}")
    return out


def load(path: str | Path | None = None) -> Overrides:
    env = os.environ.get("RCAP_OVERRIDES")
    if path is None:
        path = None if env and env.lower() == "none" else Path(env) if env else DEFAULT_FILE
    ov = Overrides()
    if path is not None and Path(path).is_file():
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        ov.path = str(path)
        ov.name = raw.get("name")
        if "edit_evidence" in raw:
            ov.edit_evidence = bool(raw["edit_evidence"])
        if raw.get("unit_allowlist"):
            al = Path(raw["unit_allowlist"])
            al = al if al.is_absolute() else REPO_ROOT / al
            ov.allowlist_path = str(al)
            ov.allowlist = load_allowlist(al)
        ov.notes.append(f"overrides file {path}")
    flag = os.environ.get("RCAP_EDIT_EVIDENCE")
    if flag in ("0", "1"):
        ov.edit_evidence = flag == "1"
        ov.notes.append(f"RCAP_EDIT_EVIDENCE={flag}")
    return ov
