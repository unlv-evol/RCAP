"""One definition of "a port", used by every report.

Raw VALID (the build passes, the unit's tests show no new failure) over-counts:
an audit of the 477 baseline and of the held-out PR evaluation found that about
a third of raw VALID is a leaked placeholder, the fork's own code returned, or
an import. This module turns the texts of a unit and the verdict of its row
into the labels a report may count.

Per UNIT (`score_unit`), in this order:

NO_CANDIDATE       nothing was generated;
SHAPE_MISMATCH     one function offered for a whole-file unit (never applied);
PLACEHOLDER_LEAK, DUPLICATED_BLOCK, WEAKENED, FORK_REGRESSION   (rcap.fidelity);
REVERTED           the unit is the fork's code now (comments and layout aside)
                   while an EARLIER attempt of the same unit differed: a re-prompt
                   or retry went back;
NO_CHANGE          the fork's code (an echo); COMMENT_ONLY when the comments
                   changed the way upstream's do;
NOTHING_TO_PORT(_IN_ISOLATION)   a pure rename validated alone, or a source
                   function with no counterpart in the fork, left unchanged:
                   neither a model failure nor a success;
IMPORT_ONLY, PARTIAL, FAITHFUL_CODE   (rcap.fidelity).

A unit that was not written into the built file (overlap, `found_0`, rejected)
keeps its label and has `applied=False`; it counts for nothing.

Per ROW (`score_row`): derived from the row's APPLIED units, never from the row
text. A row counts as a port only if at least one unit carries a real change
and no unit is a leak, a duplicate, a regression or a reversion.

REAL_PORT (the one figure): the row / unit PASSES build + tests (the caller says
whether standalone or in PR context) AND the label is FAITHFUL_CODE (or the unit
is listed as a hand-confirmed adapted port) AND no leak / duplicate / regression.
Import-only and comment-only changes are reported next to it, never inside it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from rcap import fidelity as fd
from rcap.reprompt import code_equal
from rcap.unitkind import FUNCTION, WHOLE_KINDS, find_methods, function_name, shape_mismatch, text_kind

NO_CANDIDATE = "NO_CANDIDATE"
SHAPE_MISMATCH = "SHAPE_MISMATCH"
REVERTED = "REVERTED"

# labels that disqualify a row, in the order they are reported
LEAK_LABELS = (fd.PLACEHOLDER_LEAK,)
DUPLICATE_LABELS = (fd.DUPLICATED_BLOCK,)
REGRESSION_LABELS = (fd.FORK_REGRESSION, fd.WEAKENED)
NOTHING_LABELS = (fd.NOTHING_TO_PORT, fd.NOTHING_TO_PORT_IN_ISOLATION)
PASSING = frozenset({"VALID", "VALID_PR"})

# row classes
ROW_LEAK = "LEAK"
ROW_DUPLICATE = "DUPLICATED_BLOCK"
ROW_REGRESSION = "REGRESSION"
ROW_REVERTED = "REVERTED"
ROW_CODE = "CODE"                    # >= 1 FAITHFUL_CODE (or hand-confirmed) unit
ROW_PARTIAL = "PARTIAL_CODE"         # code changed, not confirmed as the upstream change
ROW_IMPORT = "IMPORT_ONLY"
ROW_COMMENT = "COMMENT_ONLY"
ROW_NOTHING = "NOTHING_TO_PORT"
ROW_NO_CHANGE = "NO_CHANGE"
ROW_NOT_APPLIED = "NOTHING_APPLIED"
ROW_ORDER = (ROW_LEAK, ROW_DUPLICATE, ROW_REGRESSION, ROW_REVERTED, ROW_CODE, ROW_PARTIAL,
             ROW_IMPORT, ROW_COMMENT, ROW_NOTHING, ROW_NO_CHANGE, ROW_NOT_APPLIED)


@dataclass
class UnitScore:
    label: str
    applied: bool = True
    why_not_applied: str | None = None
    hand_confirmed: bool = False
    fidelity: fd.Fidelity | None = None
    reasons: list[str] = field(default_factory=list)

    @property
    def is_code_port(self) -> bool:
        """A real code change of this unit (before the build verdict is looked at)."""
        return self.applied and (self.label == fd.FAITHFUL_CODE or (
            self.hand_confirmed and self.label in (fd.PARTIAL, fd.IMPORT_ONLY, fd.FAITHFUL_CODE)))

    def as_dict(self) -> dict[str, object]:
        f = self.fidelity
        return {"label": self.label, "applied": self.applied, "why_not_applied": self.why_not_applied,
                "hand_confirmed": self.hand_confirmed, "is_code_port": self.is_code_port,
                "reasons": (self.reasons or (f.reasons if f else []))[:4],
                "evidence": (f.evidence if f else [])[:4], "flags": f.flags if f else [],
                "recall": f.recall if f else None, "novel_share": f.novel_share if f else None,
                "kind": f.kind if f else None, "upstream_kind": f.upstream_kind if f else None,
                "adapted": f.adapted if f else []}


def has_no_counterpart(fork_original: str, upstream_before: str, upstream_after: str,
                       language: str | None) -> bool:
    """The unit is ONE source function, its target the fork's whole file, and
    the fork's file declares no method of that name (before- or after-name)."""
    if text_kind(fork_original, language) not in WHOLE_KINDS:
        return False
    names = [n for n in (function_name(upstream_before, language), function_name(upstream_after, language)) if n]
    if not names or (text_kind(upstream_before, language) != FUNCTION and text_kind(upstream_after, language) != FUNCTION):
        return False
    return all(not find_methods(fork_original, name, language) for name, _ in names)


def score_unit(fork_original: str | None, candidate: str | None, upstream_before: str, upstream_after: str,
               *, language: str | None = "java", earlier: Sequence[str] = (), applied: bool = True,
               why_not_applied: str | None = None, rename_context: str | None = None,
               fork_has: Callable[[str], bool] | None = None,
               upstream_file: tuple[str, str] | None = None, hand_confirmed: bool = False) -> UnitScore:
    """The label of one unit. `earlier`: the unit's expanded candidates of
    earlier attempts (for REVERTED). `applied=False` when the candidate did not
    reach the built file; the label is still computed and reported."""
    fork_original = fork_original or ""
    if not candidate:
        return UnitScore(NO_CANDIDATE, applied=False, why_not_applied=why_not_applied or "no candidate")
    why = shape_mismatch(fork_original, candidate, language)
    if why:
        return UnitScore(SHAPE_MISMATCH, applied=False, why_not_applied="shape_mismatch", reasons=[why])
    absent = has_no_counterpart(fork_original, upstream_before, upstream_after, language)
    f = fd.judge(fork_original, candidate, upstream_before, upstream_after, rename_context=rename_context,
                 fork_has=fork_has, language=language, no_counterpart=absent, upstream_file=upstream_file)
    label = f.label
    if label in (fd.NO_CHANGE, fd.COMMENT_ONLY, *NOTHING_LABELS) and code_equal(candidate, fork_original) \
            and any(not code_equal(e, fork_original) for e in earlier if e):
        f.reasons.insert(0, "an earlier attempt of this unit differed from the fork; this one is the fork's code again")
        label = REVERTED
    return UnitScore(label, applied=applied, why_not_applied=None if applied else (why_not_applied or "not applied"),
                     hand_confirmed=hand_confirmed, fidelity=f)


@dataclass
class RowScore:
    verdict: str                       # the raw build/test verdict
    row_class: str
    real_port: bool                    # passes AND a code port AND nothing disqualifying
    unit_labels: list[str]
    reason: str = ""

    def as_dict(self) -> dict[str, object]:
        return dict(self.__dict__)


def score_row(verdict: str, units: Sequence[UnitScore]) -> RowScore:
    """The class of a row from its APPLIED units (see the module docstring)."""
    live = [u for u in units if u.applied]
    labels = [u.label + ("" if u.applied else " (not applied)") for u in units]
    has = lambda group: [u for u in live if u.label in group]  # noqa: E731
    if not live:
        cls, why = ROW_NOT_APPLIED, "no unit of the row was written into the file"
    elif has(LEAK_LABELS):
        cls, why = ROW_LEAK, "a unit writes a placeholder marker into the fork"
    elif has(DUPLICATE_LABELS):
        cls, why = ROW_DUPLICATE, "a unit writes a block of the fork's code a second time"
    elif has(REGRESSION_LABELS):
        cls, why = ROW_REGRESSION, "a unit displaces fork code or weakens an expectation"
    elif has((REVERTED,)):
        cls, why = ROW_REVERTED, "a unit went back to the fork's code after an attempt that differed"
    elif any(u.is_code_port for u in live):
        cls, why = ROW_CODE, ""
    elif has((fd.PARTIAL,)):
        cls, why = ROW_PARTIAL, "code changed, but no unit is confirmed as its upstream change"
    elif has((fd.IMPORT_ONLY,)):
        cls, why = ROW_IMPORT, "imports only"
    elif has((fd.COMMENT_ONLY,)):
        cls, why = ROW_COMMENT, "comments only"
    elif has(NOTHING_LABELS):
        cls, why = ROW_NOTHING, "nothing to port in this row"
    else:
        cls, why = ROW_NO_CHANGE, "every applied unit is the fork's code"
    return RowScore(verdict, cls, verdict in PASSING and cls == ROW_CODE, labels, why)


def real_port_unit(unit: UnitScore, passes: bool) -> bool:
    """REAL_PORT at unit level: the unit's build + tests pass (standalone: its
    row's verdict; in PR context: its own VALID_PR) and it is a code port."""
    return passes and unit.is_code_port
