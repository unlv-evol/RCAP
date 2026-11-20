"""End-to-end driver: one SAP directory -> one Adaptation Package (or a typed failure).

Composite cases (RCAP ref section 13) run through the SAME pipeline with
per-unit coordination: each processing unit gets its own context, request and
generation (one at a time, with sibling-edit signatures attached), and the
case still produces exactly ONE Adaptation Package aggregating the units.
"""

from __future__ import annotations

from rcap.accept import describe_duplicates, duplicated_blocks
from rcap.config import ExecutionConfig
from rcap.context import AdaptationContext, build_context
from rcap.coupling import CouplingRecord, detect_units
from rcap.evidence import pr_functions as pr_function_signatures
from rcap.generate import (
    Backend,
    GenerationRecord,
    generate,
    generate_regions,
    generate_with_placeholder_retry,
)
from rcap.intake import load_case
from rcap.materialize import materialize
from rcap.package import AdaptationPackage, build_package
from rcap.reduction_program import PLACEHOLDER, recover, reduce_program
from rcap.reduction_semantic import reduce_semantic
from rcap.request import synthesize, synthesize_regions
from rcap.selection import select
from rcap.unitkind import narrow_targets


class UnitResult:
    """One processing unit's artifacts (section 13)."""

    def __init__(self, unit, context, request, generation):
        self.unit = unit
        self.context = context
        self.request = request
        self.generation = generation
        # Set only when the opt-in placeholder retry was issued for this unit.
        self.placeholder_retry: dict[str, object] | None = None
        # A whole-target unit reduced to regions (rcap.regions): the requests
        # actually sent (empty: the whole-target `request` is sent) and the
        # plan, with the reason when the change could not be located.
        self.region_requests: list = []
        self.region_plan = None


class CaseResult:
    def __init__(self, **stages):
        self.__dict__.update(stages)


def _expected_placeholders(context: AdaptationContext) -> set[str]:
    # Only the unit's OWN target placeholders bind the candidate; a helper
    # artifact's placeholders are supporting context, never expected output.
    entity = context.target_localization.get("function")
    return {f"/* RCAP_PH_{p.ph_id} */"
            for a in context.program_context
            if a.role == "target" and a.entity == entity
            for p in a.placeholders}


def duplicate_check(context: AdaptationContext, language: str | None):
    """The `reject` hook of `generate` for one unit: an answer that, once its
    placeholders are expanded, holds a block of the fork's code one more time
    than the fork and upstream account for is `duplicated_block`, not a
    completion (rcap.accept). This also covers "placeholder kept AND the elided
    block written out again", which the placeholder check cannot see."""
    entity = context.target_localization.get("function")
    art = next((a for a in context.program_context
                if a.role == "target" and a.entity == entity), None)
    if art is None:
        return None
    original = recover(art)
    before = context.transformation.get("source.before", "")
    after = context.transformation.get("source.after", "")
    for a in context.program_context:
        if a.entity == entity and a.role == "source.before":
            before = recover(a)
        elif a.entity == entity and a.role == "source.after":
            after = recover(a)

    def reject(candidate: str) -> tuple[str, list[str]] | None:
        expanded = candidate
        for e in art.placeholders:
            expanded = expanded.replace(PLACEHOLDER.format(n=e.ph_id), e.original_text, 1)
        dup = duplicated_blocks(original, expanded, before, after, language=language)
        return ("duplicated_block", describe_duplicates(dup)) if dup else None

    return reject


def run_case(
    sap_dir, pr_manifest: dict | None, backend: Backend,
    config: ExecutionConfig | None = None,
    *, placeholder_retry: bool = False, narrow_whole_file_targets: bool = True,
    region_requests: bool = True, generate_units=None,
) -> CaseResult:
    """`narrow_whole_file_targets` (default ON): a unit whose source payloads are
    one function but whose target payload is the fork's whole file is narrowed
    to the function's counterpart when the file declares exactly one
    (rcap.unitkind.narrow_targets). Units whose target is already a function
    are never touched, so their requests are byte-identical either way.

    `region_requests` (default ON): a unit whose target is still the fork's
    whole file is asked for the part(s) of that file its change touches, when
    they can be located (rcap.regions), and the answers are spliced back; the
    whole-file request remains the fallback. Function units are never touched.

    `placeholder_retry` (default OFF): when a unit's answer is rejected for
    its placeholder comments, issue ONE targeted repair request
    (rcap.generate.generate_with_placeholder_retry). Off, the requests and
    outcomes are exactly those of the plain pipeline.

    `generate_units` (default None: every unit): a set of unit entities
    ("functions/<fn>"). Every unit's context and request are still built (so a
    unit outside the set still serves as its siblings' evidence, and every
    prompt is the same as without the set), but only the units in the set are
    sent to the backend; the others get the outcome `not_in_allowlist`."""
    config = config or ExecutionConfig()
    semantic_on = config.mode.semantic_reduction
    program_on = config.mode.program_reduction
    case = load_case(sap_dir, pr_manifest=pr_manifest)
    evidence = config.edit_evidence
    pr_funcs = tuple(pr_function_signatures(sap_dir, pr_manifest)) if evidence else ()
    reduction = None
    try:
        selection = select(case)
        reduction = reduce_semantic(case, selection, enabled=semantic_on)
        materialized = materialize(case, reduction)
        if narrow_whole_file_targets:
            narrow_targets(case, materialized)
        program = reduce_program(case, reduction, materialized, config,
                                 enabled=program_on, evidence_protection=semantic_on)

        coupling: CouplingRecord = detect_units(case)
        # Entities whose tau carries no change are excluded here rather than
        # failing the case: materialize() records them and only refuses when
        # every entity is vacuous. They are not helpers either — a helper is
        # supporting context, whereas these have nothing to contribute at all.
        adaptable = materialized.adaptable(reduction.materialize)
        units = [u for u in coupling.units if u.entity in adaptable]
        # Section 9(5): retained entities beyond every unit's tau are helpers —
        # supporting program context, not transformations of their own.
        helpers = tuple(e for e in adaptable
                        if e not in {u.entity for u in units})
        unit_results: list[UnitResult] = []
        if len(units) <= 1:
            # Single-unit case: the plain path (unit=None keeps the context
            # and request byte-identical to a non-composite build).
            context = build_context(case, reduction, materialized, program,
                                    helpers=helpers, edit_evidence=evidence,
                                    pr_functions=pr_funcs, evidence_filter=config.evidence_filter)
            unit_results.append(UnitResult(units[0] if units else None,
                                           context, None, None))
        else:
            # Section 13 per-unit processing, in application order, each with
            # the signatures of its siblings. Units are never joined.
            for unit in units:
                context = build_context(
                    case, reduction, materialized, program, unit=unit,
                    siblings=tuple(s for s in units if s is not unit),
                    helpers=helpers, edit_evidence=evidence, pr_functions=pr_funcs,
                    evidence_filter=config.evidence_filter)
                unit_results.append(UnitResult(unit, context, None, None))
        for ur in unit_results:
            ur.context.placeholder_scheme = config.placeholder_scheme
            ur.request = synthesize(ur.context)
            if region_requests:
                ur.region_requests, ur.region_plan = synthesize_regions(ur.context)
    except Exception as exc:
        # Section 14: a typed failure record keeps the stable case identity and
        # the evidence dispositions established before the failure.
        if getattr(exc, "stage", None) is not None:
            exc.case_id = case.case_id
            exc.characterization = case.characterization_scores()
            if reduction is not None:
                counts: dict[str, int] = {}
                for d in reduction.dispositions.values():
                    counts[d.value] = counts.get(d.value, 0) + 1
                exc.dispositions = counts
        raise

    for ur in unit_results:
        if generate_units is not None:
            entity = ur.unit.entity if ur.unit is not None \
                else ur.context.target_localization.get("function")
            if entity not in generate_units:
                ur.generation = GenerationRecord(
                    case_id=ur.request.case_id, outcome="not_in_allowlist",
                    diagnostics=[f"{entity}: not in the unit allowlist; built as evidence, not sent"],
                    request_hash=ur.request.request_hash, exec_id="")
                continue
        reject = duplicate_check(ur.context, case.language)
        policy = "tolerant" if config.placeholder_scheme == "shared-v1" else "strict"
        if ur.region_requests:
            ur.generation, ur.placeholder_retry = generate_regions(
                ur.request, ur.region_requests, ur.context.transformation["target"], backend,
                expected_placeholders=_expected_placeholders(ur.context), reject=reject,
                placeholder_retry=placeholder_retry, placeholder_policy=policy)
        elif placeholder_retry and policy == "strict":
            ur.generation, ur.placeholder_retry = generate_with_placeholder_retry(
                ur.request, backend,
                expected_placeholders=_expected_placeholders(ur.context), reject=reject)
        else:
            ur.generation = generate(ur.request, backend,
                                     expected_placeholders=_expected_placeholders(ur.context),
                                     reject=reject, placeholder_policy=policy)

    package: AdaptationPackage | None = None
    if all(ur.generation.outcome == "completion" for ur in unit_results):
        package = build_package(case, unit_results[0].context,
                                unit_results[0].generation,
                                units=unit_results, coupling=coupling)

    primary = unit_results[0]
    return CaseResult(case=case, selection=selection, reduction=reduction,
                      materialized=materialized, program=program,
                      coupling=coupling, units=unit_results,
                      context=primary.context, request=primary.request,
                      generation=primary.generation, package=package)
