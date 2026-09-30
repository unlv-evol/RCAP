"""End-to-end driver: one SAP directory -> one Adaptation Package (or a typed failure).

Composite cases (RCAP ref section 13) run through the SAME pipeline with
per-unit coordination: each processing unit gets its own context, request and
generation (one at a time, with sibling-edit signatures attached), and the
case still produces exactly ONE Adaptation Package aggregating the units.
"""

from __future__ import annotations

from rcap.config import ExecutionConfig
from rcap.context import AdaptationContext, build_context
from rcap.coupling import CouplingRecord, detect_units
from rcap.generate import Backend, generate
from rcap.intake import load_case
from rcap.materialize import materialize
from rcap.package import AdaptationPackage, build_package
from rcap.reduction_program import reduce_program
from rcap.reduction_semantic import reduce_semantic
from rcap.request import synthesize
from rcap.selection import select


class UnitResult:
    """One processing unit's artifacts (section 13)."""

    def __init__(self, unit, context, request, generation):
        self.unit = unit
        self.context = context
        self.request = request
        self.generation = generation


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


def run_case(
    sap_dir, pr_manifest: dict | None, backend: Backend,
    config: ExecutionConfig | None = None,
) -> CaseResult:
    config = config or ExecutionConfig()
    semantic_on = config.mode.semantic_reduction
    program_on = config.mode.program_reduction
    case = load_case(sap_dir, pr_manifest=pr_manifest)
    reduction = None
    try:
        selection = select(case)
        reduction = reduce_semantic(case, selection, enabled=semantic_on)
        materialized = materialize(case, reduction)
        program = reduce_program(case, reduction, materialized, config,
                                 enabled=program_on, evidence_protection=semantic_on)

        coupling: CouplingRecord = detect_units(case)
        units = [u for u in coupling.units if u.entity in reduction.materialize]
        # Section 9(5): retained entities beyond every unit's tau are helpers —
        # supporting program context, not transformations of their own.
        helpers = tuple(e for e in reduction.materialize
                        if e not in {u.entity for u in units})
        unit_results: list[UnitResult] = []
        if len(units) <= 1:
            # Single-unit case: the plain path (unit=None keeps the context
            # and request byte-identical to a non-composite build).
            context = build_context(case, reduction, materialized, program,
                                    helpers=helpers)
            unit_results.append(UnitResult(units[0] if units else None,
                                           context, None, None))
        else:
            # Section 13 per-unit processing, in application order, each with
            # the signatures of its siblings. Units are never joined.
            for unit in units:
                context = build_context(
                    case, reduction, materialized, program, unit=unit,
                    siblings=tuple(s for s in units if s is not unit),
                    helpers=helpers)
                unit_results.append(UnitResult(unit, context, None, None))
        for ur in unit_results:
            ur.request = synthesize(ur.context)
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
        ur.generation = generate(ur.request, backend,
                                 expected_placeholders=_expected_placeholders(ur.context))

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
