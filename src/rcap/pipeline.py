"""End-to-end driver: one SAP directory -> one Adaptation Package (or a typed failure)."""

from __future__ import annotations

from rcap.config import ExecutionConfig
from rcap.context import AdaptationContext, build_context
from rcap.generate import Backend, GenerationRecord, generate
from rcap.intake import load_case
from rcap.materialize import materialize
from rcap.package import AdaptationPackage, build_package
from rcap.reduction_program import reduce_program
from rcap.reduction_semantic import reduce_semantic
from rcap.request import AdaptationRequest, synthesize
from rcap.selection import select


class CaseResult:
    def __init__(self, **stages):
        self.__dict__.update(stages)


def run_case(
    sap_dir, pr_manifest: dict | None, backend: Backend,
    config: ExecutionConfig | None = None,
) -> CaseResult:
    config = config or ExecutionConfig()
    semantic_on = config.mode.semantic_reduction
    program_on = config.mode.program_reduction
    case = load_case(sap_dir, pr_manifest=pr_manifest)
    selection = select(case)
    reduction = reduce_semantic(case, selection, enabled=semantic_on)
    materialized = materialize(case, reduction)
    program = reduce_program(case, reduction, materialized, config,
                             enabled=program_on, evidence_protection=semantic_on)
    context: AdaptationContext = build_context(case, reduction, materialized, program)
    request: AdaptationRequest = synthesize(context)
    expected = {f"/* RCAP_PH_{p.ph_id} */"
                for a in context.program_context if a.role == "target"
                for p in a.placeholders}
    generation: GenerationRecord = generate(request, backend,
                                            expected_placeholders=expected)
    package: AdaptationPackage | None = None
    if generation.outcome == "completion":
        package = build_package(case, context, generation)
    return CaseResult(case=case, selection=selection, reduction=reduction,
                      materialized=materialized, program=program, context=context,
                      request=request, generation=generation, package=package)
