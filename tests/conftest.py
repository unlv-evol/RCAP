"""Synthetic SAP fixture, built with SALP's own models and writer.

Because the fixture is written by `salp.packaging.write_sap`/`write_pr_group`
and gated by `salp validate`'s own schema check, synthetic and real SAPs are
guaranteed to share one shape.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from salp.characterization import Characterizer
from salp.models import (
    SAP,
    Category,
    CategoryEvidence,
    EvidenceObject,
    EvidenceState,
    FunctionPayload,
    Hunk,
    PRGroup,
    Provenance,
    Relationship,
    SAPReference,
    TransformationUnit,
)
from salp.packaging import write_pr_group, write_sap

F_S = """void validate(Config c) {
    if (c == null) {
        throw new NullPointerException("config");
    }
    int v = c.getInt("version");
    if (v > 2)
        setup(v);
    for (Node n : c.nodes()) {
        register(n);
    }
}
"""
F_S_PRIME = F_S.replace("if (v > 2)", "if (v >= 2)")
F_T = F_S.replace("register(n);", "registerNode(n);")  # drifted fork copy

HUNK_DIFF = """@@ -5,1 +5,1 @@
-    if (v > 2)
+    if (v >= 2)
"""


def _prov(component: str, **pin: str) -> Provenance:
    from salp.models.evidence import RepositoryStatePin
    rp = RepositoryStatePin(repo=pin["repo"], commit=pin["commit"]) if pin else None
    return Provenance(analysis_component=component, repository_pin=rp)


def _element(oid: str, element: str, state: EvidenceState, **kw) -> EvidenceObject:
    return EvidenceObject(
        object_id=oid, object_type=element, state=state,
        provenance=kw.pop("provenance", _prov("fixture")), **kw,
    )


def build_fixture_pr(root: Path, *, enriched: bool = False) -> Path:
    """Write one PR grouping with one single-hunk mapped SAP; return the PR dir.

    `enriched=True` simulates SALP edge promotion: extra evidence elements and
    typed relationship edges beyond `aligned_to`, so relationship-guided
    reduction can be exercised before real enriched SAPs exist.
    """
    pr_dir = root / "linkedinKafka-apacheKafka" / "PR-1"
    sap_id = "sap-Validator"

    fn = FunctionPayload(
        fn_id="fn-1", ext="java",
        has_source_before=True, has_source_after=True, has_target=True,
        has_structure=True,
        signature="void validate(Config)", method_name="validate",
        target_signature="void validate(Config)", target_match_kind="exact",
    )

    categories = {
        Category.SOURCE_CHANGE: CategoryEvidence(category=Category.SOURCE_CHANGE, elements=[
            _element("H-1:source_change:edit_region", "edit_region", EvidenceState.PRESENT,
                     provenance=_prov("fixture", repo="apache/kafka", commit="a" * 40)),
        ]),
        Category.TARGET_LOCALIZATION: CategoryEvidence(category=Category.TARGET_LOCALIZATION, elements=[
            _element("H-1:localization:target_function", "localized_target_function",
                     EvidenceState.PRESENT,
                     provenance=_prov("fixture", repo="linkedin/kafka", commit="b" * 40)),
        ]),
        Category.FUNCTION_TRANSFORMATION: CategoryEvidence(category=Category.FUNCTION_TRANSFORMATION, elements=[
            _element("H-1:transformation:triple", "transformation_unit", EvidenceState.PRESENT),
        ]),
        Category.STRUCTURAL: CategoryEvidence(category=Category.STRUCTURAL, elements=[
            _element("H-1:structural:edit_region_structure", "edit_region_structure",
                     EvidenceState.PRESENT),
        ]),
        Category.REFACTORING: CategoryEvidence(category=Category.REFACTORING, elements=[
            _element("H-1:refactoring:rename", "refactoring_findings",
                     EvidenceState.VERIFIED_ABSENT),
        ]),
        Category.COMPATIBILITY: CategoryEvidence(category=Category.COMPATIBILITY, elements=[
            _element("H-1:compatibility:source_apis", "source_apis", EvidenceState.PRESENT),
        ]),
        Category.SURROUNDING: CategoryEvidence(category=Category.SURROUNDING, elements=[
            _element("H-1:surrounding:enclosing_context", "enclosing_context",
                     EvidenceState.PRESENT),
        ]),
        Category.VERIFICATION: CategoryEvidence(category=Category.VERIFICATION, elements=[
            _element("H-1:verification:covering_tests", "covering_tests",
                     EvidenceState.UNAVAILABLE),
        ]),
    }

    relationships = [
        Relationship(src="H-1:ER-1", rel="aligned_to", dst="functions/fn-1"),
    ]
    if enriched:
        categories[Category.REFACTORING].elements.extend([
            _element("H-1:refactoring:rename_setup", "refactoring_findings",
                     EvidenceState.PRESENT,
                     attributes={"kind": "rename", "from": "setup", "to": "setupInternal"}),
            _element("H-1:refactoring:unrelated_rename", "refactoring_findings",
                     EvidenceState.PRESENT,
                     attributes={"kind": "rename", "from": "helperElsewhere", "to": "helper2"}),
        ])
        categories[Category.COMPATIBILITY].elements.extend([
            _element("H-1:compatibility:api_diff", "dependency_diff",
                     EvidenceState.PRESENT, attributes={"added": [], "removed": []}),
            _element("H-1:compatibility:objects_absent", "target_apis",
                     EvidenceState.VERIFIED_ABSENT),
            _element("H-1:compatibility:version_conflict", "dependency_constraints",
                     EvidenceState.PRESENT, blocking_conflict=True),
        ])
        relationships.extend([
            Relationship(src="H-1:ER-1", rel="depends_on", dst="H-1:compatibility:api_diff"),
            Relationship(src="functions/fn-1", rel="renamed_to",
                         dst="H-1:refactoring:rename_setup"),
            Relationship(src="H-1:ER-1", rel="depends_on",
                         dst="H-1:verification"),
            Relationship(src="H-1:ER-1", rel="depends_on",
                         dst="H-1:compatibility:objects_absent"),
            Relationship(src="H-1:ER-1", rel="depends_on",
                         dst="H-1:compatibility:version_conflict"),
            Relationship(src="H-1:ER-1", rel="same_pull_request",
                         dst="H-1:refactoring:unrelated_rename",
                         evidence="same_pull_request"),
        ])

    hunk = Hunk(
        hunk_id="H-1",
        transformation=TransformationUnit(fn_id="fn-1", edit_regions=["H-1:ER-1"]),
        categories={c.value: ce for c, ce in categories.items()},
        relationships=relationships,
        provenance=_prov("fixture"),
    )

    sap = SAP(
        sap_id=sap_id, change_id="PR-1", schema_version="1.1",
        source_file="src/main/java/kafka/Validator.java",
        target_file="src/main/java/kafka/Validator.java",
        functions={"fn-1": fn}, hunks=[hunk], hunk_order=["H-1"],
        provenance=_prov("fixture", repo="apache/kafka", commit="a" * 40),
    )
    sap.add_payload(fn.source_before_ref, F_S)
    sap.add_payload(fn.source_after_ref, F_S_PRIME)
    sap.add_payload(fn.target_ref, F_T)
    sap.add_payload("hunks/H-1/hunk.diff", HUNK_DIFF)

    profile = Characterizer().characterize(categories)
    write_sap(sap, pr_dir / sap_id, profiles={"H-1": profile})

    group = PRGroup(
        pr_id="PR-1", variant_pair="linkedinKafka-apacheKafka",
        source_repo="apache/kafka", target_repo="linkedin/kafka",
        saps=[SAPReference(sap_id=sap_id, gacpd_classification="MO",
                           path=sap_id, hunk_count=1)],
    )
    write_pr_group(group, pr_dir)
    return pr_dir


@pytest.fixture(scope="session")
def pr_dir(tmp_path_factory) -> Path:
    return build_fixture_pr(tmp_path_factory.mktemp("sapout"))


@pytest.fixture(scope="session")
def sap_dir(pr_dir: Path) -> Path:
    return pr_dir / "sap-Validator"


@pytest.fixture(scope="session")
def pr_manifest(pr_dir: Path) -> dict:
    return json.loads((pr_dir / "pr.json").read_text(encoding="utf-8"))


F_S2 = """int compute(int a) {
    int base = lookup(a);
    return base + OFFSET;
}
"""
F_S2_PRIME = F_S2.replace("base + OFFSET", "base + OFFSET + 1")
F_T2 = F_S2.replace("lookup(a)", "lookupNode(a)")  # drifted fork copy


def _hunk(hunk_id: str, fn_id: str) -> Hunk:
    """A minimal mapped hunk carrying every schema-required category."""
    categories = {
        Category.SOURCE_CHANGE: CategoryEvidence(category=Category.SOURCE_CHANGE, elements=[
            _element(f"{hunk_id}:source_change:edit_region", "edit_region",
                     EvidenceState.PRESENT),
        ]),
        Category.TARGET_LOCALIZATION: CategoryEvidence(
            category=Category.TARGET_LOCALIZATION, elements=[
                _element(f"{hunk_id}:localization:target_function",
                         "localized_target_function", EvidenceState.PRESENT),
            ]),
        Category.FUNCTION_TRANSFORMATION: CategoryEvidence(
            category=Category.FUNCTION_TRANSFORMATION, elements=[
                _element(f"{hunk_id}:transformation:triple", "transformation_unit",
                         EvidenceState.PRESENT),
            ]),
        Category.STRUCTURAL: CategoryEvidence(category=Category.STRUCTURAL, elements=[
            _element(f"{hunk_id}:structural:edit_region_structure",
                     "edit_region_structure", EvidenceState.PRESENT),
        ]),
        Category.REFACTORING: CategoryEvidence(category=Category.REFACTORING, elements=[
            _element(f"{hunk_id}:refactoring:rename", "refactoring_findings",
                     EvidenceState.VERIFIED_ABSENT),
        ]),
        Category.COMPATIBILITY: CategoryEvidence(category=Category.COMPATIBILITY, elements=[
            _element(f"{hunk_id}:compatibility:source_apis", "source_apis",
                     EvidenceState.PRESENT),
        ]),
        Category.SURROUNDING: CategoryEvidence(category=Category.SURROUNDING, elements=[
            _element(f"{hunk_id}:surrounding:enclosing_context", "enclosing_context",
                     EvidenceState.PRESENT),
        ]),
        Category.VERIFICATION: CategoryEvidence(category=Category.VERIFICATION, elements=[
            _element(f"{hunk_id}:verification:covering_tests", "covering_tests",
                     EvidenceState.UNAVAILABLE),
        ]),
    }
    return Hunk(
        hunk_id=hunk_id,
        transformation=TransformationUnit(fn_id=fn_id, edit_regions=[f"{hunk_id}:ER-1"]),
        categories={c.value: ce for c, ce in categories.items()},
        relationships=[Relationship(src=f"{hunk_id}:ER-1", rel="aligned_to",
                                    dst=f"functions/{fn_id}")],
        provenance=_prov("fixture"),
    )


def build_composite_pr(root: Path) -> Path:
    """A composite SAP (section 13): fn-1 edited by H-1 (independent) and
    fn-2 edited by H-2 and H-3 (shared target function, no explicit link —
    the inferred-coupling shape real SALP output exhibits)."""
    pr_dir = root / "linkedinKafka-apacheKafka" / "PR-9"
    sap_id = "sap-Composite"

    fns = {
        "fn-1": (FunctionPayload(fn_id="fn-1", ext="java", has_source_before=True,
                                 has_source_after=True, has_target=True),
                 F_S, F_S_PRIME, F_T),
        "fn-2": (FunctionPayload(fn_id="fn-2", ext="java", has_source_before=True,
                                 has_source_after=True, has_target=True),
                 F_S2, F_S2_PRIME, F_T2),
    }
    hunks = [_hunk("H-1", "fn-1"), _hunk("H-2", "fn-2"), _hunk("H-3", "fn-2")]

    sap = SAP(
        sap_id=sap_id, change_id="PR-9", schema_version="1.1",
        source_file="src/main/java/kafka/Composite.java",
        target_file="src/main/java/kafka/Composite.java",
        functions={fid: f for fid, (f, *_) in fns.items()},
        hunks=hunks, hunk_order=["H-1", "H-2", "H-3"],
        provenance=_prov("fixture", repo="apache/kafka", commit="a" * 40),
    )
    for fn, before, after, target in fns.values():
        sap.add_payload(fn.source_before_ref, before)
        sap.add_payload(fn.source_after_ref, after)
        sap.add_payload(fn.target_ref, target)
    for h in hunks:
        sap.add_payload(f"hunks/{h.hunk_id}/hunk.diff",
                        "@@ -1,1 +1,1 @@\n-old\n+new\n")

    characterizer = Characterizer()
    profiles = {h.hunk_id: characterizer.characterize(
        {Category(c): ce for c, ce in h.categories.items()}) for h in hunks}
    write_sap(sap, pr_dir / sap_id, profiles=profiles)

    group = PRGroup(
        pr_id="PR-9", variant_pair="linkedinKafka-apacheKafka",
        source_repo="apache/kafka", target_repo="linkedin/kafka",
        saps=[SAPReference(sap_id=sap_id, gacpd_classification="MO",
                           path=sap_id, hunk_count=3)],
    )
    write_pr_group(group, pr_dir)
    return pr_dir


@pytest.fixture(scope="session")
def composite_pr_dir(tmp_path_factory) -> Path:
    return build_composite_pr(tmp_path_factory.mktemp("sapout-composite"))


@pytest.fixture(scope="session")
def composite_sap_dir(composite_pr_dir: Path) -> Path:
    return composite_pr_dir / "sap-Composite"


@pytest.fixture(scope="session")
def composite_pr_manifest(composite_pr_dir: Path) -> dict:
    return json.loads((composite_pr_dir / "pr.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def enriched_pr_dir(tmp_path_factory) -> Path:
    return build_fixture_pr(tmp_path_factory.mktemp("sapout-enriched"), enriched=True)


@pytest.fixture(scope="session")
def enriched_sap_dir(enriched_pr_dir: Path) -> Path:
    return enriched_pr_dir / "sap-Validator"
