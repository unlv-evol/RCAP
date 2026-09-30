"""Small spec-conformance items: signature-endpoint normalization (section 4),
relationship provenance_ref (section 9 schema), persisted context with hash
parity (section 2)."""

import hashlib
import json
import shutil

from rcap.config import ExecutionConfig
from rcap.context import build_context
from rcap.eval_harness import run_configs
from rcap.generate import StubBackend
from rcap.intake import load_case
from rcap.materialize import materialize
from rcap.pipeline import run_case
from rcap.reduction_program import reduce_program
from rcap.reduction_semantic import reduce_semantic
from rcap.selection import select

SIGNATURE = "void validate(Config)"


def test_signature_endpoint_normalized_to_structural_element(sap_dir, pr_manifest,
                                                             tmp_path):
    """Section 4: a signature string is not a valid endpoint; it is normalized
    to the object id of the structural element representing that method, and
    the normalization is recorded."""
    sap = tmp_path / "sap-SigEdge"
    shutil.copytree(sap_dir, sap)

    hunk_path = sap / "hunks" / "H-1" / "hunk.json"
    hunk = json.loads(hunk_path.read_text())
    hunk["relationships"].append({"from": SIGNATURE, "rel": "depends_on",
                                  "to": "H-1:ER-1", "state": "PRESENT"})
    hunk_path.write_text(json.dumps(hunk))

    element_path = sap / "hunks" / "H-1" / "edit_region.json"
    doc = json.loads(element_path.read_text())
    doc["elements"][0]["attributes"] = {**(doc["elements"][0].get("attributes") or {}),
                                        "method": SIGNATURE}
    element_path.write_text(json.dumps(doc))

    case = load_case(sap, pr_manifest=pr_manifest)
    edge = next(e for e in case.relationships if e.rel == "depends_on")
    element_oid = doc["elements"][0]["object_id"]
    assert edge.src == element_oid, "signature replaced by the element's object id"
    assert any("signature endpoint" in d and "normalized" in d
               for d in case.diagnostics)


def test_retained_relationships_carry_provenance_ref(enriched_sap_dir, pr_manifest):
    case = load_case(enriched_sap_dir, pr_manifest=pr_manifest)
    red = reduce_semantic(case, select(case))
    mat = materialize(case, red)
    prog = reduce_program(case, red, mat)
    ctx = build_context(case, red, mat, prog)
    assert ctx.relationships, "enriched fixture must retain edges"
    for rel in ctx.relationships:
        assert rel.provenance_ref and rel.provenance_ref.startswith("hunks/"), \
            "each retained relationship resolves back to its SAP index"


def test_persisted_context_matches_package_context_ref(sap_dir, pr_manifest,
                                                       tmp_path):
    """The exact context is persisted next to the package, compact, so
    sha256(context.json) == context_ref: the run is reconstructable from its
    outputs alone."""
    candidate = "```java\nvoid validate(Config c) { /* RCAP_PH_target.1 */ }\n```"
    run_configs(sap_dir, pr_manifest, StubBackend(candidate),
                artifacts_dir=tmp_path)
    persisted = (tmp_path / "full_rcap" / "context.json").read_bytes()

    result = run_case(sap_dir, pr_manifest, StubBackend(candidate),
                      ExecutionConfig())
    assert result.package is not None
    assert hashlib.sha256(persisted).hexdigest() == result.package.context_ref
