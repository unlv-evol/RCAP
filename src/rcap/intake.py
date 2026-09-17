"""SAP intake: load and validate a SAP directory into a case model (RCAP ref section 3).

Reuses SALP as the schema authority: `salp.packaging.validate_sap_dir` is the
first gate, then the index files are loaded — never the program payloads (I7).
Unresolved mandatory references are a typed IntakeFailure with diagnostics, not
a silently dropped field. Evidence states are copied verbatim (I4);
characterization is attached as metadata only (I10); no prior integration
outcome is consulted (I12).
"""

from __future__ import annotations

import json
from pathlib import Path

from salp.packaging import validate_sap_dir

from rcap.model import (
    CaseModel,
    EvidenceRecord,
    FunctionEntry,
    RelationshipEdge,
    Transformation,
)

_FOUNDATIONAL = ("source_change", "target_localization", "function_transformation")


class IntakeFailure(Exception):
    """A SAP that cannot become a valid case model; carries stage attribution."""

    def __init__(self, sap_dir: Path, diagnostics: list[str]):
        self.stage = "intake"
        self.sap_dir = str(sap_dir)
        self.diagnostics = diagnostics
        super().__init__(f"intake failure for {sap_dir}: {len(diagnostics)} finding(s)")


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def repo_pin(provenance: object) -> dict | None:
    """The resolved repository pin out of a provenance dict, or None."""
    if isinstance(provenance, dict):
        pin = provenance.get("repository_pin")
        if isinstance(pin, dict) and pin.get("repo") and pin.get("commit"):
            return pin
    return None


def _bind_repo_state(evidence: dict[str, EvidenceRecord], provenance: dict,
                     pr_manifest: dict | None) -> tuple[dict, list[str], list[str]]:
    """Repository-state binding (RCAP ref section 3(3), section 4).

    Every evidence-level pin plus the SAP-level provenance pin is grouped per
    repository: one resolved commit per repo becomes the case binding;
    two different commits for the same repo are evidence bound to incompatible
    repository states — an intake failure (I2), never combined. The PR-manifest
    commit binding is reconciled, not enforced: the evidence-level resolved pin
    is authoritative and a lagging or UNAVAILABLE manifest binding is recorded.
    """
    by_repo: dict[str, dict[str, list[str]]] = {}

    def add(pin: dict, source: str) -> None:
        by_repo.setdefault(pin["repo"], {}).setdefault(pin["commit"], []).append(source)

    resolved_from: dict[str, str] = {}
    for rec in evidence.values():
        pin = repo_pin(rec.provenance)
        if pin:
            add(pin, rec.object_id)
            if pin.get("resolved_from"):
                resolved_from.setdefault(f"{pin['repo']}@{pin['commit']}",
                                         pin["resolved_from"])
    sap_pin = repo_pin(provenance)
    if sap_pin:
        add(sap_pin, "provenance.json")

    conflicts: list[str] = []
    bindings: dict[str, dict] = {}
    for repo, commits in sorted(by_repo.items()):
        if len(commits) > 1:
            detail = "; ".join(f"{c} <- {', '.join(sorted(srcs)[:3])}"
                               for c, srcs in sorted(commits.items()))
            conflicts.append(
                f"evidence bound to incompatible repository states for {repo}: {detail}")
            continue
        commit, sources = next(iter(commits.items()))
        bindings[repo] = {"commit": commit,
                          "resolved_from": resolved_from.get(f"{repo}@{commit}"),
                          "pinned_by": len(sources)}

    reconciliation: list[str] = []
    if bindings:
        manifest_binding = (pr_manifest or {}).get("commit_binding")
        if not isinstance(manifest_binding, dict):
            reconciliation.append(
                "repo-state reconciliation: PR manifest carries no commit binding "
                "(UNAVAILABLE); evidence-level resolved pins are authoritative")
        else:
            for repo, binding in bindings.items():
                declared = manifest_binding.get(repo)
                if declared != binding["commit"]:
                    reconciliation.append(
                        f"repo-state reconciliation: {repo} evidence-level pin "
                        f"{binding['commit']} is authoritative over PR-manifest "
                        f"binding {declared!r}")
    return bindings, reconciliation, conflicts


def load_case(sap_dir: str | Path, *, pr_manifest: dict | None = None) -> CaseModel:
    """Load one SAP directory into a validated, index-level case model.

    `pr_manifest` (the parsed pr.json) supplies repo identity and cross-file
    edges when the case may involve them; it is optional (RCAP ref section 3).
    """
    sap_dir = Path(sap_dir)

    # Gate 1: SALP's own schema conformance over the written package.
    report = validate_sap_dir(sap_dir)
    if not report.ok:
        raise IntakeFailure(sap_dir, report.errors)

    manifest = _read_json(sap_dir / "sap.json")
    change = _read_json(sap_dir / "change.json")
    provenance = _read_json(sap_dir / "provenance.json")
    characterization = {}
    if (sap_dir / "characterization.json").is_file():
        characterization = _read_json(sap_dir / "characterization.json")

    diagnostics: list[str] = []
    evidence: dict[str, EvidenceRecord] = {}
    relationships: list[RelationshipEdge] = []
    transformations: list[Transformation] = []
    category_confidence: dict[str, float] = {}

    for hunk_id in manifest.get("hunks", []):
        hdir = sap_dir / "hunks" / hunk_id
        index = _read_json(hdir / "hunk.json")

        # Foundational categories must be present as index entries (payloads not yet).
        states = index.get("evidence", {})
        if index.get("change_type") == "mapped":
            for cat in _FOUNDATIONAL:
                if cat not in states:
                    diagnostics.append(f"{hunk_id}: foundational category {cat} missing from index")

        tr = index.get("transformation", {})
        transformations.append(Transformation(
            hunk_id=hunk_id,
            fn_id=_fn_id_of(tr),
            edit_regions=list(tr.get("edit_regions", [])),
            f_s_ref=tr.get("f_s_before"),
            f_s_prime_ref=tr.get("f_s_after"),
            f_t_ref=tr.get("f_t"),
        ))

        for edge in index.get("relationships", []):
            relationships.append(RelationshipEdge(
                src=edge.get("from", ""), rel=edge.get("rel", ""), dst=edge.get("to", ""),
                state=edge.get("state", "PRESENT"), evidence=edge.get("evidence"),
                hunk_id=hunk_id,
            ))

        # The hunk.json evidence map is the index spine: every category the
        # SAP carries appears there with a state, so intake follows it rather
        # than a hardcoded filename list — a category is never silently
        # missing (I4, RCAP ref section 4). Entries without a resolvable
        # document (UNAVAILABLE, NOT_APPLICABLE, categories stored elsewhere)
        # become category-level records with their state preserved verbatim.
        for cat, entry in (index.get("evidence") or {}).items():
            ref = entry.get("ref")
            doc = None
            if ref:
                for base in (hdir, sap_dir):
                    if (base / ref).is_file():
                        doc = _read_json(base / ref)
                        break
            if doc and isinstance(doc.get("confidence"), (int, float)):
                # Section 9(7): index confidence is metadata for ranking,
                # never a retention gate.
                category_confidence[f"{hunk_id}:{cat}"] = float(doc["confidence"])
            if doc and doc.get("elements"):
                _collect_elements(doc, hunk_id, evidence)
            else:
                oid = f"{hunk_id}:{cat}"
                if oid not in evidence:
                    evidence[oid] = EvidenceRecord(
                        object_id=oid, element=cat, category=cat, hunk_id=hunk_id,
                        state=entry.get("state", ""),
                        blocking_conflict=bool(entry.get("blocking_conflict")),
                        attributes={"reason": entry["reason"]} if entry.get("reason") else {},
                    )

    # Section 13 cross-file relationships: explicit SALP-recorded semantic
    # edges from the PR manifest. A dependency is followed only through such
    # an edge; shared PR/commit/directory membership is never sufficient (I8)
    # — that stays enforced by the admissible-relationship policy.
    for edge in (pr_manifest or {}).get("cross_file_relationships") or []:
        relationships.append(RelationshipEdge(
            src=edge.get("from", ""), rel=edge.get("rel", ""), dst=edge.get("to", ""),
            state=edge.get("state", "PRESENT"), evidence=edge.get("evidence"),
            hunk_id="PR",
        ))

    functions = _load_functions(sap_dir, manifest)

    if diagnostics and any("foundational" in d for d in diagnostics):
        raise IntakeFailure(sap_dir, diagnostics)

    bindings, reconciliation, conflicts = _bind_repo_state(evidence, provenance,
                                                           pr_manifest)
    if conflicts:
        raise IntakeFailure(sap_dir, diagnostics + conflicts)
    diagnostics.extend(reconciliation)

    pr = pr_manifest or {}
    case = CaseModel(
        case_id=f"{pr.get('pr_id', sap_dir.parent.name)}/{manifest['sap_id']}",
        sap_id=manifest["sap_id"],
        change_id=manifest["change_id"],
        change_type=manifest["change_type"],
        schema_version=manifest["schema_version"],
        sap_dir=str(sap_dir),
        source_repo=pr.get("source_repo"),
        target_repo=pr.get("target_repo"),
        source_file=manifest.get("source_file"),
        target_file=manifest.get("target_file"),
        transformations=transformations,
        hunk_order=list(change.get("hunk_order", [])),
        functions=functions,
        evidence=evidence,
        relationships=relationships,
        characterization=characterization,
        category_confidence=category_confidence,
        repo_state={"bindings": bindings,
                    "reconciliation": reconciliation,
                    "provenance": provenance.get("repository_pin")},
        diagnostics=diagnostics,
    )

    # Endpoint resolution: an endpoint that resolves to nothing is a recorded
    # diagnostic, never a silently dropped edge (RCAP ref section 6).
    for edge in case.relationships:
        for raw in (edge.src, edge.dst):
            if case.resolve_endpoint(raw) is None:
                case.diagnostics.append(
                    f"{edge.hunk_id}: edge endpoint {raw!r} ({edge.rel}) resolves to nothing"
                )
    return case


def _fn_id_of(transformation: dict) -> str | None:
    for key in ("f_s_before", "f_s_after", "f_t"):
        ref = transformation.get(key)
        if isinstance(ref, str) and ref.startswith("functions/"):
            return ref.split("/")[1]
    return None


def _collect_elements(document: dict, hunk_id: str, out: dict[str, EvidenceRecord]) -> None:
    category = document.get("category", "")
    for e in document.get("elements", []):
        oid = e.get("object_id")
        if not oid:
            continue
        out[oid] = EvidenceRecord(
            object_id=oid,
            element=e.get("element", ""),
            category=category,
            hunk_id=hunk_id,
            state=e.get("state", ""),
            representation=e.get("representation"),
            payload_ref=e.get("payload_ref"),
            attributes=e.get("attributes") or {},
            blocking_conflict=bool(e.get("blocking_conflict")),
            provenance=e.get("provenance"),
        )


def _load_functions(sap_dir: Path, manifest: dict) -> dict[str, FunctionEntry]:
    functions: dict[str, FunctionEntry] = {}
    for fn_id in manifest.get("functions", []):
        fdir = sap_dir / "functions" / fn_id
        refs = {p.name: f"functions/{fn_id}/{p.name}" for p in sorted(fdir.iterdir())
                if p.is_file()} if fdir.is_dir() else {}
        functions[fn_id] = FunctionEntry(fn_id=fn_id, refs=refs)
    return functions
