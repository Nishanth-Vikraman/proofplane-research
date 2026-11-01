from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from proofplane_v0.corpus.store import CorpusStore
from proofplane_v0.datasets.base import DatasetAdapter, DatasetCase
from proofplane_v0.kite.ast_units import function_nodes, source_segment
from proofplane_v0.kite.models import (
    FocalBenchmarkCase,
    FocalBenchmarkManifest,
    FocalMethodSpec,
)

BENCHMARK_VERSION = "kite-focal-benchmark-v1"

_NON_PRODUCTION_PARTS = {
    ".git",
    ".venv",
    "benchmarks",
    "docs",
    "docs_src",
    "examples",
    "oracle_tests",
    "scripts",
    "test",
    "tests",
    "testing",
}


def _production_patch_file(path: Path) -> bool:
    if path.suffix != ".py":
        return False
    lowered = [part.lower() for part in path.parts]
    if any(part in _NON_PRODUCTION_PARTS for part in lowered):
        return False
    if any(part.startswith("test_") or part.endswith("_tests") for part in lowered):
        return False
    return not (path.name.startswith("test_") or path.name.endswith("_test.py"))


def _semantic_ast(node: ast.AST) -> str:
    """Ignore source formatting while retaining decorators, annotations, and behavior."""
    return ast.dump(node, annotate_fields=True, include_attributes=False)


def localize_focal_methods(reference: Path, candidate: Path) -> list[FocalMethodSpec]:
    """Resolve fixed/reference methods changed by the hidden patch.

    This function is intentionally offline-only. Its result contains reference-side
    identity and integrity metadata, but no diff, buggy source, or mechanism labels.
    """
    localized: list[FocalMethodSpec] = []
    for reference_path in sorted(reference.rglob("*.py")):
        relative = reference_path.relative_to(reference)
        if not _production_patch_file(relative):
            continue
        candidate_path = candidate / relative
        reference_source = reference_path.read_text(encoding="utf-8", errors="replace")
        candidate_source = (
            candidate_path.read_text(encoding="utf-8", errors="replace")
            if candidate_path.is_file()
            else ""
        )
        if reference_source == candidate_source:
            continue
        reference_nodes = {name: node for name, node in function_nodes(reference_source)}
        candidate_nodes = {name: node for name, node in function_nodes(candidate_source)}
        for qualname, reference_node in sorted(reference_nodes.items()):
            candidate_node = candidate_nodes.get(qualname)
            if candidate_node is not None and _semantic_ast(reference_node) == _semantic_ast(
                candidate_node
            ):
                continue
            source = source_segment(reference_source, reference_node)
            localized.append(
                FocalMethodSpec(
                    file=relative.as_posix(),
                    qualname=qualname,
                    start_line=int(getattr(reference_node, "lineno", 1)),
                    end_line=int(
                        getattr(reference_node, "end_lineno", getattr(reference_node, "lineno", 1))
                    ),
                    reference_source_hash=hashlib.sha256(source.encode()).hexdigest(),
                )
            )
    return localized


def _eligible_case_ids(store: CorpusStore, split: str) -> list[str]:
    split_ids = {case_id for case_id, value in store.splits().items() if value == split}
    rows = store.connection.execute(
        "SELECT id,prepared,exclusion FROM dataset_cases ORDER BY id"
    ).fetchall()
    return [
        str(case_id)
        for case_id, prepared, exclusion in rows
        if case_id in split_ids and bool(prepared) and not str(exclusion)
    ]


def _case_entry(
    case: DatasetCase,
    reference: Path,
    candidate: Path,
    max_focal_methods: int,
) -> FocalBenchmarkCase:
    if not reference.is_dir() or not candidate.is_dir():
        return FocalBenchmarkCase(
            case_id=case.case_id,
            project=case.project,
            eligible=False,
            exclusion_reason="validated workspaces are missing",
        )
    methods = localize_focal_methods(reference, candidate)
    if not methods:
        reason = "patch does not resolve to a reference-side production Python method"
    elif len(methods) > max_focal_methods:
        reason = f"patch resolves to {len(methods)} focal methods; limit is {max_focal_methods}"
    else:
        reason = ""
    return FocalBenchmarkCase(
        case_id=case.case_id,
        project=case.project,
        eligible=bool(methods) and len(methods) <= max_focal_methods,
        focal_methods=methods if methods and len(methods) <= max_focal_methods else [],
        exclusion_reason=reason,
    )


def build_focal_benchmark(
    *,
    adapter: DatasetAdapter,
    store: CorpusStore,
    data_dir: Path,
    split: str,
    max_focal_methods: int,
    output: Path,
    limit: int = 0,
) -> FocalBenchmarkManifest:
    if max_focal_methods < 1:
        raise ValueError("max_focal_methods must be at least 1")
    cases = {case.case_id: case for case in adapter.list_cases()}
    selected = _eligible_case_ids(store, split)
    if limit:
        selected = selected[:limit]
    entries: list[FocalBenchmarkCase] = []
    for case_id in selected:
        case = cases.get(case_id)
        if case is None:
            continue
        root = data_dir / "workspaces" / case.case_id.replace(":", "-")
        entries.append(
            _case_entry(case, root / "reference", root / "candidate", max_focal_methods)
        )
    payload = [entry.model_dump(mode="json") for entry in entries]
    manifest = FocalBenchmarkManifest(
        split=split,
        max_focal_methods=max_focal_methods,
        cases=entries,
        content_hash=hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    return manifest


def load_focal_benchmark(path: Path) -> FocalBenchmarkManifest:
    return FocalBenchmarkManifest.model_validate_json(path.read_text(encoding="utf-8"))
