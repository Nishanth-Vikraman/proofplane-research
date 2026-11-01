from __future__ import annotations

import hashlib
import json
from pathlib import Path

from proofplane_v0.config import Settings
from proofplane_v0.context.bundle import build_bundle
from proofplane_v0.corpus.enrich import OpenAIEnricher, deterministic_hidden, deterministic_visible
from proofplane_v0.corpus.store import CorpusStore
from proofplane_v0.datasets.base import DatasetAdapter, DatasetCase
from proofplane_v0.datasets.bugsinpy import BugsInPyAdapter
from proofplane_v0.datasets.synthetic import SyntheticAdapter
from proofplane_v0.experiments.splits import assert_isolated, project_split
from proofplane_v0.models import CorpusBugRecord, Provenance
from proofplane_v0.retrieval.embeddings import deterministic_embedding, serialize


def adapter_for(name: str, root: Path) -> DatasetAdapter:
    if name == "synthetic":
        return SyntheticAdapter()
    if name == "bugsinpy":
        return BugsInPyAdapter(root)
    raise ValueError(f"unknown adapter: {name}")


def register_dataset(adapter: DatasetAdapter, store: CorpusStore) -> list[DatasetCase]:
    cases = adapter.list_cases()
    for case in cases:
        store.put_case(case.case_id, case.dataset, case.project, case.bug_id, case.metadata)
    return cases


def prepare_cases(
    adapter: DatasetAdapter,
    store: CorpusStore,
    settings: Settings,
    case_ids: list[str],
    limit: int,
    continue_on_error: bool = False,
) -> list[str]:
    cases = register_dataset(adapter, store)
    selected = [case for case in cases if not case_ids or case.case_id in case_ids]
    if limit:
        selected = selected[:limit]
    prepared: list[str] = []
    for case in selected:
        target = settings.data_dir / "workspaces" / case.case_id.replace(":", "-")
        preparation_path = (
            settings.artifacts_dir
            / "preparations"
            / f"{case.case_id.replace(':', '-')}.json"
        )
        try:
            adapter.materialize(case, target)
            store.mark_prepared(case.case_id)
            if preparation_path.exists():
                preparation_path.unlink()
            prepared.append(case.case_id)
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            if isinstance(adapter, BugsInPyAdapter):
                # Materialization is infrastructure, not evidence that a published case is bad.
                store.mark_prepared(case.case_id, False, "")
                preparation_path.parent.mkdir(parents=True, exist_ok=True)
                preparation_path.write_text(
                    json.dumps(
                        {
                            "case_id": case.case_id,
                            "status": "infrastructure_error",
                            "error": message,
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            else:
                store.mark_prepared(case.case_id, False, message)
            if not continue_on_error:
                raise
    return prepared


def create_splits(adapter: DatasetAdapter, store: CorpusStore, seed: int) -> dict[str, str]:
    cases = register_dataset(adapter, store)
    values = project_split(cases, seed)
    assert_isolated(cases, values)
    store.put_splits(values)
    return values


def _case_paths(settings: Settings, case: DatasetCase) -> tuple[Path, Path, Path]:
    root = settings.data_dir / "workspaces" / case.case_id.replace(":", "-")
    return root / "candidate", root / "reference", root / "oracle_tests"


def enrich_split(
    adapter: DatasetAdapter,
    store: CorpusStore,
    settings: Settings,
    split: str,
    limit: int,
    mock: bool = False,
    case_ids: list[str] | None = None,
    allow_budget_overrun: bool = False,
) -> list[str]:
    cases = adapter.list_cases()
    splits = store.splits()
    if not splits:
        splits = create_splits(adapter, store, settings.split_seed)
    selected = [
        case
        for case in cases
        if splits.get(case.case_id) == split and (not case_ids or case.case_id in case_ids)
    ]
    if cases and cases[0].dataset == "bugsinpy":
        validated: list[DatasetCase] = []
        for case in selected:
            validation_path = (
                settings.artifacts_dir
                / "validations"
                / f"{case.case_id.replace(':', '-')}.json"
            )
            if not validation_path.exists():
                continue
            try:
                if json.loads(validation_path.read_text(encoding="utf-8")).get("valid"):
                    validated.append(case)
            except (json.JSONDecodeError, OSError):
                continue
        selected = validated
    if limit:
        selected = selected[:limit]
    if not selected:
        raise RuntimeError(f"no {split} cases selected")
    enricher = (
        None
        if mock
        else OpenAIEnricher(
            store,
            settings.reasoning_model,
            budget_usd=settings.api_budget_usd,
            reserve_usd=settings.api_call_reserve_usd,
            allow_budget_overrun=allow_budget_overrun,
            api_retry_attempts=settings.api_retry_attempts,
            api_retry_base_seconds=settings.api_retry_base_seconds,
        )
    )
    enriched: list[str] = []
    for case in selected:
        candidate, reference, oracle_root = _case_paths(settings, case)
        if not candidate.exists() or not reference.exists():
            raise RuntimeError(
                f"{case.case_id} is not prepared; run dataset prepare with this case ID"
            )
        # Oracle tests live outside both workspaces; recorded pre-fix tests remain visible.
        bundle = build_bundle(
            case.case_id,
            reference,
            candidate,
            settings.max_context_chars,
            set(),
            settings.max_nearby_tests,
            settings.max_test_snippet_chars,
            settings.max_repository_tree_entries,
        )
        context_path = (
            settings.artifacts_dir / "contexts" / f"{case.case_id.replace(':', '-')}.json"
        )
        context_path.parent.mkdir(parents=True, exist_ok=True)
        context_path.write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
        oracle_tests = (
            {
                path.relative_to(oracle_root).as_posix(): path.read_text(
                    encoding="utf-8", errors="replace"
                )
                for path in oracle_root.rglob("*.py")
            }
            if oracle_root.exists()
            else case.hidden_tests
        )
        bug_dir = Path(str(case.metadata.get("bug_dir", "")))
        patch_path = bug_dir / "bug_patch.txt"
        patch = (
            patch_path.read_text(encoding="utf-8", errors="replace") if patch_path.exists() else ""
        )
        if mock:
            if case.dataset != "synthetic":
                raise RuntimeError("mock enrichment is supported only for the synthetic adapter")
            visible, hidden = deterministic_visible(case, bundle), deterministic_hidden(case)
        else:
            assert enricher is not None
            visible = enricher.visible(bundle)
            hidden = enricher.hidden(case, bundle, oracle_tests, patch, split)
        record = CorpusBugRecord(
            dataset=case.dataset,
            project=case.project,
            bug_id=case.bug_id,
            split=split,
            visible_descriptor=visible,
            hidden_payload=hidden,
            provenance=Provenance(
                buggy_commit=str(case.metadata.get("buggy_commit_id", "")),
                fixed_commit=str(case.metadata.get("fixed_commit_id", "")),
                test_files=list(case.metadata.get("test_files", [])),
                source=str(bug_dir),
            ),
            source_hashes={
                "context": bundle.content_hash,
                "patch": hashlib.sha256(patch.encode()).hexdigest(),
            },
            prompt_versions={"visible": "visible-v1", "hidden": "hidden-v2-registry"},
        )
        store.put(record)
        enriched.append(record.corpus_id)
    return enriched


def build_index(store: CorpusStore, output: Path) -> int:
    # The retrieval corpus is historical training data only. Dev/eval descriptors may be
    # enriched for analysis and target queries, but must never enter the primary index.
    records = store.eligible_records("train")
    store.connection.execute("DELETE FROM embeddings WHERE kind != 'oracle_hidden'")
    store.connection.commit()
    for record in records:
        descriptor = record.visible_descriptor
        values = {
            "descriptor": descriptor.retrieval_text,
            "code_change": descriptor.visible_change_summary,
            "sequence": " ".join(descriptor.event_sequence),
            "component_operation": " ".join(
                descriptor.component_roles + descriptor.operations_changed
            ),
        }
        for kind, text in values.items():
            vector = deterministic_embedding(text)
            store.put_embedding(record.corpus_id, kind, serialize(vector), len(vector))
    export = store.primary_index_export("train")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(export, indent=2), encoding="utf-8")
    return len(export)
