from __future__ import annotations

import tempfile
import json
import re
import shutil
from datetime import UTC, datetime
from pathlib import Path

from proofplane_v0.config import Settings
from proofplane_v0.context.bundle import build_bundle
from proofplane_v0.corpus.enrich import OpenAIEnricher, deterministic_hidden, deterministic_visible
from proofplane_v0.corpus.store import CorpusStore
from proofplane_v0.datasets.base import DatasetAdapter
from proofplane_v0.datasets.synthetic import SyntheticAdapter
from proofplane_v0.evaluation.docker_runner import build_workspace_image, run_test
from proofplane_v0.evaluation.scorer import classify, syntax_valid
from proofplane_v0.experiments.buckets import evaluation_bucket
from proofplane_v0.experiments.report import build_report
from proofplane_v0.generation.arms import arm_spec
from proofplane_v0.generation.arms import CausalProgram, HistoricalEvidence
from proofplane_v0.generation.provider import (
    MockProvider,
    OpenAIProvider,
    build_generation_prompt,
    sanitized_evidence,
)
from proofplane_v0.models import CorpusBugRecord, EvaluationResult
from proofplane_v0.retrieval.hybrid import Retrieved, retrieve


class OracleNoMatchingFamilyError(RuntimeError):
    """The frozen training registry has no exact family match for this target."""


def oracle_family_matches(
    target: CorpusBugRecord, corpus: list[CorpusBugRecord], top_k: int
) -> list[CorpusBugRecord]:
    target_family = target.hidden_payload.bug_family.casefold()
    if target_family.startswith("unseen:"):
        return []
    return [
        record
        for record in corpus
        if record.hidden_payload.bug_family.casefold() == target_family
    ][:top_k]


def target_summary_evidence(target: CorpusBugRecord) -> list[HistoricalEvidence]:
    """Expose only the target's audited portable summary, never other hidden fields."""
    hidden = target.hidden_payload
    return [
        HistoricalEvidence(
            evidence_kind="portable_mechanism",
            summary=hidden.portable_prompt_summary,
        )
    ]


def _portable_causal_text(value: str) -> str:
    """Mechanically remove code-shaped tokens and exact numbers; no LLM is involved."""
    value = value.replace("/", " or ").replace("\\", " or ")
    value = re.sub(
        r"\bvalidation[- ]errors?\b", "input rejection details", value, flags=re.I
    )
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    value = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", value)
    value = value.replace("_", " ")
    value = re.sub(r"\b\d+(?:\.\d+)?\b", "a configured value", value)
    return re.sub(r"\s+", " ", value).strip()


def target_causal_program_evidence(target: CorpusBugRecord) -> list[HistoricalEvidence]:
    """Deterministically restructure existing enrichment; this performs no API call."""
    hidden = target.hidden_payload
    trigger = [_portable_causal_text(item) for item in hidden.trigger_sequence]
    return [
        HistoricalEvidence(
            evidence_kind="target_causal_program",
            summary="Deterministically structured abstract target-mechanism clue",
            causal_program=CausalProgram(
                preconditions=trigger[:1],
                action_sequence=trigger[1:-1] if len(trigger) > 2 else trigger[1:],
                state_transition=_portable_causal_text(hidden.mechanism_summary),
                lost_property=_portable_causal_text(hidden.violated_invariant),
                observable=_portable_causal_text(hidden.observable_symptom),
                test_shape=[_portable_causal_text(item) for item in hidden.test_strategy],
            ),
        )
    ]


def _workspace_root(settings: Settings, case_id: str) -> Path:
    return settings.data_dir / "workspaces" / case_id.replace(":", "-")


def unique_generated_path(relative_path: str, index: int) -> str:
    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", Path(relative_path).name)
    if safe_name.startswith("test_"):
        safe_name = safe_name[5:]
    return f"proofplane_generated/test_{index}_{safe_name}"


def success_at_k(success_flags: list[bool], values: list[int]) -> dict[str, bool]:
    return {
        str(value): bool(value <= len(success_flags) and any(success_flags[:value]))
        for value in values
    }


def execute_case(
    settings: Settings,
    adapter: DatasetAdapter,
    case_id: str,
    arm: str,
    k: int,
    mock: bool = False,
    unsafe_local_execution: bool = False,
    allow_budget_overrun: bool = False,
) -> tuple[EvaluationResult, Path]:
    spec = arm_spec(arm)
    store = CorpusStore(settings.database)
    case = adapter.get_case(case_id)
    target = next((r for r in store.records() if r.corpus_id == case_id), None)
    if target is None:
        raise RuntimeError(f"{case_id} has no enriched record")
    root = _workspace_root(settings, case_id)
    candidate_ws, reference_ws = root / "candidate", root / "reference"
    if not candidate_ws.exists() or not reference_ws.exists():
        raise RuntimeError(f"{case_id} is not prepared")
    if case.dataset == "bugsinpy":
        validation_path = (
            settings.artifacts_dir / "validations" / f"{case_id.replace(':', '-')}.json"
        )
        if not validation_path.exists() or not json.loads(
            validation_path.read_text(encoding="utf-8")
        ).get("valid"):
            raise RuntimeError(
                f"{case_id} has not passed counterfactual validation; run `proofplane dataset validate {case_id}`"
            )
    for workspace in (candidate_ws, reference_ws):
        generated_root = workspace / "proofplane_generated"
        if generated_root.exists():
            shutil.rmtree(generated_root)
    bundle = build_bundle(
        case_id,
        reference_ws,
        candidate_ws,
        settings.max_context_chars,
        set(),
        settings.max_nearby_tests,
        settings.max_test_snippet_chars,
        settings.max_repository_tree_entries,
    )
    query = (
        deterministic_visible(case, bundle)
        if mock
        else OpenAIEnricher(
            store,
            settings.fast_model,
            budget_usd=settings.api_budget_usd,
            reserve_usd=settings.api_call_reserve_usd,
            allow_budget_overrun=allow_budget_overrun,
            api_retry_attempts=settings.api_retry_attempts,
            api_retry_base_seconds=settings.api_retry_base_seconds,
        ).visible(bundle)
    )
    corpus = [
        r
        for r in store.eligible_records("train")
        if r.dataset == target.dataset and r.project != case.project
    ]
    if spec.retrieval_mode is not None and not spec.target_oracle and not corpus:
        raise RuntimeError("no causally isolated training records are indexed")
    retrieval_mode = spec.retrieval_mode or "none"
    found: list[Retrieved]
    if retrieval_mode == "none":
        found = []
    elif spec.target_oracle:
        found = []
    elif spec.oracle:
        ordered = oracle_family_matches(target, corpus, settings.retrieval_top_k)
        if not ordered:
            raise OracleNoMatchingFamilyError(
                f"no training record matches frozen family {target.hidden_payload.bug_family}"
            )
        found = [Retrieved(record, {"oracle": 1.0}) for record in ordered]
    else:
        found = retrieve(
            query,
            corpus,
            retrieval_mode,
            settings.retrieval_top_k,
            settings.hybrid_weights,
            settings.metadata_field_weights,
            settings.split_seed,
            settings.mmr_lambda,
            settings.candidate_pool_size,
        )
    case.metadata["runtime_context"] = bundle.model_dump(mode="json")
    provider = (
        MockProvider()
        if mock
        else OpenAIProvider(
            settings.fast_model,
            settings.api_budget_usd,
            allow_budget_overrun,
            store,
            settings.api_call_reserve_usd,
            settings.generation_max_output_tokens,
            api_retry_attempts=settings.api_retry_attempts,
            api_retry_base_seconds=settings.api_retry_base_seconds,
        )
    )
    if arm == "oracle_target_causal_program":
        evidence = target_causal_program_evidence(target)
    elif spec.target_oracle:
        evidence = target_summary_evidence(target)
    else:
        evidence = sanitized_evidence(found, arm)
    prompt = build_generation_prompt(case, arm, evidence, k)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + f"-{case.project}-{case.bug_id}-{arm}"
    artifact = settings.artifacts_dir / run_id
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact / "context.json").write_text(bundle.model_dump_json(indent=2), encoding="utf-8")
    (artifact / "generation_input.json").write_text(
        json.dumps(
            {
                "arm": arm,
                "retrieval_mode": retrieval_mode,
                "historical_evidence": [item.model_dump(mode="json") for item in evidence],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (artifact / "prompt.txt").write_text(prompt, encoding="utf-8")
    (artifact / "retrieval.json").write_text(
        json.dumps(
            [{"id": item.record.corpus_id, "scores": item.scores} for item in found], indent=2
        ),
        encoding="utf-8",
    )
    try:
        generated = provider.generate(case, arm, evidence, k)
    except Exception as error:
        (artifact / "generation_error.json").write_text(
            json.dumps(
                {"type": type(error).__name__, "message": str(error)},
                indent=2,
            ),
            encoding="utf-8",
        )
        raise
    for index, item in enumerate(generated):
        item.relative_path = unique_generated_path(item.relative_path, index)
    (artifact / "generated.json").write_text(
        json.dumps([item.model_dump(mode="json") for item in generated], indent=2), encoding="utf-8"
    )
    valid_candidates = [item for item in generated if syntax_valid(item.test_code)]
    for item in valid_candidates:
        for workspace in (candidate_ws, reference_ws):
            path = workspace / item.relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(item.test_code, encoding="utf-8")
    candidate_image = reference_image = "proofplane-runner:py312"
    if not unsafe_local_execution:
        python_version = str(case.metadata.get("python_version", "3.12"))
        candidate_image = f"proofplane-{case.project}-{case.bug_id}-candidate:latest"
        reference_image = f"proofplane-{case.project}-{case.bug_id}-reference:latest"
        build_workspace_image(candidate_ws, candidate_image, python_version)
        build_workspace_image(reference_ws, reference_image, python_version)
    success = runnable = candidate_fails = reference_passes = 0
    success_flags = [False] * len(generated)
    errors: list[str] = []
    total_candidate_time = total_reference_time = 0.0
    for item in valid_candidates:
        candidate_out = run_test(
            candidate_ws,
            item.relative_path,
            settings.execution_timeout,
            unsafe_local_execution,
            candidate_image,
        )
        reference_out = run_test(
            reference_ws,
            item.relative_path,
            settings.execution_timeout,
            unsafe_local_execution,
            reference_image,
        )
        total_candidate_time += candidate_out.duration_seconds
        total_reference_time += reference_out.duration_seconds
        runnable += int(
            candidate_out.status in {"pass", "fail"} and reference_out.status in {"pass", "fail"}
        )
        candidate_fails += int(candidate_out.status == "fail")
        reference_passes += int(reference_out.status == "pass")
        category = classify(candidate_out, reference_out)
        success += int(category == "candidate-only failure")
        success_flags[generated.index(item)] = category == "candidate-only failure"
        if category not in {"candidate-only failure", "pass on both versions"}:
            errors.append(f"{item.candidate_id}: {category}")
    paths = [item.relative_path for item in valid_candidates]
    if paths:
        combined_candidate = run_test(
            candidate_ws,
            paths[0] if len(paths) == 1 else "proofplane_generated",
            settings.execution_timeout,
            unsafe_local_execution,
            candidate_image,
        )
        combined_reference = run_test(
            reference_ws,
            paths[0] if len(paths) == 1 else "proofplane_generated",
            settings.execution_timeout,
            unsafe_local_execution,
            reference_image,
        )
        (artifact / "combined_execution.json").write_text(
            json.dumps(
                {
                    "candidate": combined_candidate.model_dump(mode="json"),
                    "reference": combined_reference.model_dump(mode="json"),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    result = EvaluationResult(
        case_id=case_id,
        split=target.split,
        evaluation_bucket=evaluation_bucket(
            target.hidden_payload, [r.hidden_payload for r in corpus]
        ),
        generation_arm=arm,
        retrieval_mode=retrieval_mode,
        retrieval_parameters={"top_k": settings.retrieval_top_k, "mmr_lambda": settings.mmr_lambda},
        retrieved_corpus_ids=[item.record.corpus_id for item in found],
        retrieval_scores=[item.scores for item in found],
        generated_test_count=len(generated),
        syntax_valid_count=len(valid_candidates),
        runnable_count=runnable,
        tests_failing_on_candidate=candidate_fails,
        tests_passing_on_reference=reference_passes,
        successful_regression_revealing_tests=success,
        success_at_k=success_at_k(success_flags, settings.generation_k),
        execution_times={"candidate": total_candidate_time, "reference": total_reference_time},
        errors=errors,
    )
    build_report([result], artifact, store.api_usage())
    return result, artifact


def run_experiment(
    settings: Settings,
    adapter: DatasetAdapter,
    arms: list[str],
    limit: int,
    mock: bool,
    unsafe_local_execution: bool,
    allow_budget_overrun: bool,
    k: int = 1,
    continue_on_error: bool = False,
) -> Path:
    available = adapter.list_cases()
    dataset = available[0].dataset if available else ""
    eval_ids = [r.corpus_id for r in store_records(settings, "eval") if r.dataset == dataset]
    if limit:
        eval_ids = eval_ids[:limit]
    if not eval_ids:
        raise RuntimeError("no enriched evaluation records")
    results: list[EvaluationResult] = []
    for case_id in eval_ids:
        for arm in arms:
            try:
                result, _ = execute_case(
                    settings,
                    adapter,
                    case_id,
                    arm,
                    k,
                    mock,
                    unsafe_local_execution,
                    allow_budget_overrun,
                )
            except Exception as error:
                if not continue_on_error:
                    raise
                spec = arm_spec(arm)
                result = EvaluationResult(
                    case_id=case_id,
                    split="eval",
                    evaluation_bucket="unclassified",
                    generation_arm=arm,
                    retrieval_mode=spec.retrieval_mode or "none",
                    generated_test_count=0,
                    errors=[f"{type(error).__name__}: {error}"],
                    exclusions=["arm execution aborted before scoring"],
                )
            results.append(result)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-experiment"
    return build_report(
        results, settings.artifacts_dir / run_id, CorpusStore(settings.database).api_usage()
    )


def store_records(settings: Settings, split: str) -> list[CorpusBugRecord]:
    return CorpusStore(settings.database).eligible_records(split)


def smoke(settings: Settings, output: Path | None = None) -> Path:
    adapter, provider = SyntheticAdapter(), MockProvider()
    cases = adapter.list_cases()
    records: list[CorpusBugRecord] = []
    with tempfile.TemporaryDirectory(prefix="proofplane-") as temp:
        root = Path(temp)
        bundles = {}
        for case in cases:
            candidate, reference = adapter.materialize(case, root / case.project)
            bundles[case.case_id] = (
                candidate,
                reference,
                build_bundle(
                    case.case_id,
                    reference,
                    candidate,
                    settings.max_context_chars,
                    set(case.hidden_tests),
                    settings.max_nearby_tests,
                    settings.max_test_snippet_chars,
                    settings.max_repository_tree_entries,
                ),
            )
            record = CorpusBugRecord(
                dataset=case.dataset,
                project=case.project,
                bug_id=case.bug_id,
                split="train",
                visible_descriptor=deterministic_visible(case, bundles[case.case_id][2]),
                hidden_payload=deterministic_hidden(case),
                prompt_versions={"visible": "mock-v1", "hidden": "mock-v1"},
            )
            records.append(record)
        store = CorpusStore(settings.database)
        for record in records:
            store.put(record)
        results: list[EvaluationResult] = []
        for index, case in enumerate(cases):
            candidate_ws, reference_ws, bundle = bundles[case.case_id]
            query = deterministic_visible(case, bundle)
            corpus = [r for i, r in enumerate(records) if i != index]
            found = retrieve(
                query,
                corpus,
                "hybrid_mmr",
                min(2, len(corpus)),
                settings.hybrid_weights,
                settings.metadata_field_weights,
                mmr_lambda=settings.mmr_lambda,
                candidate_pool_size=settings.candidate_pool_size,
            )
            generated = provider.generate(
                case,
                "retrieved_mechanisms_mmr",
                sanitized_evidence(found, "retrieved_mechanisms_mmr"),
                1,
            )
            success = runnable = valid = candidate_fails = reference_passes = 0
            for test in generated:
                if not syntax_valid(test.test_code):
                    continue
                valid += 1
                for workspace in (candidate_ws, reference_ws):
                    (workspace / test.relative_path).write_text(test.test_code, encoding="utf-8")
                candidate_out = run_test(
                    candidate_ws, test.relative_path, settings.execution_timeout, True
                )
                reference_out = run_test(
                    reference_ws, test.relative_path, settings.execution_timeout, True
                )
                runnable += int(
                    candidate_out.status in {"pass", "fail"}
                    and reference_out.status in {"pass", "fail"}
                )
                candidate_fails += int(candidate_out.status == "fail")
                reference_passes += int(reference_out.status == "pass")
                success += int(classify(candidate_out, reference_out) == "candidate-only failure")
            results.append(
                EvaluationResult(
                    case_id=case.case_id,
                    split="eval",
                    evaluation_bucket=evaluation_bucket(
                        deterministic_hidden(case), [r.hidden_payload for r in corpus]
                    ),
                    generation_arm="retrieved_mechanisms_mmr",
                    retrieval_mode="hybrid_mmr",
                    retrieved_corpus_ids=[x.record.corpus_id for x in found],
                    retrieval_scores=[x.scores for x in found],
                    generated_test_count=len(generated),
                    syntax_valid_count=valid,
                    runnable_count=runnable,
                    tests_failing_on_candidate=candidate_fails,
                    tests_passing_on_reference=reference_passes,
                    successful_regression_revealing_tests=success,
                    success_at_k={"1": bool(success)},
                )
            )
    destination = output or settings.artifacts_dir / "synthetic-smoke"
    return build_report(results, destination)
