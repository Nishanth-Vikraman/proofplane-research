from __future__ import annotations

import csv
import hashlib
import json
import random
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from proofplane_v0.config import Settings
from proofplane_v0.corpus.store import CorpusStore
from proofplane_v0.datasets.base import DatasetAdapter, DatasetCase
from proofplane_v0.evaluation.docker_runner import (
    build_workspace_image,
    remove_workspace_image,
    run_test,
)
from proofplane_v0.kite.api import OpenAIService
from proofplane_v0.kite.benchmark import load_focal_benchmark
from proofplane_v0.kite.feedback import (
    feedback_payload,
    generate_tests,
    materialize_test,
    synthetic_feedback,
)
from proofplane_v0.kite.injection import VALIDATOR_VERSION as INJECTION_VALIDATOR_VERSION
from proofplane_v0.kite.injection import (
    apply_and_validate,
    injection_alignment_score,
    propose_injection,
)
from proofplane_v0.kite.knowledge import KNOWLEDGE_VERSION, build_mechanisms, knowledge_manifest
from proofplane_v0.kite.models import (
    InjectionPlan,
    InjectionValidation,
    KiteArmResult,
    KiteCaseResult,
    KiteTestDraft,
    RetrievalMatch,
    SyntheticTestAttempt,
    TargetCodeUnit,
    TransferableMechanism,
)
from proofplane_v0.kite.retrieval import RETRIEVAL_VERSION, retrieve
from proofplane_v0.kite.target_profile import (
    PROFILE_VERSION,
    SEMANTIC_PROFILE_VERSION,
    build_reference_profile,
    enrich_target_semantics,
)

KITE_ARMS = {
    "direct_reference",
    "direct_reference_repaired",
    "retrieved_mechanism_text",
    "retrieved_mechanism_sbf",
    "random_mechanism_sbf",
}
FEEDBACK_PROTOCOL_VERSION = "kite-shared-reference-suite-multi-injection-v2"


@dataclass(frozen=True)
class HiddenEvaluationHandle:
    """Opaque handle deliberately accepted only by the final real-bug evaluator."""

    workspace: Path
    image: str


def _safe_tag(case: DatasetCase, role: str) -> str:
    base = re.sub(r"[^a-z0-9_.-]", "-", f"{case.project}-{case.bug_id}".lower())
    return f"proofplane-kite-{base}-{role}:latest"


def _success_at_k(flags: list[bool], k: int) -> dict[str, bool]:
    requested = sorted({1, min(3, k), k})
    return {f"success@{value}": any(flags[:value]) for value in requested}


def _public_outcome(value: object) -> dict[str, object]:
    status = getattr(value, "status", "error")
    return {
        "status": str(status),
        "return_code": getattr(value, "return_code", None),
        "duration_seconds": float(getattr(value, "duration_seconds", 0.0)),
    }


def _evaluate_hidden(
    *,
    drafts: list[KiteTestDraft],
    arm: str,
    reference: Path,
    reference_image: str,
    hidden: HiddenEvaluationHandle,
    timeout: int,
    k: int,
) -> tuple[int, int, list[bool], list[dict[str, object]]]:
    """The only function in KITE-lite that can see the held-out real buggy workspace."""
    reference_passes = real_bug_kills = 0
    flags: list[bool] = []
    public_rows: list[dict[str, object]] = []
    for index, draft in enumerate(drafts[:k]):
        reference_path = materialize_test(reference, f"{arm}_final", index, draft)
        hidden_path = materialize_test(hidden.workspace, f"{arm}_final", index, draft)
        reference_out = run_test(reference, reference_path, timeout, image=reference_image)
        hidden_out = run_test(hidden.workspace, hidden_path, timeout, image=hidden.image)
        success = reference_out.status == "pass" and hidden_out.status == "fail"
        reference_passes += int(reference_out.status == "pass")
        real_bug_kills += int(success)
        flags.append(success)
        # Hidden stdout/stderr is intentionally never persisted into runtime artifacts.
        public_rows.append(
            {
                "candidate_index": index,
                "reference": _public_outcome(reference_out),
                "hidden_real_bug": _public_outcome(hidden_out),
                "success": success,
            }
        )
    flags.extend([False] * max(0, k - len(flags)))
    return reference_passes, real_bug_kills, flags, public_rows


def _arm_result_path(case_dir: Path, arm: str) -> Path:
    return case_dir / "arms" / f"{arm}.json"


def _run_non_feedback_arm(
    *,
    case: DatasetCase,
    arm: str,
    match: RetrievalMatch,
    service: OpenAIService,
    reference: Path,
    reference_image: str,
    hidden: HiddenEvaluationHandle,
    settings: Settings,
    k: int,
    case_dir: Path,
) -> KiteArmResult:
    mechanism = match.mechanism if arm == "retrieved_mechanism_text" else None
    drafts = generate_tests(
        service=service,
        arm=arm,
        unit=match.target_unit,
        mechanism=mechanism,
        k=k,
    )
    reference_passes, kills, flags, execution = _evaluate_hidden(
        drafts=drafts,
        arm=arm,
        reference=reference,
        reference_image=reference_image,
        hidden=hidden,
        timeout=settings.execution_timeout,
        k=k,
    )
    artifact = case_dir / "arms" / arm
    artifact.mkdir(parents=True, exist_ok=True)
    (artifact / "generated.json").write_text(
        json.dumps([draft.model_dump(mode="json") for draft in drafts], indent=2),
        encoding="utf-8",
    )
    (artifact / "final_evaluation.json").write_text(
        json.dumps(execution, indent=2), encoding="utf-8"
    )
    return KiteArmResult(
        case_id=case.case_id,
        arm=arm,
        retrieved_mechanism_ids=[match.mechanism.mechanism_id] if mechanism else [],
        target_unit_ids=[match.target_unit.unit_id],
        generated_count=len(drafts),
        reference_passes=reference_passes,
        real_bug_kills=kills,
        success_at_k=_success_at_k(flags, k),
    )


def _random_match(
    case: DatasetCase,
    primary: RetrievalMatch,
    mechanisms: list[TransferableMechanism],
) -> RetrievalMatch:
    eligible = [
        mechanism
        for mechanism in mechanisms
        if case.project not in (mechanism.source_projects or [mechanism.source_project])
        and mechanism.review_status != "rejected"
    ]
    generator = random.Random(hashlib.sha256(case.case_id.encode()).digest())
    mechanism = generator.choice(eligible)
    return primary.model_copy(
        update={
            "mechanism": mechanism,
            "semantic_score": 0.0,
            "structural_score": 0.0,
            "fused_score": 0.0,
        }
    )


AcceptedInjection = tuple[RetrievalMatch, InjectionPlan, InjectionValidation]


def _select_injections(
    *,
    matches: list[RetrievalMatch],
    service: OpenAIService,
    store: CorpusStore,
    reference: Path,
    reference_image: str,
    settings: Settings,
    artifact: Path,
) -> tuple[list[AcceptedInjection], int]:
    accepted: list[AcceptedInjection] = []
    accepted_patches: dict[str, int] = {}
    attempts = 0
    for match in matches[: settings.kite_max_injection_attempts]:
        attempts += 1
        attempt_dir = artifact / "injections" / f"attempt-{attempts}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        try:
            plan = propose_injection(match, service)
            (attempt_dir / "plan.json").write_text(
                plan.model_dump_json(indent=2), encoding="utf-8"
            )
            alignment_score = injection_alignment_score(match, plan, service)
            selection: dict[str, object] = {
                "mechanism_id": match.mechanism.mechanism_id,
                "alignment_score": alignment_score,
                "alignment_threshold": settings.kite_injection_alignment_threshold,
            }
            if alignment_score < settings.kite_injection_alignment_threshold:
                selection.update(
                    {
                        "selected": False,
                        "reason": "mechanism_alignment_below_threshold",
                    }
                )
                (attempt_dir / "selection.json").write_text(
                    json.dumps(selection, indent=2), encoding="utf-8"
                )
                continue
            validation = apply_and_validate(
                match=match,
                plan=plan,
                reference=reference,
                synthetic_root=settings.data_dir / "kite-synthetic",
                image=reference_image,
                settings=settings,
                store=store,
            )
            (attempt_dir / "validation.json").write_text(
                validation.model_dump_json(indent=2), encoding="utf-8"
            )
            if validation.accepted:
                patch_identity = hashlib.sha256(validation.patch.encode()).hexdigest()
                if patch_identity in accepted_patches:
                    selection.update(
                        {
                            "selected": False,
                            "reason": "duplicate_patch",
                            "duplicate_of_attempt": accepted_patches[patch_identity],
                        }
                    )
                    (attempt_dir / "selection.json").write_text(
                        json.dumps(selection, indent=2), encoding="utf-8"
                    )
                    continue
                accepted_patches[patch_identity] = attempts
                accepted.append((match, plan, validation))
                selection.update(
                    {
                        "selected": True,
                        "reason": "unique_aligned_valid_injection",
                        "patch_identity": patch_identity,
                    }
                )
            else:
                selection.update({"selected": False, "reason": "validator_rejected"})
            (attempt_dir / "selection.json").write_text(
                json.dumps(selection, indent=2), encoding="utf-8"
            )
            if (
                validation.accepted
                and len(accepted) >= settings.kite_max_accepted_injections
            ):
                break
        except Exception as error:  # noqa: BLE001 - a failed plan is a rejected bounded attempt
            (attempt_dir / "error.json").write_text(
                json.dumps({"type": type(error).__name__, "message": str(error)}, indent=2),
                encoding="utf-8",
            )
    return accepted, attempts


def _reference_filter(
    *,
    drafts: list[KiteTestDraft],
    reference: Path,
    image: str,
    timeout: int,
    arm: str,
    index_offset: int = 0,
) -> tuple[list[KiteTestDraft], list[dict[str, object]]]:
    passing: list[KiteTestDraft] = []
    feedback: list[dict[str, object]] = []
    for local_index, draft in enumerate(drafts):
        path = materialize_test(reference, arm, index_offset + local_index, draft)
        outcome = run_test(reference, path, timeout, image=image)
        if outcome.status == "pass":
            passing.append(draft)
        else:
            feedback.append(
                {
                    "rationale": draft.rationale,
                    "reference": {
                        "status": outcome.status,
                        "stdout": outcome.stdout[-1600:],
                        "stderr": outcome.stderr[-1600:],
                    },
                    "instruction": "Repair this test so it compiles, executes, and passes on reference.",
                }
            )
    return passing, feedback


def _suite_gate(
    *,
    drafts: list[KiteTestDraft],
    injections: list[AcceptedInjection],
    reference: Path,
    image: str,
    timeout: int,
    arm: str,
    round_number: int,
) -> tuple[bool, list[bool]]:
    if not drafts:
        return False, [False] * len(injections)
    killed: list[bool] = []
    reference_ok = True
    for index, (_match, _plan, validation) in enumerate(injections):
        attempts = synthetic_feedback(
            drafts=drafts,
            reference=reference,
            synthetic=Path(validation.synthetic_workspace),
            image=image,
            timeout=timeout,
            arm=f"{arm}_promotion_{index}",
            round_number=round_number,
        )
        reference_ok = reference_ok and all(value.reference.status == "pass" for value in attempts)
        killed.append(any(value.frozen for value in attempts))
    return reference_ok, killed


def _build_reference_suite(
    *,
    unit: TargetCodeUnit,
    service: OpenAIService,
    reference: Path,
    image: str,
    settings: Settings,
    k: int,
    arm_label: str,
) -> tuple[list[KiteTestDraft], int]:
    """Build the shared correct-side-filtered suite used by treatment and matched control."""
    base_drafts = generate_tests(
        service=service,
        arm="direct_reference",
        unit=unit,
        k=k,
    )
    generated_count = len(base_drafts)
    canonical, reference_feedback = _reference_filter(
        drafts=base_drafts,
        reference=reference,
        image=image,
        timeout=settings.execution_timeout,
        arm=f"{arm_label}_base",
    )
    base_round = 0
    while len(canonical) < k and base_round < settings.kite_max_test_repairs:
        base_round += 1
        missing = k - len(canonical)
        repair_drafts = generate_tests(
            service=service,
            arm="direct_reference",
            unit=unit,
            feedback=reference_feedback,
            current_suite=canonical,
            k=missing,
            round_number=base_round,
        )
        generated_count += len(repair_drafts)
        repaired, reference_feedback = _reference_filter(
            drafts=repair_drafts,
            reference=reference,
            image=image,
            timeout=settings.execution_timeout,
            arm=f"{arm_label}_base_repair_{base_round}",
        )
        canonical.extend(repaired[: k - len(canonical)])
    return canonical, generated_count


def _run_repaired_direct_arm(
    *,
    case: DatasetCase,
    match: RetrievalMatch,
    service: OpenAIService,
    reference: Path,
    reference_image: str,
    hidden: HiddenEvaluationHandle,
    settings: Settings,
    k: int,
    case_dir: Path,
    prepared_base_suite: tuple[list[KiteTestDraft], int] | None = None,
) -> KiteArmResult:
    arm = "direct_reference_repaired"
    artifact = case_dir / "arms" / arm
    artifact.mkdir(parents=True, exist_ok=True)
    if prepared_base_suite is None:
        canonical, generated_count = _build_reference_suite(
            unit=match.target_unit,
            service=service,
            reference=reference,
            image=reference_image,
            settings=settings,
            k=k,
            arm_label=arm,
        )
    else:
        canonical, generated_count = list(prepared_base_suite[0]), prepared_base_suite[1]
    reference_passes, kills, flags, execution = _evaluate_hidden(
        drafts=canonical,
        arm=arm,
        reference=reference,
        reference_image=reference_image,
        hidden=hidden,
        timeout=settings.execution_timeout,
        k=k,
    )
    (artifact / "frozen_tests.json").write_text(
        json.dumps([value.model_dump(mode="json") for value in canonical], indent=2),
        encoding="utf-8",
    )
    (artifact / "final_evaluation.json").write_text(
        json.dumps(execution, indent=2), encoding="utf-8"
    )
    return KiteArmResult(
        case_id=case.case_id,
        arm=arm,
        target_unit_ids=[match.target_unit.unit_id],
        generated_count=generated_count,
        reference_passes=reference_passes,
        real_bug_kills=kills,
        success_at_k=_success_at_k(flags, k),
    )


def _run_feedback_arm(
    *,
    case: DatasetCase,
    arm: str,
    matches: list[RetrievalMatch],
    service: OpenAIService,
    store: CorpusStore,
    reference: Path,
    reference_image: str,
    hidden: HiddenEvaluationHandle,
    settings: Settings,
    k: int,
    case_dir: Path,
    prepared_injections: list[AcceptedInjection] | None = None,
    prepared_attempts: int = 0,
    prepared_base_suite: tuple[list[KiteTestDraft], int] | None = None,
) -> KiteArmResult:
    artifact = case_dir / "arms" / arm
    artifact.mkdir(parents=True, exist_ok=True)
    accepted, attempts = prepared_injections or [], prepared_attempts
    if not accepted and not prepared_attempts:
        accepted, attempts = _select_injections(
            matches=matches,
            service=service,
            store=store,
            reference=reference,
            reference_image=reference_image,
            settings=settings,
            artifact=artifact,
        )
    if not matches:
        raise RuntimeError("synthetic-feedback arm requires at least one mechanism match")
    primary_match = accepted[0][0] if accepted else matches[0]
    if prepared_base_suite is None:
        canonical, generated_count = _build_reference_suite(
            unit=primary_match.target_unit,
            service=service,
            reference=reference,
            image=reference_image,
            settings=settings,
            k=k,
            arm_label=arm,
        )
    else:
        canonical, generated_count = list(prepared_base_suite[0]), prepared_base_suite[1]

    if not accepted:
        reference_passes, kills, flags, execution = _evaluate_hidden(
            drafts=canonical,
            arm=arm,
            reference=reference,
            reference_image=reference_image,
            hidden=hidden,
            timeout=settings.execution_timeout,
            k=k,
        )
        (artifact / "synthetic_feedback.json").write_text("[]\n", encoding="utf-8")
        (artifact / "frozen_tests.json").write_text(
            json.dumps([value.model_dump(mode="json") for value in canonical], indent=2),
            encoding="utf-8",
        )
        (artifact / "suite_history.json").write_text(
            json.dumps(
                [{"status": "no_valid_injection_base_suite_retained", "round": 0}],
                indent=2,
            ),
            encoding="utf-8",
        )
        (artifact / "final_evaluation.json").write_text(
            json.dumps(execution, indent=2), encoding="utf-8"
        )
        return KiteArmResult(
            case_id=case.case_id,
            arm=arm,
            retrieved_mechanism_ids=[match.mechanism.mechanism_id for match in matches],
            target_unit_ids=list(dict.fromkeys(match.target_unit.unit_id for match in matches)),
            injection_accepted=False,
            injection_attempts=attempts,
            generated_count=generated_count,
            reference_passes=reference_passes,
            real_bug_kills=kills,
            success_at_k=_success_at_k(flags, k),
            errors=["no valid local synthetic injection was accepted; base suite retained"],
        )

    feedback_attempts: list[SyntheticTestAttempt] = []
    killed_injections: list[AcceptedInjection] = []
    history: list[dict[str, object]] = []
    enhancement_rounds = 0
    promoted_tests = 0
    for injection_index, current in enumerate(accepted):
        match, plan, validation = current
        initial = synthetic_feedback(
            drafts=canonical,
            reference=reference,
            synthetic=Path(validation.synthetic_workspace),
            image=reference_image,
            timeout=settings.execution_timeout,
            arm=f"{arm}_injection_{injection_index}",
            round_number=0,
        )
        feedback_attempts.extend(initial)
        if initial and all(value.reference.status == "pass" for value in initial) and any(
            value.frozen for value in initial
        ):
            killed_injections.append(current)
            history.append(
                {"injection": injection_index, "status": "already_killed", "round": 0}
            )
            continue

        last_feedback = initial
        promoted = False
        for enhancement_round in range(1, settings.kite_max_enhancement_rounds + 1):
            enhancement_rounds += 1
            candidates = generate_tests(
                service=service,
                arm=arm,
                unit=match.target_unit,
                mechanism=match.mechanism,
                plan=plan,
                validation=validation,
                feedback=feedback_payload(last_feedback),
                current_suite=canonical,
                previously_killed_patches=[
                    value[2].patch for value in killed_injections
                ],
                k=k,
                round_number=enhancement_round,
            )
            generated_count += len(candidates)
            candidate_attempts = synthetic_feedback(
                drafts=candidates,
                reference=reference,
                synthetic=Path(validation.synthetic_workspace),
                image=reference_image,
                timeout=settings.execution_timeout,
                arm=f"{arm}_injection_{injection_index}",
                round_number=enhancement_round,
                index_offset=len(feedback_attempts),
            )
            feedback_attempts.extend(candidate_attempts)
            last_feedback = candidate_attempts
            for candidate_attempt in candidate_attempts:
                if not candidate_attempt.frozen:
                    continue
                candidate = candidate_attempt.candidate
                trials = (
                    [canonical + [candidate]]
                    if len(canonical) < k
                    else [
                        canonical[:replace] + [candidate] + canonical[replace + 1 :]
                        for replace in range(len(canonical))
                    ]
                )
                for trial in trials:
                    reference_ok, killed = _suite_gate(
                        drafts=trial,
                        injections=[*killed_injections, current],
                        reference=reference,
                        image=reference_image,
                        timeout=settings.execution_timeout,
                        arm=f"{arm}_gate_{injection_index}_{enhancement_round}",
                        round_number=enhancement_round,
                    )
                    if reference_ok and all(killed):
                        canonical = trial
                        promoted = True
                        promoted_tests += 1
                        break
                if promoted:
                    break
            if promoted:
                killed_injections.append(current)
                history.append(
                    {
                        "injection": injection_index,
                        "status": "promoted",
                        "round": enhancement_round,
                    }
                )
                break
        if not promoted:
            history.append(
                {
                    "injection": injection_index,
                    "status": "survived",
                    "round": settings.kite_max_enhancement_rounds,
                }
            )

    final_reference_ok, final_kills = _suite_gate(
        drafts=canonical,
        injections=accepted,
        reference=reference,
        image=reference_image,
        timeout=settings.execution_timeout,
        arm=f"{arm}_final_synthetic",
        round_number=settings.kite_max_enhancement_rounds + 1,
    )
    if not final_reference_ok:
        canonical = []
    reference_passes, kills, flags, execution = _evaluate_hidden(
        drafts=canonical,
        arm=arm,
        reference=reference,
        reference_image=reference_image,
        hidden=hidden,
        timeout=settings.execution_timeout,
        k=k,
    )
    (artifact / "synthetic_feedback.json").write_text(
        json.dumps([value.model_dump(mode="json") for value in feedback_attempts], indent=2),
        encoding="utf-8",
    )
    (artifact / "frozen_tests.json").write_text(
        json.dumps([value.model_dump(mode="json") for value in canonical], indent=2),
        encoding="utf-8",
    )
    (artifact / "suite_history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    (artifact / "final_evaluation.json").write_text(
        json.dumps(execution, indent=2), encoding="utf-8"
    )
    return KiteArmResult(
        case_id=case.case_id,
        arm=arm,
        retrieved_mechanism_ids=[value[0].mechanism.mechanism_id for value in accepted],
        target_unit_ids=list(dict.fromkeys(value[0].target_unit.unit_id for value in accepted)),
        injection_accepted=True,
        injection_attempts=attempts,
        generated_count=generated_count,
        synthetic_kills=sum(final_kills),
        unique_injections=len(accepted),
        enhancement_rounds=enhancement_rounds,
        promoted_tests=promoted_tests,
        reference_passes=reference_passes,
        real_bug_kills=kills,
        success_at_k=_success_at_k(flags, k),
        attempts=feedback_attempts,
    )


def _write_report(results: list[KiteCaseResult], output: Path, run_id: str) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    rows = [arm for case in results for arm in case.arms]
    jsonl = output / "results.jsonl"
    jsonl.write_text(
        "".join(case.model_dump_json() + "\n" for case in results), encoding="utf-8"
    )
    with (output / "summary.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=[
                "case_id",
                "arm",
                "injection_accepted",
                "unique_injections",
                "generated_count",
                "synthetic_kills",
                "enhancement_rounds",
                "promoted_tests",
                "reference_passes",
                "real_bug_kills",
                "success_at_3",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "case_id": row.case_id,
                    "arm": row.arm,
                    "injection_accepted": row.injection_accepted,
                    "unique_injections": row.unique_injections,
                    "generated_count": row.generated_count,
                    "synthetic_kills": row.synthetic_kills,
                    "enhancement_rounds": row.enhancement_rounds,
                    "promoted_tests": row.promoted_tests,
                    "reference_passes": row.reference_passes,
                    "real_bug_kills": row.real_bug_kills,
                    "success_at_3": row.success_at_k.get("success@3", False),
                }
            )
    by_arm: dict[str, list[KiteArmResult]] = {}
    for row in rows:
        by_arm.setdefault(row.arm, []).append(row)
    lines = [
        f"# KITE-lite experiment {run_id}",
        "",
        (
            "Strict protocol: offline benchmark construction exported only focal method identity; "
            "profiling, retrieval, injection, and feedback used the reference checkout only. The "
            "held-out real buggy checkout was accessed only by final scoring."
        ),
        "",
        "| Arm | Cases | Success@3 cases | Real-bug kills | Cases with injection | Unique injections | Synthetic kills | Enhancement rounds | Promotions |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm, values in sorted(by_arm.items()):
        lines.append(
            f"| {arm} | {len(values)} | "
            f"{sum(value.success_at_k.get('success@3', False) for value in values)} | "
            f"{sum(value.real_bug_kills for value in values)} | "
            f"{sum(value.injection_accepted is True for value in values)} | "
            f"{sum(value.unique_injections for value in values)} | "
            f"{sum(value.synthetic_kills for value in values)} | "
            f"{sum(value.enhancement_rounds for value in values)} | "
            f"{sum(value.promoted_tests for value in values)} |"
        )
    lines.extend(
        [
            "",
            (
                "The primary matched causal comparison is `retrieved_mechanism_sbf` versus "
                "`direct_reference_repaired`: both start from the same reference-filtered direct suite "
                "on the same selected target unit; only the former receives an executable synthetic "
                "counterfactual and bounded feedback. `retrieved_mechanism_text` remains a diagnostic "
                "for textual mechanism conditioning without executable feedback."
            ),
            (
                "Treatment activation is reported explicitly: unique injections remove duplicate patches; "
                "enhancement rounds count synthetic-guided generation calls; promotions count candidates "
                "that passed the reference/synthetic regression gate. If no injection is accepted, the arm "
                "retains and evaluates its comparable direct-reference base suite instead of returning zero tests."
            ),
        ]
    )
    report = output / "REPORT.md"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def run_kite_experiment(
    *,
    settings: Settings,
    adapter: DatasetAdapter,
    arms: list[str],
    limit: int,
    k: int,
    run_id: str | None = None,
    benchmark_manifest: Path | None = None,
    allow_budget_overrun: bool = False,
    continue_on_error: bool = True,
) -> Path:
    unknown = set(arms) - KITE_ARMS
    if unknown:
        raise ValueError(f"unknown KITE arms: {', '.join(sorted(unknown))}")
    run_id = run_id or datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-kite-lite"
    output = settings.artifacts_dir / run_id
    output.mkdir(parents=True, exist_ok=True)
    store = CorpusStore(settings.database)
    service = OpenAIService(store, settings, allow_budget_overrun=allow_budget_overrun)
    mechanisms = build_mechanisms(settings, store)
    (output / "knowledge-manifest.json").write_text(
        json.dumps(knowledge_manifest(mechanisms), indent=2), encoding="utf-8"
    )
    manifest_path = benchmark_manifest or settings.artifacts_dir / "kite-benchmark" / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(
            f"KITE focal benchmark is missing: {manifest_path}. Run `kite benchmark-build` first."
        )
    benchmark = load_focal_benchmark(manifest_path)
    if benchmark.split != "eval":
        raise ValueError(f"KITE experiment requires an eval benchmark, got {benchmark.split!r}")
    focal_cases = {value.case_id: value for value in benchmark.cases if value.eligible}
    protocol_manifest = {
        "feedback_protocol": FEEDBACK_PROTOCOL_VERSION,
        "knowledge_version": KNOWLEDGE_VERSION,
        "target_profile_version": PROFILE_VERSION,
        "target_semantic_profile_version": SEMANTIC_PROFILE_VERSION,
        "retrieval_version": RETRIEVAL_VERSION,
        "injection_validator_version": INJECTION_VALIDATOR_VERSION,
        "benchmark_content_hash": benchmark.content_hash,
        "arms": sorted(arms),
        "k": k,
        "model": settings.reasoning_model,
        "temperature": settings.kite_temperature,
        "llm_target_profile": settings.kite_llm_target_profile,
        "max_enhancement_rounds": settings.kite_max_enhancement_rounds,
        "max_accepted_injections": settings.kite_max_accepted_injections,
        "injection_alignment_threshold": settings.kite_injection_alignment_threshold,
    }
    protocol_path = output / "protocol-manifest.json"
    if protocol_path.is_file():
        existing_protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
        if existing_protocol != protocol_manifest:
            raise RuntimeError(
                "run-id belongs to a different KITE protocol/configuration; choose a new run-id"
            )
    else:
        protocol_path.write_text(json.dumps(protocol_manifest, indent=2), encoding="utf-8")
    (output / "benchmark-manifest.json").write_text(
        benchmark.model_dump_json(indent=2), encoding="utf-8"
    )
    eval_records = [
        record for record in store.eligible_records("eval") if record.corpus_id in focal_cases
    ]
    if limit:
        eval_records = eval_records[:limit]
    cases_by_id = {case.case_id: case for case in adapter.list_cases()}
    results: list[KiteCaseResult] = []
    store.put_run(
        run_id,
        "running",
        {
            "arms": arms,
            "k": k,
            "case_count": len(eval_records),
            "output": str(output),
            "benchmark_manifest": str(manifest_path),
            "benchmark_content_hash": benchmark.content_hash,
            "max_focal_methods": benchmark.max_focal_methods,
        },
    )
    for record in eval_records:
        case = cases_by_id.get(record.corpus_id)
        if case is None:
            continue
        case_dir = output / "cases" / case.case_id.replace(":", "-")
        case_dir.mkdir(parents=True, exist_ok=True)
        result_path = case_dir / "result.json"
        if result_path.exists():
            prior = KiteCaseResult.model_validate_json(result_path.read_text(encoding="utf-8"))
            if prior.status == "complete" and {value.arm for value in prior.arms} >= set(arms):
                results.append(prior)
                continue
        root = settings.data_dir / "workspaces" / case.case_id.replace(":", "-")
        reference, candidate = root / "reference", root / "candidate"
        reference_image, candidate_image = _safe_tag(case, "reference"), _safe_tag(case, "hidden")
        case_arms: list[KiteArmResult] = []
        errors: list[str] = []
        status: Literal["complete", "partial", "error"] = "error"
        try:
            if not reference.exists() or not candidate.exists():
                raise RuntimeError("validated reference/candidate workspaces are missing")
            profile = build_reference_profile(
                case,
                reference,
                store,
                settings.kite_max_target_units,
                settings.kite_max_unit_chars,
                focal_methods=focal_cases[case.case_id].focal_methods,
            )
            if settings.kite_llm_target_profile:
                profile = enrich_target_semantics(profile, service)
            matches = retrieve(profile, mechanisms, service, store, settings)
            (case_dir / "reference-profile.json").write_text(
                profile.model_dump_json(indent=2), encoding="utf-8"
            )
            (case_dir / "retrieval.json").write_text(
                json.dumps([value.model_dump(mode="json") for value in matches], indent=2),
                encoding="utf-8",
            )
            python_version = str(case.metadata.get("python_version", "3.12"))
            build_workspace_image(reference, reference_image, python_version)
            build_workspace_image(candidate, candidate_image, python_version)
            hidden = HiddenEvaluationHandle(candidate, candidate_image)
            shared_injections: list[AcceptedInjection] = []
            shared_attempts = 0
            if "retrieved_mechanism_sbf" in arms:
                shared_injections, shared_attempts = _select_injections(
                    matches=matches,
                    service=service,
                    store=store,
                    reference=reference,
                    reference_image=reference_image,
                    settings=settings,
                    artifact=case_dir / "shared-retrieved-injection",
                )
            comparison_match = shared_injections[0][0] if shared_injections else matches[0]
            shared_base_suite: tuple[list[KiteTestDraft], int] | None = None
            if {
                "direct_reference_repaired",
                "retrieved_mechanism_sbf",
            } & set(arms):
                shared_base_suite = _build_reference_suite(
                    unit=comparison_match.target_unit,
                    service=service,
                    reference=reference,
                    image=reference_image,
                    settings=settings,
                    k=k,
                    arm_label="shared_reference_suite",
                )
                shared_artifact = case_dir / "shared-reference-suite"
                shared_artifact.mkdir(parents=True, exist_ok=True)
                (shared_artifact / "frozen_tests.json").write_text(
                    json.dumps(
                        [
                            value.model_dump(mode="json")
                            for value in shared_base_suite[0]
                        ],
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                (shared_artifact / "metadata.json").write_text(
                    json.dumps(
                        {
                            "generated_count": shared_base_suite[1],
                            "frozen_count": len(shared_base_suite[0]),
                            "target_unit_id": comparison_match.target_unit.unit_id,
                            "content_hash": hashlib.sha256(
                                json.dumps(
                                    [
                                        value.model_dump(mode="json")
                                        for value in shared_base_suite[0]
                                    ],
                                    sort_keys=True,
                                    separators=(",", ":"),
                                ).encode()
                            ).hexdigest(),
                        },
                        indent=2,
                    ),
                    encoding="utf-8",
                )
            for arm in arms:
                arm_path = _arm_result_path(case_dir, arm)
                if arm_path.exists():
                    case_arms.append(
                        KiteArmResult.model_validate_json(arm_path.read_text(encoding="utf-8"))
                    )
                    continue
                try:
                    if arm in {"direct_reference", "retrieved_mechanism_text"}:
                        arm_result = _run_non_feedback_arm(
                            case=case,
                            arm=arm,
                            match=comparison_match,
                            service=service,
                            reference=reference,
                            reference_image=reference_image,
                            hidden=hidden,
                            settings=settings,
                            k=k,
                            case_dir=case_dir,
                        )
                    elif arm == "direct_reference_repaired":
                        arm_result = _run_repaired_direct_arm(
                            case=case,
                            match=comparison_match,
                            service=service,
                            reference=reference,
                            reference_image=reference_image,
                            hidden=hidden,
                            settings=settings,
                            k=k,
                            case_dir=case_dir,
                            prepared_base_suite=shared_base_suite,
                        )
                    else:
                        arm_matches = (
                            [_random_match(case, matches[0], mechanisms)]
                            if arm == "random_mechanism_sbf"
                            else matches
                        )
                        arm_result = _run_feedback_arm(
                            case=case,
                            arm=arm,
                            matches=arm_matches,
                            service=service,
                            store=store,
                            reference=reference,
                            reference_image=reference_image,
                            hidden=hidden,
                            settings=settings,
                            k=k,
                            case_dir=case_dir,
                            prepared_injections=(
                                shared_injections if arm == "retrieved_mechanism_sbf" else None
                            ),
                            prepared_attempts=(
                                shared_attempts if arm == "retrieved_mechanism_sbf" else 0
                            ),
                            prepared_base_suite=(
                                shared_base_suite
                                if arm == "retrieved_mechanism_sbf"
                                else None
                            ),
                        )
                    arm_path.parent.mkdir(parents=True, exist_ok=True)
                    arm_path.write_text(arm_result.model_dump_json(indent=2), encoding="utf-8")
                    case_arms.append(arm_result)
                except Exception as error:
                    arm_result = KiteArmResult(
                        case_id=case.case_id,
                        arm=arm,
                        errors=[f"{type(error).__name__}: {error}"],
                        success_at_k=_success_at_k([False] * k, k),
                    )
                    arm_path.parent.mkdir(parents=True, exist_ok=True)
                    arm_path.write_text(arm_result.model_dump_json(indent=2), encoding="utf-8")
                    case_arms.append(arm_result)
                    if not continue_on_error:
                        raise
            if len(case_arms) == len(arms):
                status = "complete"
            else:
                status = "partial"
        except Exception as error:
            errors.append(f"{type(error).__name__}: {error}")
            status = "error"
            if not continue_on_error:
                raise
        finally:
            remove_workspace_image(reference_image)
            remove_workspace_image(candidate_image)
        case_result = KiteCaseResult(
            case_id=case.case_id,
            project=case.project,
            arms=case_arms,
            status=status,
            errors=errors,
        )
        result_path.write_text(case_result.model_dump_json(indent=2), encoding="utf-8")
        results.append(case_result)
        _write_report(results, output, run_id)
        heartbeat = {
            "run_id": run_id,
            "status": "running",
            "completed_cases": len(results),
            "total_cases": len(eval_records),
            "last_case": case.case_id,
            "updated_utc": datetime.now(UTC).isoformat(),
        }
        (output / "heartbeat.json").write_text(json.dumps(heartbeat, indent=2), encoding="utf-8")
        store.put_run(run_id, "running", heartbeat)
    report = _write_report(results, output, run_id)
    final = {
        "run_id": run_id,
        "status": "complete",
        "completed_cases": len(results),
        "total_cases": len(eval_records),
        "updated_utc": datetime.now(UTC).isoformat(),
        "report": str(report),
    }
    (output / "heartbeat.json").write_text(json.dumps(final, indent=2), encoding="utf-8")
    store.put_run(run_id, "complete", final)
    return report
