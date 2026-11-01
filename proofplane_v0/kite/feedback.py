from __future__ import annotations

from pathlib import Path

from pydantic import Field

from proofplane_v0.evaluation.docker_runner import run_test
from proofplane_v0.kite.api import OpenAIService
from proofplane_v0.kite.models import (
    InjectionPlan,
    InjectionValidation,
    KiteTestDraft,
    SyntheticTestAttempt,
    TargetCodeUnit,
    TransferableMechanism,
)
from proofplane_v0.models import StrictModel

TEST_PROMPT_VERSION = "kite-test-generation-v3-visible-api-grounded"
REPAIR_PROMPT_VERSION = "kite-test-repair-v3-visible-api-grounded"


def _unit_payload(unit: TargetCodeUnit) -> dict[str, object]:
    return {
        "file": unit.file,
        "qualname": unit.qualname,
        "signature": unit.signature,
        "docstring": unit.docstring,
        "source": unit.source,
        "semantic_profile": {
            "what": unit.semantic_what,
            "where": unit.semantic_where,
            "how": unit.semantic_how,
        },
        "related_pre_existing_tests": unit.related_tests,
        "repository_context": unit.repository_context.model_dump(mode="json"),
        "structural": {
            "coarse_key": unit.structural.coarse_key,
            "fingerprint": unit.structural.fingerprint,
            "signatures": unit.structural.signatures,
            "query_units": [
                {
                    "coarse_key": value.coarse_key,
                    "fingerprint": value.fingerprint,
                    "signatures": value.signatures,
                    "relations": value.relations,
                }
                for value in unit.structural_queries
            ],
        },
    }


def _mechanism_payload(mechanism: TransferableMechanism | None) -> dict[str, object] | None:
    if mechanism is None:
        return None
    return {
        "mechanism_id": mechanism.mechanism_id,
        "what": mechanism.what,
        "where": mechanism.where,
        "how": mechanism.how,
        "expected_buggy_behavior": mechanism.expected_buggy_behavior,
        "representative_evidence": mechanism.representative_evidence,
        "supporting_bug_unit_count": len(mechanism.supporting_bug_unit_ids),
        "preconditions": mechanism.preconditions,
        "actions": mechanism.actions,
        "transition": mechanism.transition,
        "violated_property": mechanism.violated_property,
        "observable": mechanism.observable,
        "test_shape": mechanism.test_shape,
        "semantic_group": mechanism.semantic_group,
    }


def generate_tests(
    *,
    service: OpenAIService,
    arm: str,
    unit: TargetCodeUnit,
    k: int,
    mechanism: TransferableMechanism | None = None,
    plan: InjectionPlan | None = None,
    validation: InjectionValidation | None = None,
    feedback: list[dict[str, object]] | None = None,
    current_suite: list[KiteTestDraft] | None = None,
    previously_killed_patches: list[str] | None = None,
    round_number: int = 0,
) -> list[KiteTestDraft]:
    if arm == "direct_reference":
        arm_instruction = (
            "Generate diverse high-value pytest regression tests directly from the reference code and "
            "its pre-existing tests. Look for boundary, state, lifecycle, ordering, and interaction contracts."
        )
        if round_number:
            arm_instruction += (
                " Repair the failing canonical-suite candidates using the supplied correct-side execution "
                "feedback. Every replacement must compile, execute, and pass on the reference."
            )
    elif arm == "retrieved_mechanism_text":
        arm_instruction = (
            "Use the historical portable mechanism only as transferable test-design inspiration. State "
            "a target-specific hypothesis in each rationale, then test that hypothesis. Historical details "
            "are not facts about the target."
        )
    elif arm in {"retrieved_mechanism_sbf", "random_mechanism_sbf"}:
        arm_instruction = (
            "Use the concrete synthetic patch as executable counterfactual feedback. Generate tests whose "
            "intended objective is PASS on the supplied reference implementation and FAIL on the supplied "
            "synthetic variant. You have full target source, the local patch, mechanism, trigger recipe, and "
            "feedback; do not ignore target APIs or invent inaccessible setup."
            " Treat current_canonical_suite as the suite being enhanced. A proposed replacement must preserve "
            "all previously killed synthetic patches as well as killing the current synthetic patch."
        )
    else:
        raise ValueError(f"unknown KITE arm: {arm}")

    payload: dict[str, object] = {
        "arm": arm,
        "requested_candidates": k,
        "reference_target_unit": _unit_payload(unit),
        "historical_mechanism": _mechanism_payload(mechanism),
        "synthetic_patch": validation.patch if validation and validation.accepted else None,
        "injection_expected_behavior": plan.expected_behavioral_change if plan else None,
        "trigger_recipe": plan.trigger_recipe if plan else None,
        "prior_reference_and_synthetic_feedback": feedback or [],
        "current_canonical_suite": [
            value.model_dump(mode="json") for value in (current_suite or [])
        ],
        "previously_killed_synthetic_patches": previously_killed_patches or [],
    }

    class ExpectedBatch(StrictModel):
        candidates: list[KiteTestDraft] = Field(min_length=k, max_length=k)

    expected = service.parse(
        schema=ExpectedBatch,
        purpose=f"kite-test-{arm}",
        version=(REPAIR_PROMPT_VERSION if round_number else TEST_PROMPT_VERSION)
        + f"-{arm}-k{k}-r{round_number}",
        system=(
            "You generate raw executable pytest files for an existing Python repository. The supplied "
            "REFERENCE code is the only authoritative target code. Never assume knowledge of a hidden real "
            "bug or hidden developer test. "
            + arm_instruction
            + f" Return exactly {k} candidates. test_code must be raw Python only, no Markdown. Each test "
            "must be self-contained enough to run in the repository and should reuse visible fixtures/import "
            "styles when available. Only import or call APIs demonstrated by the focal source, repository "
            "context, or related pre-existing tests; do not assume modern package re-exports exist. Prefer "
            "the exact import paths and setup patterns shown. Do not edit production code."
        ),
        payload=payload,
    )
    return list(expected.candidates)


def materialize_test(workspace: Path, arm: str, index: int, draft: KiteTestDraft) -> str:
    safe_arm = arm.replace("-", "_")
    relative = f"proofplane_generated/kite/{safe_arm}/test_candidate_{index}.py"
    target = workspace / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(draft.test_code, encoding="utf-8")
    return relative


def synthetic_feedback(
    *,
    drafts: list[KiteTestDraft],
    reference: Path,
    synthetic: Path,
    image: str,
    timeout: int,
    arm: str,
    round_number: int,
    index_offset: int = 0,
) -> list[SyntheticTestAttempt]:
    attempts: list[SyntheticTestAttempt] = []
    for local_index, draft in enumerate(drafts):
        index = index_offset + local_index
        reference_path = materialize_test(reference, arm, index, draft)
        synthetic_path = materialize_test(synthetic, arm, index, draft)
        if reference_path != synthetic_path:
            raise RuntimeError("reference/synthetic test paths diverged")
        reference_out = run_test(reference, reference_path, timeout, image=image)
        synthetic_out = run_test(synthetic, synthetic_path, timeout, image=image)
        attempts.append(
            SyntheticTestAttempt(
                candidate=draft,
                round=round_number,
                reference=reference_out,
                synthetic=synthetic_out,
                frozen=reference_out.status == "pass" and synthetic_out.status == "fail",
            )
        )
    return attempts


def feedback_payload(attempts: list[SyntheticTestAttempt]) -> list[dict[str, object]]:
    return [
        {
            "rationale": attempt.candidate.rationale,
            "reference": {
                "status": attempt.reference.status,
                "stdout": attempt.reference.stdout[-1600:],
                "stderr": attempt.reference.stderr[-1600:],
            },
            "synthetic": {
                "status": attempt.synthetic.status,
                "stdout": attempt.synthetic.stdout[-1600:],
                "stderr": attempt.synthetic.stderr[-1600:],
            },
            "instruction": (
                "Keep the useful hypothesis but correct setup/assertions so reference passes and synthetic fails."
            ),
        }
        for attempt in attempts
        if not attempt.frozen
    ]
