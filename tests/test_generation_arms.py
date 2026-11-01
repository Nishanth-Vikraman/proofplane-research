from __future__ import annotations

from types import SimpleNamespace

import pytest

from proofplane_v0.datasets.base import DatasetCase
from proofplane_v0.generation.arms import ARM_SPECS, HistoricalEvidence
from proofplane_v0.generation.provider import (
    OpenAIProvider,
    build_generation_prompt,
    sanitized_evidence,
)
from proofplane_v0.models import CausalRetrievalDescriptor, CorpusBugRecord, HiddenBugPayload
from proofplane_v0.experiments.runner import (
    oracle_family_matches,
    target_causal_program_evidence,
    target_summary_evidence,
)
from proofplane_v0.retrieval.hybrid import Retrieved


def target_case() -> DatasetCase:
    return DatasetCase(
        dataset="synthetic",
        project="target",
        bug_id="1",
        buggy_source={"module.py": "VALUE = 2\n"},
        fixed_source={"module.py": "VALUE = 1\n"},
        pre_fix_tests={"test_module.py": "def test_existing(): pass\n"},
        hidden_tests={},
        metadata={
            "runtime_context": {
                "visible_diff": "--- base/module.py\n+++ candidate/module.py\n-VALUE = 1\n+VALUE = 2"
            }
        },
    )


def historical_record() -> Retrieved:
    record = CorpusBugRecord(
        dataset="synthetic",
        project="history",
        bug_id="2",
        split="train",
        visible_descriptor=CausalRetrievalDescriptor(
            visible_change_summary="VISIBLE ANALOGY: a boundary option changes",
            retrieval_text="boundary option change",
            confidence=1.0,
        ),
        hidden_payload=HiddenBugPayload(
            mechanism_summary="private mechanism details",
            portable_prompt_summary="PORTABLE MECHANISM: presence is confused with truth value",
            trigger_sequence=["TRIGGER STEP: provide an explicit false value", "observe branch"],
            violated_invariant="explicit false remains false",
            observable_symptom="the wrong branch runs",
            test_strategy=["TEST STRATEGY: contrast absent, false, and true"],
            mechanism_primitives=["presence-vs-value"],
            bug_family="predicate-semantics",
            root_cause="presence check",
            fix_strategy="test value",
            confidence=1.0,
        ),
    )
    return Retrieved(record, {"total": 1.0})


def test_all_design_spec_arms_have_explicit_contracts():
    assert list(ARM_SPECS) == [
        "direct",
        "brainstorm",
        "random_mechanisms",
        "retrieved_raw_context",
        "retrieved_mechanisms",
        "retrieved_mechanisms_mmr",
        "oracle",
        "oracle_target_summary",
        "oracle_target_causal_program",
    ]
    assert ARM_SPECS["direct"].retrieval_mode is None
    assert ARM_SPECS["brainstorm"].retrieval_mode is None
    assert ARM_SPECS["random_mechanisms"].retrieval_mode == "random"
    assert ARM_SPECS["retrieved_raw_context"].evidence_kind == "visible_context"
    assert ARM_SPECS["retrieved_mechanisms"].retrieval_mode == "hybrid"
    assert ARM_SPECS["retrieved_mechanisms_mmr"].retrieval_mode == "hybrid_mmr"
    assert ARM_SPECS["oracle"].oracle
    assert ARM_SPECS["oracle_target_summary"].target_oracle
    assert ARM_SPECS["oracle_target_causal_program"].target_oracle


@pytest.mark.parametrize("arm", ["direct", "brainstorm"])
def test_unconditioned_arms_receive_no_historical_evidence(arm: str):
    evidence = sanitized_evidence([historical_record()], arm)
    prompt = build_generation_prompt(target_case(), arm, evidence, 3)

    assert evidence == []
    assert "PORTABLE MECHANISM" not in prompt
    assert "VISIBLE ANALOGY" not in prompt


def test_direct_and_brainstorm_use_distinct_instructions():
    direct = build_generation_prompt(target_case(), "direct", [], 3)
    brainstorm = build_generation_prompt(target_case(), "brainstorm", [], 3)

    assert direct != brainstorm
    assert "Derive hypotheses directly" in direct
    assert "First infer a diverse set" in brainstorm


def test_raw_context_arm_gets_visible_description_only():
    evidence = sanitized_evidence([historical_record()], "retrieved_raw_context")
    prompt = build_generation_prompt(target_case(), "retrieved_raw_context", evidence, 1)

    assert evidence[0].evidence_kind == "visible_context"
    assert "VISIBLE ANALOGY" in prompt
    assert "PORTABLE MECHANISM" not in prompt
    assert "TRIGGER STEP" not in prompt
    assert "TEST STRATEGY" not in prompt


@pytest.mark.parametrize(
    "arm",
    [
        "random_mechanisms",
        "retrieved_mechanisms",
        "retrieved_mechanisms_mmr",
        "oracle",
        "oracle_target_summary",
    ],
)
def test_mechanism_arms_get_the_complete_portable_payload(arm: str):
    evidence = sanitized_evidence([historical_record()], arm)
    prompt = build_generation_prompt(target_case(), arm, evidence, 1)

    assert evidence[0].evidence_kind == "portable_mechanism"
    assert "PORTABLE MECHANISM" in prompt
    assert "TRIGGER STEP" in prompt
    assert "TEST STRATEGY" in prompt
    assert "private mechanism details" not in prompt
    assert "presence check" not in prompt
    assert "predicate-semantics" not in prompt


def test_prompt_makes_reference_candidate_direction_unambiguous():
    prompt = build_generation_prompt(target_case(), "direct", [], 1)

    assert "base/reference version defines intended behavior" in prompt
    assert "pass on base/reference and fail on candidate" in prompt
    assert "'-' lines are base/reference and '+' lines are candidate" in prompt


def test_arm_rejects_the_wrong_evidence_type():
    wrong = HistoricalEvidence(evidence_kind="visible_context", summary="wrong kind")
    with pytest.raises(ValueError, match="wrong historical evidence type"):
        build_generation_prompt(target_case(), "retrieved_mechanisms", [wrong], 1)


def test_generation_retries_truncated_structured_output_with_shorter_repair_prompt():
    class FakeResponses:
        def __init__(self):
            self.inputs: list[str] = []
            self.max_outputs: list[int] = []

        def parse(self, *, input, text_format, max_output_tokens, **_kwargs):
            self.inputs.append(input)
            self.max_outputs.append(max_output_tokens)
            if len(self.inputs) == 1:
                text_format.model_validate_json('{"candidates":[{"rationale":"truncated')
            parsed = text_format.model_validate(
                {
                    "candidates": [
                        {
                            "rationale": "focused boundary hypothesis",
                            "targeted_mechanisms": [],
                            "relative_path": "test_regression.py",
                            "test_code": "def test_regression():\n    assert True\n",
                            "expected_behavior": "passes on reference",
                        }
                    ]
                }
            )
            return SimpleNamespace(output_parsed=parsed, usage=None)

    responses = FakeResponses()
    fake_client = SimpleNamespace(responses=responses)
    provider = OpenAIProvider("test-model", 1.0, client=fake_client)  # type: ignore[arg-type]

    generated = provider.generate(target_case(), "direct", [], 1)

    assert len(generated) == 1
    assert len(responses.inputs) == 2
    assert "prior response was truncated or invalid" in responses.inputs[1]
    assert responses.max_outputs == [10000, 10000]


def test_oracle_has_no_arbitrary_fallback():
    target = historical_record().record.model_copy(deep=True)
    target.hidden_payload.bug_family = "bf-target"
    other = historical_record().record.model_copy(deep=True)
    other.hidden_payload.bug_family = "bf-other"

    assert oracle_family_matches(target, [other], 3) == []
    other.hidden_payload.bug_family = "bf-target"
    assert oracle_family_matches(target, [other], 3) == [other]


def test_target_summary_oracle_exposes_only_portable_summary():
    record = historical_record().record
    evidence = target_summary_evidence(record)
    prompt = build_generation_prompt(target_case(), "oracle_target_summary", evidence, 1)

    assert len(evidence) == 1
    assert evidence[0].summary == "PORTABLE MECHANISM: presence is confused with truth value"
    assert "PRIVILEGED TARGET SUMMARY" in prompt
    assert "PORTABLE MECHANISM" in prompt
    assert "TRIGGER STEP" not in prompt
    assert "TEST STRATEGY" not in prompt
    assert "private mechanism details" not in prompt
    assert "presence check" not in prompt
    assert "predicate-semantics" not in prompt
    assert "presence-vs-value" not in prompt
    assert "useful but fallible hypothesis" in prompt
    assert "exactly one independently runnable pytest test" in prompt
    assert "Do not rely on test ordering" in prompt
    assert "old library version" in prompt
    assert "Use only visible target context; never invent knowledge" not in prompt
    assert "plus only the explicitly supplied privileged abstract summary" in prompt


def test_target_causal_program_restructures_existing_fields_without_hidden_labels():
    record = historical_record().record
    record.hidden_payload.mechanism_summary = "StateThing loses VALUE_42 at step 422"
    evidence = target_causal_program_evidence(record)
    prompt = build_generation_prompt(
        target_case(), "oracle_target_causal_program", evidence, 1
    )

    assert len(evidence) == 1
    assert evidence[0].evidence_kind == "target_causal_program"
    program = evidence[0].causal_program
    assert program is not None
    assert program.derivation == "deterministic-existing-enrichment-v1"
    assert program.preconditions == ["TRIGGER STEP: provide an explicit false value"]
    assert program.state_transition == "State Thing loses VALUE a configured value at step a configured value"
    assert program.lost_property == "explicit false remains false"
    assert program.observable == "the wrong branch runs"
    assert program.test_shape == ["TEST STRATEGY: contrast absent, false, and true"]
    assert "PRIVILEGED TARGET CAUSAL PROGRAM" in prompt
    assert "visible context remains available" in prompt.casefold()
    assert "primary source for concrete imports" in prompt
    assert "private mechanism details" not in prompt
    assert "presence check" not in prompt
    assert "predicate-semantics" not in prompt
    assert "presence-vs-value" not in prompt
