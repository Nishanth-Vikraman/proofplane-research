import pytest
from pydantic import ValidationError
from proofplane_v0.models import CausalRetrievalDescriptor, HiddenBugPayload


def descriptor(**extra):
    return CausalRetrievalDescriptor(
        visible_change_summary="change", retrieval_text="python change", confidence=0.8, **extra
    )


def test_visible_rejects_hidden_field():
    with pytest.raises(ValidationError):
        descriptor(root_cause="secret")


def test_hidden_portable_summary_rejects_paths():
    with pytest.raises(ValidationError):
        HiddenBugPayload(
            mechanism_summary="x",
            portable_prompt_summary="foo.py fails",
            trigger_sequence=[],
            violated_invariant="x",
            observable_symptom="x",
            test_strategy=[],
            mechanism_primitives=[],
            bug_family="x",
            root_cause="x",
            fix_strategy="x",
            confidence=0.8,
        )
