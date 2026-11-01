from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

HIDDEN_NAMES = {
    "bug_family",
    "root_cause",
    "violated_invariant",
    "observed_failure",
    "fix_shape",
    "actual_trigger",
    "developer_regression_test",
    "hidden_issue_explanation",
    "hidden_payload",
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class EvidenceReference(StrictModel):
    source: str
    locator: str
    excerpt_hash: str = ""


class CausalRetrievalDescriptor(StrictModel):
    language: str = "python"
    frameworks: list[str] = Field(default_factory=list)
    component_roles: list[str] = Field(default_factory=list)
    changed_symbol_roles: list[str] = Field(default_factory=list)
    operations_present: list[str] = Field(default_factory=list)
    operations_changed: list[str] = Field(default_factory=list)
    stateful_resources: list[str] = Field(default_factory=list)
    interaction_topology: list[str] = Field(default_factory=list)
    temporal_capabilities: list[str] = Field(default_factory=list)
    control_flow_features: list[str] = Field(default_factory=list)
    data_properties: list[str] = Field(default_factory=list)
    external_side_effects: list[str] = Field(default_factory=list)
    concurrency_signals: list[str] = Field(default_factory=list)
    lifecycle_stages: list[str] = Field(default_factory=list)
    boundary_surfaces: list[str] = Field(default_factory=list)
    existing_test_shapes: list[str] = Field(default_factory=list)
    apparent_test_gaps: list[str] = Field(default_factory=list)
    event_sequence: list[str] = Field(default_factory=list)
    visible_change_summary: str
    retrieval_text: str
    evidence: list[EvidenceReference] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="before")
    @classmethod
    def reject_hidden_keys(cls, value: Any) -> Any:
        if isinstance(value, dict) and HIDDEN_NAMES.intersection(value):
            raise ValueError("hidden/oracle fields are forbidden in visible descriptors")
        return value


class HiddenBugPayload(StrictModel):
    mechanism_summary: str
    portable_prompt_summary: str
    trigger_sequence: list[str]
    violated_invariant: str
    observable_symptom: str
    test_strategy: list[str]
    mechanism_primitives: list[str]
    bug_family: str
    root_cause: str
    fix_strategy: str
    evidence: list[EvidenceReference] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)

    @field_validator("portable_prompt_summary")
    @classmethod
    def portable(cls, value: str) -> str:
        if "/" in value or "\\" in value or ".py" in value:
            raise ValueError("portable summary must not include paths or Python filenames")
        return value


class Provenance(StrictModel):
    buggy_commit: str = ""
    fixed_commit: str = ""
    test_files: list[str] = Field(default_factory=list)
    source: str = ""


class CorpusBugRecord(StrictModel):
    dataset: str
    project: str
    bug_id: str
    split: str
    visible_descriptor: CausalRetrievalDescriptor
    hidden_payload: HiddenBugPayload
    provenance: Provenance = Field(default_factory=Provenance)
    source_hashes: dict[str, str] = Field(default_factory=dict)
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def corpus_id(self) -> str:
        return f"{self.dataset}:{self.project}:{self.bug_id}"


class TokenUsage(StrictModel):
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost_usd: float = 0.0


class GeneratedTestCandidate(StrictModel):
    candidate_id: str
    rationale: str
    targeted_mechanisms: list[str]
    relative_path: str
    test_code: str
    expected_behavior: str
    generation_arm: str
    token_usage: TokenUsage | None = None


class ExecutionOutcome(StrictModel):
    status: Literal["pass", "fail", "syntax", "import_setup", "timeout", "error"]
    return_code: int | None = None
    duration_seconds: float = 0.0
    stdout: str = ""
    stderr: str = ""


class EvaluationResult(StrictModel):
    case_id: str
    split: str
    evaluation_bucket: str
    generation_arm: str
    retrieval_mode: str
    retrieval_parameters: dict[str, Any] = Field(default_factory=dict)
    retrieved_corpus_ids: list[str] = Field(default_factory=list)
    retrieval_scores: list[dict[str, Any]] = Field(default_factory=list)
    generated_test_count: int = 0
    syntax_valid_count: int = 0
    runnable_count: int = 0
    tests_failing_on_candidate: int = 0
    tests_passing_on_reference: int = 0
    successful_regression_revealing_tests: int = 0
    success_at_k: dict[str, bool] = Field(default_factory=dict)
    execution_times: dict[str, float] = Field(default_factory=dict)
    model_usage: TokenUsage | None = None
    errors: list[str] = Field(default_factory=list)
    exclusions: list[str] = Field(default_factory=list)
