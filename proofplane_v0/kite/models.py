from __future__ import annotations

from typing import Literal

from pydantic import Field, ValidationInfo, field_validator

from proofplane_v0.models import ExecutionOutcome, StrictModel, TokenUsage


class StructuralUnit(StrictModel):
    file: str
    qualname: str
    anchor_role: str
    node_type: str
    coarse_key: str
    fingerprint: str
    normalized_tree: str
    signatures: list[str] = Field(default_factory=list)
    relations: list[str] = Field(default_factory=list)
    anchor_count: int = 1
    relation_count: int = 0
    buggy_fingerprint: str = ""
    buggy_normalized_tree: str = ""
    edit_operations: list[str] = Field(default_factory=list)
    start_line: int
    end_line: int
    source_hash: str


class TransferableMechanism(StrictModel):
    mechanism_id: str
    corpus_id: str
    source_project: str
    what: str
    where: str
    how: str
    preconditions: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    transition: str
    violated_property: str
    observable: str
    test_shape: list[str] = Field(default_factory=list)
    structural_units: list[StructuralUnit] = Field(default_factory=list)
    derivation: Literal[
        "existing_enrichment", "train_llm_explicit", "train_llm_grouped"
    ] = "existing_enrichment"
    enrichment_version: str = "kite-mechanism-derived-v1"
    review_status: Literal["unreviewed", "accepted", "rejected"] = "unreviewed"
    review_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    semantic_group: str = ""
    rejection_reason: str = ""
    expected_buggy_behavior: str = ""
    representative_evidence: list[str] = Field(default_factory=list, max_length=12)
    supporting_bug_unit_ids: list[str] = Field(default_factory=list)
    source_projects: list[str] = Field(default_factory=list)

    @property
    def semantic_text(self) -> str:
        return "\n".join(
            (
                f"WHAT: {self.what}",
                f"WHERE: {self.where}",
                f"HOW: {self.how}",
                f"SEMANTIC GROUP: {self.semantic_group or 'unreviewed'}",
                f"EXPECTED BUGGY BEHAVIOR: {self.expected_buggy_behavior or self.observable}",
                f"REPRESENTATIVE EVIDENCE: {'; '.join(self.representative_evidence)}",
                f"PRECONDITIONS: {'; '.join(self.preconditions)}",
                f"ACTIONS: {'; '.join(self.actions)}",
                f"TRANSITION: {self.transition}",
                f"VIOLATED PROPERTY: {self.violated_property}",
                f"OBSERVABLE: {self.observable}",
                f"TEST SHAPE: {'; '.join(self.test_shape)}",
            )
        )


class ExplicitMechanismFields(StrictModel):
    """Optional future train-only enrichment; never computed from an eval target."""

    what: str = Field(max_length=1600)
    where: str = Field(max_length=1600)
    how: str = Field(max_length=1600)
    applicability_signals: list[str] = Field(default_factory=list, max_length=12)
    injection_constraints: list[str] = Field(default_factory=list, max_length=12)
    transferable: bool
    review_confidence: float = Field(ge=0.0, le=1.0)
    semantic_group: str = Field(max_length=240)
    rejection_reason: str = Field(default="", max_length=800)


class BugUnitSemanticAbstraction(StrictModel):
    """Paper-aligned semantic representation for one historical bug unit."""

    what: str = Field(max_length=1600)
    where: str = Field(max_length=1600)
    how: str = Field(max_length=1600)
    expected_buggy_behavior: str = Field(max_length=1600)
    applicability_signals: list[str] = Field(default_factory=list, max_length=12)
    injection_constraints: list[str] = Field(default_factory=list, max_length=12)
    representative_evidence: list[str] = Field(default_factory=list, max_length=8)
    abstraction_confidence: float = Field(ge=0.0, le=1.0)


class ReviewedMechanismGroup(StrictModel):
    """Reviewed synthesis of one semantically close historical bug-unit group."""

    what: str = Field(max_length=1600)
    where: str = Field(max_length=1600)
    how: str = Field(max_length=1600)
    semantic_group: str = Field(max_length=240)
    expected_buggy_behavior: str = Field(max_length=1600)
    preconditions: list[str] = Field(default_factory=list, max_length=12)
    actions: list[str] = Field(default_factory=list, max_length=12)
    transition: str = Field(max_length=1600)
    violated_property: str = Field(max_length=1600)
    observable: str = Field(max_length=1600)
    test_shape: list[str] = Field(default_factory=list, max_length=12)
    representative_evidence: list[str] = Field(default_factory=list, max_length=12)
    transferable: bool
    review_confidence: float = Field(ge=0.0, le=1.0)
    rejection_reason: str = Field(default="", max_length=1000)


class FocalMethodSpec(StrictModel):
    """Sanitized benchmark metadata exported across the hidden-patch boundary."""

    file: str
    qualname: str
    start_line: int
    end_line: int
    reference_source_hash: str


class FocalBenchmarkCase(StrictModel):
    case_id: str
    project: str
    eligible: bool
    focal_methods: list[FocalMethodSpec] = Field(default_factory=list)
    exclusion_reason: str = ""


class FocalBenchmarkManifest(StrictModel):
    split: str
    max_focal_methods: int = Field(ge=1)
    cases: list[FocalBenchmarkCase]
    content_hash: str
    schema_version: str = "kite-focal-benchmark-v1"


class RepositoryContext(StrictModel):
    same_class_helpers: dict[str, str] = Field(default_factory=dict)
    called_dependencies: dict[str, str] = Field(default_factory=dict)
    related_tests: dict[str, str] = Field(default_factory=dict)
    documentation: list[str] = Field(default_factory=list)
    type_constraints: list[str] = Field(default_factory=list)


class TargetSemanticFields(StrictModel):
    mechanism_bearing_behavior: str = Field(max_length=1600)
    applicability_context: str = Field(max_length=1600)
    edit_affordance: str = Field(max_length=1600)


class TargetCodeUnit(StrictModel):
    unit_id: str
    file: str
    qualname: str
    signature: str
    docstring: str
    source: str
    semantic_text: str
    semantic_what: str = ""
    semantic_where: str = ""
    semantic_how: str = ""
    structural: StructuralUnit
    structural_queries: list[StructuralUnit] = Field(default_factory=list)
    related_tests: dict[str, str] = Field(default_factory=dict)
    repository_context: RepositoryContext = Field(default_factory=RepositoryContext)
    deterministic_score: float = 0.0
    content_hash: str


class ReferenceTargetProfile(StrictModel):
    case_id: str
    project: str
    reference_commit: str
    repository_files: list[str] = Field(default_factory=list)
    units: list[TargetCodeUnit]
    content_hash: str
    schema_version: str = "kite-reference-profile-v3-focal-context-anchors"


class RetrievalMatch(StrictModel):
    rank: int
    mechanism: TransferableMechanism
    target_unit: TargetCodeUnit
    semantic_score: float
    structural_score: float
    fused_score: float
    semantic_rank: int | None = None
    structural_rank: int | None = None
    support_paths: list[Literal["semantic", "structural"]] = Field(default_factory=list)


class TargetAnchor(StrictModel):
    file: str
    qualname: str
    start_line: int
    end_line: int
    source_hash: str
    ast_fingerprint: str


class MechanismAlignment(StrictModel):
    dimension: Literal["what", "where", "how", "precondition", "observable"]
    historical_element: str = Field(max_length=1000)
    target_evidence: str = Field(max_length=1200)
    edit_effect: str = Field(max_length=1200)


class InjectionPlan(StrictModel):
    mechanism_to_reproduce: str = Field(max_length=1600)
    target_anchor_id: str = Field(max_length=32)
    applicability_reason: str = Field(max_length=1600)
    before: str = Field(max_length=7000)
    after: str = Field(max_length=7000)
    expected_behavioral_change: str = Field(max_length=1600)
    trigger_recipe: list[str] = Field(default_factory=list, max_length=12)
    mechanism_alignment: list[MechanismAlignment] = Field(min_length=3, max_length=8)
    behavioral_probe: str = Field(max_length=9000)

    @field_validator("before", "after", "behavioral_probe")
    @classmethod
    def raw_python_fragment(cls, value: str, info: ValidationInfo) -> str:
        import ast

        if "```" in value:
            raise ValueError("injection fragments must be raw Python without Markdown fences")
        if value and info.field_name == "behavioral_probe":
            ast.parse(value)
        return value


class InjectionValidation(StrictModel):
    accepted: bool
    gates: dict[str, bool]
    reasons: list[str] = Field(default_factory=list)
    patch: str = ""
    synthetic_workspace: str = ""
    execution: ExecutionOutcome | None = None
    reference_compilation: ExecutionOutcome | None = None
    synthetic_compilation: ExecutionOutcome | None = None
    reference_repository_probe: ExecutionOutcome | None = None
    synthetic_repository_probe: ExecutionOutcome | None = None
    behavioral_reference: ExecutionOutcome | None = None
    behavioral_synthetic: ExecutionOutcome | None = None


class KiteTestDraft(StrictModel):
    rationale: str = Field(max_length=1400)
    targeted_mechanisms: list[str] = Field(default_factory=list, max_length=8)
    relative_path: str = Field(max_length=240)
    test_code: str = Field(max_length=9000)
    expected_behavior: str = Field(max_length=1000)

    @field_validator("test_code")
    @classmethod
    def executable_python(cls, value: str) -> str:
        import ast
        import re

        if "```" in value or re.search(r"(?m)^\s*---+\s*$", value):
            raise ValueError("test_code contains Markdown formatting")
        ast.parse(value)
        return value


class KiteTestBatch(StrictModel):
    candidates: list[KiteTestDraft]


class SyntheticTestAttempt(StrictModel):
    candidate: KiteTestDraft
    round: int = 0
    reference: ExecutionOutcome
    synthetic: ExecutionOutcome
    frozen: bool


class KiteArmResult(StrictModel):
    case_id: str
    arm: str
    retrieved_mechanism_ids: list[str] = Field(default_factory=list)
    target_unit_ids: list[str] = Field(default_factory=list)
    injection_accepted: bool | None = None
    injection_attempts: int = 0
    generated_count: int = 0
    synthetic_kills: int = 0
    unique_injections: int = 0
    enhancement_rounds: int = 0
    promoted_tests: int = 0
    reference_passes: int = 0
    real_bug_kills: int = 0
    success_at_k: dict[str, bool] = Field(default_factory=dict)
    attempts: list[SyntheticTestAttempt] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    token_usage: TokenUsage | None = None


class KiteCaseResult(StrictModel):
    case_id: str
    project: str
    arms: list[KiteArmResult]
    status: Literal["complete", "partial", "error"]
    errors: list[str] = Field(default_factory=list)
