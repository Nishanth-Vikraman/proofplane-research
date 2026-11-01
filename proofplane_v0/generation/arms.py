from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


EvidenceKind = Literal[
    "none", "visible_context", "portable_mechanism", "target_causal_program"
]


@dataclass(frozen=True)
class GenerationArmSpec:
    name: str
    retrieval_mode: str | None
    evidence_kind: EvidenceKind
    prompt_template: str
    oracle: bool = False
    target_oracle: bool = False


ARM_SPECS: dict[str, GenerationArmSpec] = {
    "direct": GenerationArmSpec(
        "direct", None, "none", "generate_direct.md"
    ),
    "brainstorm": GenerationArmSpec(
        "brainstorm", None, "none", "generate_brainstorm.md"
    ),
    "random_mechanisms": GenerationArmSpec(
        "random_mechanisms", "random", "portable_mechanism", "generate_random_mechanisms.md"
    ),
    "retrieved_raw_context": GenerationArmSpec(
        "retrieved_raw_context", "hybrid", "visible_context", "generate_retrieved_raw_context.md"
    ),
    "retrieved_mechanisms": GenerationArmSpec(
        "retrieved_mechanisms", "hybrid", "portable_mechanism", "generate_retrieved_mechanisms.md"
    ),
    "retrieved_mechanisms_mmr": GenerationArmSpec(
        "retrieved_mechanisms_mmr",
        "hybrid_mmr",
        "portable_mechanism",
        "generate_retrieved_mechanisms_mmr.md",
    ),
    "oracle": GenerationArmSpec(
        "oracle", "oracle_hidden_family", "portable_mechanism", "generate_oracle.md", oracle=True
    ),
    "oracle_target_summary": GenerationArmSpec(
        "oracle_target_summary",
        "oracle_target_summary",
        "portable_mechanism",
        "generate_oracle_target_summary.md",
        oracle=True,
        target_oracle=True,
    ),
    "oracle_target_causal_program": GenerationArmSpec(
        "oracle_target_causal_program",
        "oracle_target_causal_program",
        "target_causal_program",
        "generate_oracle_target_causal_program.md",
        oracle=True,
        target_oracle=True,
    ),
}


class CausalProgram(BaseModel):
    """Portable causal structure derived without another enrichment call."""

    model_config = ConfigDict(extra="forbid", strict=True)

    derivation: Literal["deterministic-existing-enrichment-v1"] = (
        "deterministic-existing-enrichment-v1"
    )
    preconditions: list[str] = Field(default_factory=list)
    action_sequence: list[str] = Field(default_factory=list)
    state_transition: str
    lost_property: str
    observable: str
    test_shape: list[str] = Field(default_factory=list)


class HistoricalEvidence(BaseModel):
    """The only mechanism information that may cross into a generation prompt."""

    model_config = ConfigDict(extra="forbid", strict=True)

    evidence_kind: Literal[
        "visible_context", "portable_mechanism", "target_causal_program"
    ]
    summary: str
    trigger_sequence: list[str] = Field(default_factory=list)
    test_strategy: list[str] = Field(default_factory=list)
    causal_program: CausalProgram | None = None


def arm_spec(arm: str) -> GenerationArmSpec:
    try:
        return ARM_SPECS[arm]
    except KeyError as exc:
        supported = ", ".join(ARM_SPECS)
        raise ValueError(f"unsupported generation arm {arm!r}; choose one of: {supported}") from exc


def arm_names() -> list[str]:
    return list(ARM_SPECS)
