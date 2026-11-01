from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class Settings(BaseModel):
    data_dir: Path = Path("data")
    database: Path = Path("data/proofplane.sqlite3")
    artifacts_dir: Path = Path("data/artifacts")
    split_seed: int = 0
    case_limit: int = 5
    generation_k: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    retrieval_top_k: int = 3
    candidate_pool_size: int = 10
    mmr_lambda: float = 0.7
    execution_timeout: int = 30
    max_context_chars: int = 30000
    max_nearby_tests: int = 8
    max_test_snippet_chars: int = 2500
    max_repository_tree_entries: int = 100
    generation_max_output_tokens: int = 10000
    api_budget_usd: float = 5.0
    api_call_reserve_usd: float = 0.25
    api_retry_attempts: int = 5
    api_retry_base_seconds: float = 2.0
    kite_api_budget_usd: float = 15.0
    kite_temperature: float | None = None
    reasoning_model: str = "gpt-5"
    fast_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"
    kite_max_target_units: int = 180
    kite_max_focal_methods: int = 1
    kite_max_unit_chars: int = 12000
    kite_llm_target_profile: bool = True
    kite_retrieval_top_k: int = 3
    kite_semantic_weight: float = 0.7
    kite_structural_weight: float = 0.3
    kite_max_injection_attempts: int = 3
    kite_max_accepted_injections: int = 8
    kite_injection_alignment_threshold: float = 0.42
    kite_max_test_repairs: int = 4
    kite_max_enhancement_rounds: int = 5
    kite_embedding_batch_size: int = 96
    kite_knowledge_group_similarity_threshold: float = 0.72
    kite_knowledge_max_group_size: int = 8
    kite_knowledge_review_min_confidence: float = 0.65
    hybrid_weights: dict[str, float] = Field(
        default_factory=lambda: {"descriptor": 0.4, "code": 0.25, "sequence": 0.2, "metadata": 0.15}
    )
    metadata_field_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "component_roles": 1.0,
            "operations_changed": 1.0,
            "stateful_resources": 1.0,
        }
    )

    @classmethod
    def load(cls, path: Path | None = None) -> "Settings":
        raw: dict[str, Any] = {}
        if path and path.exists():
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        models = raw.pop("models", {})
        raw.update(
            {
                "reasoning_model": os.getenv(
                    "OPENAI_REASONING_MODEL", models.get("reasoning", "gpt-5")
                ),
                "fast_model": os.getenv("OPENAI_FAST_MODEL", models.get("fast", "gpt-4.1-mini")),
                "embedding_model": os.getenv(
                    "OPENAI_EMBEDDING_MODEL", models.get("embedding", "text-embedding-3-small")
                ),
            }
        )
        return cls.model_validate(raw)
