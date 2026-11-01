from __future__ import annotations

from abc import ABC, abstractmethod
import json
from pathlib import Path
import re
import time
import ast

from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError, field_validator
from proofplane_v0.corpus.store import CorpusStore, cache_key
from proofplane_v0.datasets.base import DatasetCase
from proofplane_v0.generation.arms import HistoricalEvidence, arm_spec
from proofplane_v0.models import GeneratedTestCandidate


class GenerationProvider(ABC):
    @abstractmethod
    def generate(
        self, case: DatasetCase, arm: str, evidence: list[HistoricalEvidence], k: int
    ) -> list[GeneratedTestCandidate]: ...


class MockProvider(GenerationProvider):
    """Deterministic offline provider; emits the benchmark oracle test only for smoke plumbing."""

    def generate(
        self, case: DatasetCase, arm: str, evidence: list[HistoricalEvidence], k: int
    ) -> list[GeneratedTestCandidate]:
        name, code = next(iter(case.hidden_tests.items()))
        return [
            GeneratedTestCandidate(
                candidate_id=f"{case.project}-{case.bug_id}-{index}",
                rationale="deterministic smoke candidate",
                targeted_mechanisms=[item.summary for item in evidence],
                relative_path=f"generated_{index}_{name}",
                test_code=code,
                expected_behavior="passes on reference and fails on candidate",
                generation_arm=arm,
            )
            for index in range(k)
        ]


class OpenAIProvider(GenerationProvider):
    def __init__(
        self,
        model: str,
        budget_usd: float,
        allow_budget_overrun: bool = False,
        store: CorpusStore | None = None,
        reserve_usd: float = 0.25,
        max_output_tokens: int = 10000,
        client: OpenAI | None = None,
        api_retry_attempts: int = 5,
        api_retry_base_seconds: float = 2.0,
    ):
        self.model, self.budget_usd, self.allow_budget_overrun = (
            model,
            budget_usd,
            allow_budget_overrun,
        )
        self.store = store
        self.reserve_usd = reserve_usd
        self.max_output_tokens = max_output_tokens
        self.client = client or OpenAI()
        self.api_retry_attempts = api_retry_attempts
        self.api_retry_base_seconds = api_retry_base_seconds

    @staticmethod
    def _transient(error: Exception) -> bool:
        return type(error).__name__ in {
            "APIConnectionError",
            "APITimeoutError",
            "RateLimitError",
            "InternalServerError",
        } or int(getattr(error, "status_code", 0) or 0) >= 500

    def generate(
        self, case: DatasetCase, arm: str, evidence: list[HistoricalEvidence], k: int
    ) -> list[GeneratedTestCandidate]:
        if self.budget_usd <= 0 and not self.allow_budget_overrun:
            raise RuntimeError(
                "API spend guard prevents this request; raise the configured budget explicitly"
            )

        class CandidateDraft(BaseModel):
            rationale: str = Field(max_length=1200)
            targeted_mechanisms: list[str] = Field(default_factory=list, max_length=8)
            relative_path: str = Field(max_length=240)
            test_code: str = Field(
                max_length=8000,
                description=(
                    "Raw executable Python pytest source only; no Markdown fences, separators, "
                    "explanatory prose, or nested JSON."
                ),
            )
            expected_behavior: str = Field(max_length=800)

            @field_validator("test_code")
            @classmethod
            def raw_executable_python(cls, value: str) -> str:
                if "```" in value or re.search(r"(?m)^\s*---+\s*$", value):
                    raise ValueError("test_code contains Markdown formatting")
                ast.parse(value)
                return value

        class CandidateBatch(BaseModel):
            candidates: list[CandidateDraft]

        prompt = build_generation_prompt(case, arm, evidence, k)
        key = cache_key(self.model, "generate-v2", CandidateBatch.__name__, prompt)
        if self.store:
            cached = self.store.cache_get(key)
            if cached:
                cached_batch = CandidateBatch.model_validate_json(cached)
                return [
                    GeneratedTestCandidate(
                        candidate_id=f"{case.project}-{case.bug_id}-{index}",
                        generation_arm=arm,
                        **draft.model_dump(),
                    )
                    for index, draft in enumerate(cached_batch.candidates[:k])
                ]
        last_error: Exception | None = None
        batch: CandidateBatch | None = None
        for attempt in range(self.api_retry_attempts):
            attempt_prompt = prompt
            if attempt:
                attempt_prompt += (
                    "\nREPAIR: The prior response was truncated or invalid. Return complete JSON and "
                    "make it substantially shorter. Keep each rationale under 1200 characters, "
                    "expected behavior under 800 characters, and test_code under 8000 characters. "
                    "Return exactly the requested number of candidates with no prose outside the schema."
                )
            usage_id = (
                self.store.reserve_api_call(
                    self.model,
                    "generate-v2",
                    self.budget_usd,
                    self.reserve_usd,
                    self.allow_budget_overrun,
                )
                if self.store
                else None
            )
            try:
                response = self.client.responses.parse(
                    model=self.model,
                    input=attempt_prompt,
                    text_format=CandidateBatch,
                    max_output_tokens=self.max_output_tokens,
                )
                batch = response.output_parsed
                if batch is None:
                    raise ValueError("model returned no structured candidates")
                if len(batch.candidates) != k:
                    raise ValueError(
                        f"model returned {len(batch.candidates)} candidates; expected exactly {k}"
                    )
                if self.store and usage_id is not None and response.usage:
                    self.store.update_api_tokens(
                        usage_id,
                        int(response.usage.input_tokens),
                        int(response.usage.output_tokens),
                    )
                break
            except Exception as error:
                last_error = error
                batch = None
                if self._transient(error) and attempt + 1 < self.api_retry_attempts:
                    time.sleep(self.api_retry_base_seconds * (2**attempt))
                elif not isinstance(error, (ValidationError, ValueError)):
                    raise
        if batch is None:
            assert last_error is not None
            if self.store:
                self.store.cache_put(key, "{}", f"{type(last_error).__name__}: {last_error}")
            raise last_error
        if self.store:
            self.store.cache_put(key, batch.model_dump_json())
        return [
            GeneratedTestCandidate(
                candidate_id=f"{case.project}-{case.bug_id}-{index}",
                generation_arm=arm,
                **draft.model_dump(),
            )
            for index, draft in enumerate(batch.candidates[:k])
        ]


def build_generation_prompt(
    case: DatasetCase, arm: str, evidence: list[HistoricalEvidence], k: int
) -> str:
    spec = arm_spec(arm)
    if spec.evidence_kind == "none" and evidence:
        raise ValueError(f"{arm} must not receive historical evidence")
    if any(item.evidence_kind != spec.evidence_kind for item in evidence):
        raise ValueError(f"{arm} received the wrong historical evidence type")

    template = (
        Path(__file__).resolve().parents[1] / "prompts" / spec.prompt_template
    ).read_text(encoding="utf-8").strip()
    visible = {
        "project": case.project,
        "candidate_source": case.buggy_source,
        "pre_existing_tests": case.pre_fix_tests,
        "runtime_context": case.metadata.get("runtime_context", {}),
    }
    historical = [item.model_dump(mode="json") for item in evidence]
    if arm == "oracle_target_causal_program":
        evidence_label = "PRIVILEGED TARGET CAUSAL PROGRAM"
        evidence_contract = (
            "The supplied causal program is privileged, abstract, and fallible evidence for this "
            "diagnostic only; it was deterministically derived from existing enrichment."
        )
        context_contract = (
            "Use the complete visible target context as primary grounding and the explicitly supplied "
            "causal program as an additional hypothesis scaffold. Do not invent any other hidden target knowledge."
        )
    elif spec.target_oracle:
        evidence_label = "PRIVILEGED TARGET SUMMARY"
        evidence_contract = (
            "The supplied target summary is privileged, abstract, and fallible evidence for this diagnostic only."
        )
        context_contract = (
            "Use the visible target context plus only the explicitly supplied privileged abstract summary; "
            "do not invent any other hidden target knowledge."
        )
    else:
        evidence_label = "HISTORICAL EVIDENCE"
        evidence_contract = "Historical examples are transferable inspiration only, never facts about the target."
        context_contract = (
            "Use only visible target context; never invent knowledge of the target's hidden bug or hidden tests."
        )
    return (
        "You are designing regression tests for a changed implementation.\n"
        "AUTHORITATIVE DIRECTION: the base/reference version defines intended behavior; "
        "the candidate version is suspected of introducing a regression. In a unified diff, "
        "'-' lines are base/reference and '+' lines are candidate. Do not assume candidate "
        "additions or removals are desired.\n"
        "SUCCESS CRITERION: each test should pass on base/reference and fail on candidate if "
        f"its hypothesis is correct. {context_contract}\n"
        f"GENERATION ARM: {arm}\n"
        f"ARM INSTRUCTIONS: {template}\n"
        f"Generate exactly {k} executable pytest test candidates. {evidence_contract}\n"
        "For every candidate, test_code must contain only raw executable Python source. Do not "
        "place Markdown fences, separators, prose, or a second JSON object inside test_code. Put "
        "the analogy and target-specific hypothesis in rationale instead.\n"
        f"{evidence_label}: {json.dumps(historical, sort_keys=True)}\n"
        f"VISIBLE TARGET CONTEXT: {json.dumps(visible, sort_keys=True)}"
    )


def sanitized_evidence(records: list, arm: str) -> list[HistoricalEvidence]:
    spec = arm_spec(arm)
    if spec.evidence_kind == "none":
        return []
    if spec.evidence_kind == "visible_context":
        return [
            HistoricalEvidence(
                evidence_kind="visible_context",
                summary=item.record.visible_descriptor.visible_change_summary,
            )
            for item in records
        ]
    return [
        HistoricalEvidence(
            evidence_kind="portable_mechanism",
            summary=item.record.hidden_payload.portable_prompt_summary,
            trigger_sequence=item.record.hidden_payload.trigger_sequence,
            test_strategy=item.record.hidden_payload.test_strategy,
        )
        for item in records
    ]
