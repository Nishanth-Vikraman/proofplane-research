from __future__ import annotations

import json
import time
from typing import Any, TypeVar

import numpy as np
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from proofplane_v0.config import Settings
from proofplane_v0.corpus.store import CorpusStore, cache_key
from proofplane_v0.retrieval.embeddings import deserialize, serialize

T = TypeVar("T", bound=BaseModel)


class OpenAIService:
    """Shared cached OpenAI wrapper for KITE structured outputs and real embeddings."""

    def __init__(
        self,
        store: CorpusStore,
        settings: Settings,
        allow_budget_overrun: bool = False,
        client: OpenAI | None = None,
    ) -> None:
        self.store = store
        self.settings = settings
        self.allow_budget_overrun = allow_budget_overrun
        self.client = client or OpenAI()

    @staticmethod
    def _transient(error: Exception) -> bool:
        return type(error).__name__ in {
            "APIConnectionError",
            "APITimeoutError",
            "RateLimitError",
            "InternalServerError",
        } or int(getattr(error, "status_code", 0) or 0) >= 500

    @staticmethod
    def _text_cost(model: str, input_tokens: int, output_tokens: int) -> float:
        if model.startswith("gpt-5.4-nano"):
            return max(
                input_tokens * 0.20 / 1_000_000 + output_tokens * 1.25 / 1_000_000,
                0.000001,
            )
        # Unknown models retain the pre-call reserve rather than undercounting spend.
        return -1.0

    def parse(
        self,
        *,
        schema: type[T],
        purpose: str,
        version: str,
        system: str,
        payload: dict[str, object],
        model: str | None = None,
        max_output_tokens: int | None = None,
    ) -> T:
        chosen_model = model or self.settings.reasoning_model
        normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        key = cache_key(
            chosen_model,
            version,
            schema.__name__,
            f"temperature={self.settings.kite_temperature}\n{system}\n{normalized}",
        )
        cached = self.store.cache_get(key)
        if cached:
            return schema.model_validate_json(cached)
        last_error: Exception | None = None
        for attempt in range(self.settings.api_retry_attempts):
            repair = ""
            if last_error is not None:
                repair = (
                    "\nThe previous response failed or was invalid. Return a shorter corrected object "
                    f"matching the schema exactly. Error: {type(last_error).__name__}: {last_error}"
                )
            usage_id = self.store.reserve_api_call(
                chosen_model,
                purpose,
                self.settings.kite_api_budget_usd,
                self.settings.api_call_reserve_usd,
                self.allow_budget_overrun,
                "kite-%",
            )
            try:
                request: Any = {
                    "model": chosen_model,
                    "input": [
                        {"role": "system", "content": system + repair},
                        {"role": "user", "content": normalized},
                    ],
                    "text_format": schema,
                    "max_output_tokens": max_output_tokens
                    or self.settings.generation_max_output_tokens,
                }
                if self.settings.kite_temperature is not None:
                    request["temperature"] = self.settings.kite_temperature
                response = self.client.responses.parse(**request)
                parsed = response.output_parsed
                if parsed is None:
                    raise ValueError("model returned no structured output")
                if response.usage:
                    input_tokens = int(response.usage.input_tokens)
                    output_tokens = int(response.usage.output_tokens)
                    cost = self._text_cost(chosen_model, input_tokens, output_tokens)
                    if cost >= 0:
                        self.store.update_api_usage(
                            usage_id, input_tokens, output_tokens, cost
                        )
                    else:
                        self.store.update_api_tokens(usage_id, input_tokens, output_tokens)
                self.store.cache_put(key, parsed.model_dump_json())
                return parsed
            except Exception as error:
                last_error = error
                retryable = self._transient(error) or isinstance(
                    error, (ValidationError, ValueError, SyntaxError)
                )
                if retryable and attempt + 1 < self.settings.api_retry_attempts:
                    time.sleep(self.settings.api_retry_base_seconds * (2**attempt))
                    continue
                if not retryable:
                    raise
                break
        assert last_error is not None
        self.store.cache_put(key, "{}", f"{type(last_error).__name__}: {last_error}")
        raise last_error

    def embed(self, texts: list[str], purpose: str = "kite-embedding-v1") -> list[np.ndarray]:
        if not texts:
            return []
        model = self.settings.embedding_model
        kind = f"openai:{model}"
        output: list[np.ndarray | None] = [None] * len(texts)
        missing: list[tuple[int, str, str]] = []
        for index, value in enumerate(texts):
            identity = cache_key(model, purpose, "embedding", value)
            record_id = f"kite-text:{identity}"
            cached = self.store.get_embedding(record_id, kind)
            if cached:
                output[index] = deserialize(*cached)
            else:
                missing.append((index, record_id, value))

        size = max(1, self.settings.kite_embedding_batch_size)
        for start in range(0, len(missing), size):
            batch = missing[start : start + size]
            last_error: Exception | None = None
            for attempt in range(self.settings.api_retry_attempts):
                usage_id = self.store.reserve_api_call(
                    model,
                    purpose,
                    self.settings.kite_api_budget_usd,
                    min(self.settings.api_call_reserve_usd, 0.02),
                    self.allow_budget_overrun,
                    "kite-%",
                )
                try:
                    response = self.client.embeddings.create(
                        model=model, input=[value for _index, _identity, value in batch]
                    )
                    ordered = sorted(response.data, key=lambda item: item.index)
                    if len(ordered) != len(batch):
                        raise ValueError("embedding response length mismatch")
                    tokens = int(getattr(response.usage, "prompt_tokens", 0) or 0)
                    # text-embedding-3-small is currently $0.02 per million input tokens.
                    estimated = max(tokens * 0.02 / 1_000_000, 0.000001)
                    self.store.update_api_usage(usage_id, tokens, 0, estimated)
                    for (index, record_id, _value), item in zip(batch, ordered, strict=True):
                        vector = np.asarray(item.embedding, dtype=np.float32)
                        self.store.put_embedding(record_id, kind, serialize(vector), len(vector))
                        output[index] = vector
                    break
                except Exception as error:
                    last_error = error
                    if self._transient(error) and attempt + 1 < self.settings.api_retry_attempts:
                        time.sleep(self.settings.api_retry_base_seconds * (2**attempt))
                        continue
                    raise
            if last_error is not None and any(output[index] is None for index, *_ in batch):
                raise last_error
        return [value for value in output if value is not None]
