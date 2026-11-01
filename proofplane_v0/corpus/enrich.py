from __future__ import annotations

import json
from pathlib import Path
import time
from typing import TypeVar

from openai import OpenAI
from pydantic import BaseModel
from proofplane_v0.context.bundle import ContextBundle
from proofplane_v0.corpus.store import CorpusStore, cache_key
from proofplane_v0.corpus.vocabulary import VocabularyRegistry
from proofplane_v0.datasets.base import DatasetCase
from proofplane_v0.models import CausalRetrievalDescriptor, HiddenBugPayload

T = TypeVar("T", bound=BaseModel)


def deterministic_visible(case: DatasetCase, bundle: ContextBundle) -> CausalRetrievalDescriptor:
    diff = bundle.visible_diff.lower()
    operations = (["mutation"] if any(x in diff for x in ("append", "extend", "=")) else []) + (
        ["comparison"] if "<" in diff or ">" in diff else []
    )
    return CausalRetrievalDescriptor(
        component_roles=["state-container"] if "class " in diff else ["pure-function"],
        changed_symbol_roles=bundle.changed_symbols,
        operations_present=operations,
        operations_changed=operations,
        stateful_resources=["in-memory-state"] if "class " in diff else [],
        control_flow_features=["exception-path"] if "raise" in diff else [],
        boundary_surfaces=["numeric-boundary"] if "<" in diff else [],
        existing_test_shapes=["unit-test"],
        apparent_test_gaps=["changed-path-not-covered"],
        event_sequence=operations,
        visible_change_summary=f"Changes {', '.join(bundle.changed_symbols)}",
        retrieval_text=f"python {' '.join(operations)} {' '.join(bundle.changed_symbols)} {bundle.visible_diff[:1000]}",
        confidence=0.75,
    )


def deterministic_hidden(case: DatasetCase) -> HiddenBugPayload:
    meta = case.metadata
    return HiddenBugPayload(
        mechanism_summary=meta["summary"],
        portable_prompt_summary=meta["summary"],
        trigger_sequence=meta["strategy"],
        violated_invariant="operation contract is preserved",
        observable_symptom="result differs between candidate and reference",
        test_strategy=meta["strategy"],
        mechanism_primitives=meta["primitives"],
        bug_family=meta["family"],
        root_cause=meta["summary"],
        fix_strategy="restore the violated invariant",
        confidence=0.95,
    )


class OpenAIEnricher:
    """Two-call structured extractor with a hard visible/hidden input boundary."""

    def __init__(
        self,
        store: CorpusStore,
        model: str,
        prompts_dir: Path | None = None,
        budget_usd: float = 5.0,
        reserve_usd: float = 0.25,
        allow_budget_overrun: bool = False,
        api_retry_attempts: int = 5,
        api_retry_base_seconds: float = 2.0,
    ):
        self.store = store
        self.model = model
        self.client = OpenAI()
        self.prompts_dir = prompts_dir or Path(__file__).parents[1] / "prompts"
        self.budget_usd = budget_usd
        self.reserve_usd = reserve_usd
        self.allow_budget_overrun = allow_budget_overrun
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

    def _parse(
        self, prompt_name: str, version: str, payload: dict[str, object], schema: type[T]
    ) -> T:
        normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        key = cache_key(self.model, version, schema.__name__, normalized)
        cached = self.store.cache_get(key)
        if cached:
            return schema.model_validate_json(cached)
        instructions = (self.prompts_dir / prompt_name).read_text(encoding="utf-8")
        last_error: Exception | None = None
        for attempt in range(self.api_retry_attempts):
            repair = ""
            if last_error is not None:
                repair = (
                    "\nThe prior structured response failed validation. Return a corrected object. "
                    "For portable_prompt_summary use only abstract mechanism language: no repository, "
                    "package, module, class, function, variable, endpoint, path, filename, or exact constant. "
                    f"Validation error: {last_error}"
                )
            try:
                usage_id = self.store.reserve_api_call(
                    self.model,
                    version,
                    self.budget_usd,
                    self.reserve_usd,
                    self.allow_budget_overrun,
                )
                response = self.client.responses.parse(
                    model=self.model,
                    input=[
                        {"role": "system", "content": instructions + repair},
                        {"role": "user", "content": normalized},
                    ],
                    text_format=schema,
                )
                parsed = response.output_parsed
                if parsed is None:
                    raise RuntimeError("structured extraction returned no parsed output")
                usage = response.usage
                if usage:
                    self.store.update_api_tokens(
                        usage_id, int(usage.input_tokens), int(usage.output_tokens)
                    )
                self.store.cache_put(key, parsed.model_dump_json())
                return parsed
            except Exception as error:
                last_error = error
                if self._transient(error) and attempt + 1 < self.api_retry_attempts:
                    time.sleep(self.api_retry_base_seconds * (2**attempt))
        assert last_error is not None
        self.store.cache_put(key, "{}", f"{type(last_error).__name__}: {last_error}")
        raise last_error

    def visible(self, bundle: ContextBundle) -> CausalRetrievalDescriptor:
        # This serialized payload is the complete allowlist for visible extraction.
        return self._parse(
            "enrich_visible.md",
            "visible-v1",
            bundle.model_dump(mode="json"),
            CausalRetrievalDescriptor,
        )

    def hidden(
        self,
        case: DatasetCase,
        bundle: ContextBundle,
        oracle_tests: dict[str, str],
        patch: str,
        split: str,
    ) -> HiddenBugPayload:
        registry = VocabularyRegistry(self.store.connection)
        allow_expand = split == "train"
        parsed = self._parse(
            "enrich_hidden.md",
            "hidden-v2-registry",
            {
                "case_id": case.case_id,
                "visible_change": bundle.visible_diff,
                "source_fix_patch": patch,
                "recorded_triggering_tests": oracle_tests,
                "run_test": case.metadata.get("run_test", ""),
                "issue_cause": case.metadata.get("project_info", {}).get("cause", ""),
                "registry_mode": "expand" if allow_expand else "frozen",
                "registry": registry.snapshot(),
            },
            HiddenBugPayload,
        )
        family = registry.resolve(
            "bug_family",
            parsed.bug_family,
            allow_expand=allow_expand,
            definition=parsed.mechanism_summary,
            reason=f"assigned from {case.case_id}",
        )
        primitives = [
            registry.resolve(
                "mechanism_primitive",
                value,
                allow_expand=allow_expand,
                definition=value.removeprefix("NEW:").strip(),
                reason=f"assigned from {case.case_id}",
            )
            for value in parsed.mechanism_primitives
        ]
        return parsed.model_copy(
            update={"bug_family": family, "mechanism_primitives": list(dict.fromkeys(primitives))}
        )
