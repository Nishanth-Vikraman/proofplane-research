import json
import pytest
from proofplane_v0.context.bundle import build_bundle
from proofplane_v0.corpus.enrich import deterministic_hidden, deterministic_visible
from proofplane_v0.corpus.store import CorpusStore, cache_key
from proofplane_v0.corpus.vocabulary import VocabularyRegistry
from proofplane_v0.datasets.synthetic import SyntheticAdapter
from proofplane_v0.models import CorpusBugRecord
from proofplane_v0.retrieval.hybrid import retrieve


def test_cache_key_stable_and_sensitive():
    assert cache_key("m", "p", "s", "i") == cache_key("m", "p", "s", "i")
    assert cache_key("m", "p", "s", "i") != cache_key("m", "p", "s", "j")


def test_api_spend_guard(tmp_path):
    store = CorpusStore(tmp_path / "budget.db")
    store.reserve_api_call("model", "test", budget=0.25, reserve=0.25)
    with pytest.raises(RuntimeError, match="spend guard"):
        store.reserve_api_call("model", "test", budget=0.25, reserve=0.25)


def test_vocab_proposal_and_approve(tmp_path):
    store = CorpusStore(tmp_path / "x.db")
    registry = VocabularyRegistry(store.connection)
    slug = registry.propose("operations_changed", "Cache Refresh")
    assert slug == "cache-refresh" and registry.list("proposed")
    registry.approve(slug)
    assert registry.list("canonical")


def test_hidden_registry_expands_only_when_allowed(tmp_path):
    store = CorpusStore(tmp_path / "registry.db")
    registry = VocabularyRegistry(store.connection)

    family = registry.resolve(
        "bug_family", "NEW: stale derived state", allow_expand=True, definition="derived state"
    )
    assert family == "bf-0001"
    assert registry.resolve("bug_family", "stale derived state", allow_expand=False) == family
    assert (
        registry.resolve("bug_family", "UNSEEN: lifetime race", allow_expand=False)
        == "unseen:lifetime-race"
    )
    assert len(registry.entries("bug_family")) == 1


def test_primary_index_and_context_have_no_hidden_data(tmp_path):
    case = SyntheticAdapter().list_cases()[0]
    candidate, reference = SyntheticAdapter().materialize(case, tmp_path / "work")
    hidden_name = next(iter(case.hidden_tests))
    (reference / hidden_name).write_text(next(iter(case.hidden_tests.values())), encoding="utf-8")
    bundle = build_bundle(case.case_id, reference, candidate, hidden_test_names={hidden_name})
    assert hidden_name not in json.dumps(bundle.model_dump())
    record = CorpusBugRecord(
        dataset=case.dataset,
        project=case.project,
        bug_id=case.bug_id,
        split="train",
        visible_descriptor=deterministic_visible(case, bundle),
        hidden_payload=deterministic_hidden(case),
    )
    store = CorpusStore(tmp_path / "store.db")
    store.put(record)
    exported = json.dumps(store.primary_index_export())
    assert "hidden_payload" not in exported and case.metadata["family"] not in exported
    with pytest.raises(PermissionError):
        retrieve(record.visible_descriptor, [record], "oracle_hidden_family", 1, {}, {})


def test_prior_generated_tests_are_not_runtime_context(tmp_path):
    case = SyntheticAdapter().list_cases()[0]
    candidate, reference = SyntheticAdapter().materialize(case, tmp_path / "generated")
    for workspace in (candidate, reference):
        generated = workspace / "proofplane_generated" / "test_prior.py"
        generated.parent.mkdir(parents=True)
        generated.write_text("SECRET_PRIOR_GENERATION = True", encoding="utf-8")
    bundle = build_bundle(case.case_id, reference, candidate)
    assert "SECRET_PRIOR_GENERATION" not in bundle.model_dump_json()


def test_context_budget_selects_only_tests_related_to_changed_code(tmp_path):
    base, candidate = tmp_path / "base", tmp_path / "candidate"
    for workspace in (base, candidate):
        (workspace / "src").mkdir(parents=True)
        (workspace / "tests").mkdir()
    (base / "src" / "pricing.py").write_text(
        "def calculate_total(value):\n    return value + 1\n", encoding="utf-8"
    )
    (candidate / "src" / "pricing.py").write_text(
        "def calculate_total(value):\n    return value + 2\n", encoding="utf-8"
    )
    relevant = "from src.pricing import calculate_total\n\ndef test_total():\n    assert calculate_total(1)\n"
    for workspace in (base, candidate):
        (workspace / "tests" / "test_pricing.py").write_text(relevant, encoding="utf-8")
        for index in range(20):
            (workspace / "tests" / f"test_noise_{index}.py").write_text(
                ("def test_noise():\n    assert True\n" * 100), encoding="utf-8"
            )

    bundle = build_bundle("case", base, candidate, max_chars=5000)

    assert list(bundle.nearby_tests) == ["tests/test_pricing.py"]
    assert len(bundle.model_dump_json()) <= 5000
    assert "calculate_total" in bundle.changed_symbols
