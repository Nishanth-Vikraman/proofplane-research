from __future__ import annotations

import ast
import hashlib
from collections import Counter
from pathlib import Path

import numpy as np

from proofplane_v0.config import Settings
from proofplane_v0.corpus.store import CorpusStore
from proofplane_v0.datasets.base import DatasetCase
from proofplane_v0.kite.ast_units import (
    structural_query_units,
    structural_similarity,
    structural_unit,
    weighted_structural_similarity,
)
from proofplane_v0.kite.benchmark import build_focal_benchmark, localize_focal_methods
from proofplane_v0.kite.conformance import verify_conformance
from proofplane_v0.kite.injection import _behavioral_probe_is_black_box, apply_and_validate
from proofplane_v0.kite.knowledge import (
    _attach_buggy_direction,
    _semantic_groups,
    _synthesize_group,
    enrich_explicit_fields,
)
from proofplane_v0.kite.models import (
    FocalBenchmarkCase,
    FocalBenchmarkManifest,
    FocalMethodSpec,
    InjectionPlan,
    KiteTestDraft,
    MechanismAlignment,
    RetrievalMatch,
    ReviewedMechanismGroup,
    TargetCodeUnit,
    TargetSemanticFields,
    TransferableMechanism,
)
from proofplane_v0.kite.retrieval import retrieve
from proofplane_v0.kite.runner import (
    HiddenEvaluationHandle,
    _run_feedback_arm,
    _run_repaired_direct_arm,
    _select_injections,
)
from proofplane_v0.kite.target_profile import build_reference_profile, enrich_target_semantics
from proofplane_v0.models import ExecutionOutcome
from proofplane_v0.retrieval.embeddings import deterministic_embedding


def _mechanism(project: str = "train-project") -> TransferableMechanism:
    source = "def historical(value):\n    return value <= 10\n"
    node = ast.parse(source).body[0]
    return TransferableMechanism(
        mechanism_id=f"mechanism:{project}:1",
        corpus_id=f"bugsinpy:{project}:1",
        source_project=project,
        what="an inclusive boundary contract becomes exclusive",
        where="comparison at a numeric boundary",
        how="replace inclusive behavior with exclusive behavior",
        preconditions=["a value lies on the boundary"],
        actions=["invoke the public operation"],
        transition="the boundary comparison changes",
        violated_property="the boundary value remains accepted",
        observable="the boundary value is rejected",
        test_shape=["assert behavior exactly at the boundary"],
        structural_units=[structural_unit("old.py", "historical", node, source)],
    )


def _target_unit(case_id: str = "bugsinpy:eval-project:1") -> TargetCodeUnit:
    source = "def allowed(value, limit):\n    return value <= limit\n"
    node = ast.parse(source).body[0]
    structural = structural_unit("pkg/rules.py", "allowed", node, source)
    return TargetCodeUnit(
        unit_id=f"{case_id}:unit",
        file="pkg/rules.py",
        qualname="allowed",
        signature="def allowed(value, limit)",
        docstring="",
        source=source,
        semantic_text="numeric boundary comparison for allowing values",
        structural=structural,
        content_hash=structural.source_hash,
    )


def _alignment() -> list[MechanismAlignment]:
    return [
        MechanismAlignment(
            dimension="what",
            historical_element="inclusive behavior becomes exclusive",
            target_evidence="the target returns an inclusive comparison",
            edit_effect="the edit rejects the boundary value",
        ),
        MechanismAlignment(
            dimension="where",
            historical_element="comparison at a numeric boundary",
            target_evidence="allowed compares value with limit",
            edit_effect="the local comparison remains executable",
        ),
        MechanismAlignment(
            dimension="how",
            historical_element="replace inclusive with exclusive behavior",
            target_evidence="the target contains <= exactly once",
            edit_effect="<= becomes < without changing the signature",
        ),
    ]


def test_reference_profile_never_reads_buggy_workspace(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    (reference / "pkg").mkdir(parents=True)
    (reference / "tests").mkdir()
    (reference / "pkg" / "rules.py").write_text(
        "def allowed(value, limit):\n    return value <= limit\n", encoding="utf-8"
    )
    (reference / "tests" / "test_rules.py").write_text(
        "from pkg.rules import allowed\n\ndef test_allowed(): assert allowed(1, 2)\n",
        encoding="utf-8",
    )
    case = DatasetCase(
        dataset="bugsinpy",
        project="eval-project",
        bug_id="1",
        buggy_source={"secret.py": "HIDDEN_REAL_BUG_MARKER"},
        fixed_source={},
        pre_fix_tests={},
        hidden_tests={"hidden.py": "HIDDEN_TEST_MARKER"},
        metadata={"fixed_commit_id": "fixed"},
    )
    store = CorpusStore(tmp_path / "corpus.sqlite3")
    profile = build_reference_profile(case, reference, store, 20, 5000)
    serialized = profile.model_dump_json()
    assert "allowed" in serialized
    assert "HIDDEN_REAL_BUG_MARKER" not in serialized
    assert "HIDDEN_TEST_MARKER" not in serialized


def test_reference_profile_finds_api_examples_for_private_focal_method(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference"
    (reference / "pkg").mkdir(parents=True)
    (reference / "tests").mkdir()
    (reference / "pkg" / "handlers.py").write_text(
        "def _normalize_response(value):\n"
        "    if value is None:\n"
        "        return {}\n"
        "    return {'value': value}\n",
        encoding="utf-8",
    )
    (reference / "tests" / "test_public_handler.py").write_text(
        "from pkg.handlers import public_handler\n\n"
        "def test_handler_api():\n    assert public_handler(None) == {}\n",
        encoding="utf-8",
    )
    case = DatasetCase(
        dataset="bugsinpy",
        project="eval-project",
        bug_id="1",
        buggy_source={},
        fixed_source={},
        pre_fix_tests={},
        hidden_tests={},
        metadata={"fixed_commit_id": "fixed"},
    )
    profile = build_reference_profile(
        case, reference, CorpusStore(tmp_path / "context.sqlite3"), 20, 5000
    )
    private = next(value for value in profile.units if value.qualname == "_normalize_response")
    assert "tests/test_public_handler.py" in private.related_tests
    assert all("candidate" not in unit.source for unit in profile.units)


def test_focal_profile_contains_only_sanitized_patch_selected_method(tmp_path: Path) -> None:
    reference = tmp_path / "reference"
    (reference / "pkg").mkdir(parents=True)
    selected = "def selected(value):\n    return value + 1\n"
    (reference / "pkg" / "rules.py").write_text(
        selected + "\ndef distracting(value):\n    return value * 100\n", encoding="utf-8"
    )
    node = ast.parse(selected).body[0]
    focal = FocalMethodSpec(
        file="pkg/rules.py",
        qualname="selected",
        start_line=1,
        end_line=2,
        reference_source_hash=hashlib.sha256(selected.rstrip("\n").encode()).hexdigest(),
    )
    case = DatasetCase(
        dataset="bugsinpy",
        project="eval-project",
        bug_id="1",
        buggy_source={},
        fixed_source={},
        pre_fix_tests={},
        hidden_tests={},
        metadata={"fixed_commit_id": "fixed"},
    )
    profile = build_reference_profile(
        case,
        reference,
        CorpusStore(tmp_path / "corpus.sqlite3"),
        20,
        5000,
        focal_methods=[focal],
    )
    assert [(unit.file, unit.qualname) for unit in profile.units] == [
        ("pkg/rules.py", "selected")
    ]
    assert ast.dump(node, include_attributes=False)


def test_target_semantic_profile_uses_only_reference_profile_payload() -> None:
    from proofplane_v0.kite.models import ReferenceTargetProfile

    class FakeService:
        def __init__(self) -> None:
            self.payloads: list[dict[str, object]] = []

        def parse(self, **kwargs):
            self.payloads.append(kwargs["payload"])
            return TargetSemanticFields(
                mechanism_bearing_behavior="preserves inclusive boundary semantics",
                applicability_context="a typed comparison and public return value",
                edit_affordance="the local comparison operator can be changed",
            )

    profile = ReferenceTargetProfile(
        case_id="bugsinpy:eval-project:1",
        project="eval-project",
        reference_commit="fixed",
        units=[_target_unit()],
        content_hash="base",
    )
    service = FakeService()
    enriched = enrich_target_semantics(profile, service)  # type: ignore[arg-type]
    serialized = enriched.model_dump_json()
    assert "inclusive boundary" in serialized
    assert "candidate" not in serialized
    assert len(service.payloads) == 1


def test_patch_localizer_resolves_methods_and_excludes_test_assets(tmp_path: Path) -> None:
    reference, candidate = tmp_path / "reference", tmp_path / "candidate"
    for workspace in (reference, candidate):
        (workspace / "pkg").mkdir(parents=True)
        (workspace / "pending_tests").mkdir()
    (candidate / "pkg" / "rules.py").write_text(
        "def changed(x):\n    return x < 1\n\ndef stable(x):\n    return x\n", encoding="utf-8"
    )
    (reference / "pkg" / "rules.py").write_text(
        "def changed(x):\n    return x <= 1\n\ndef stable(x):\n    return x\n", encoding="utf-8"
    )
    (candidate / "pending_tests" / "main.py").write_text(
        "def helper():\n    return False\n", encoding="utf-8"
    )
    (reference / "pending_tests" / "main.py").write_text(
        "def helper():\n    return True\n", encoding="utf-8"
    )
    focal = localize_focal_methods(reference, candidate)
    assert [(value.file, value.qualname) for value in focal] == [("pkg/rules.py", "changed")]


def test_focal_benchmark_max_method_knob_and_sanitized_export(tmp_path: Path) -> None:
    case = DatasetCase(
        dataset="bugsinpy",
        project="sample",
        bug_id="1",
        buggy_source={},
        fixed_source={},
        pre_fix_tests={},
        hidden_tests={},
        metadata={},
    )

    class Adapter:
        def list_cases(self):
            return [case]

    store = CorpusStore(tmp_path / "corpus.sqlite3")
    store.put_case(case.case_id, case.dataset, case.project, case.bug_id, {}, prepared=True)
    store.put_splits({case.case_id: "eval"})
    workspace = tmp_path / "data" / "workspaces" / "bugsinpy-sample-1"
    reference, candidate = workspace / "reference", workspace / "candidate"
    for value in (reference, candidate):
        (value / "pkg").mkdir(parents=True)
    candidate_source = (
        "def first(x):\n    return x < 1\n\n"
        "def second(x):\n    return x > 2\n"
    )
    reference_source = candidate_source.replace("< 1", "<= 1").replace("> 2", ">= 2")
    (candidate / "pkg" / "rules.py").write_text(candidate_source, encoding="utf-8")
    (reference / "pkg" / "rules.py").write_text(reference_source, encoding="utf-8")
    one = build_focal_benchmark(
        adapter=Adapter(),  # type: ignore[arg-type]
        store=store,
        data_dir=tmp_path / "data",
        split="eval",
        max_focal_methods=1,
        output=tmp_path / "one.json",
    )
    assert not one.cases[0].eligible
    two = build_focal_benchmark(
        adapter=Adapter(),  # type: ignore[arg-type]
        store=store,
        data_dir=tmp_path / "data",
        split="eval",
        max_focal_methods=2,
        output=tmp_path / "two.json",
    )
    assert two.cases[0].eligible
    assert len(two.cases[0].focal_methods) == 2
    exported = (tmp_path / "two.json").read_text(encoding="utf-8")
    assert "return x" not in exported
    assert "candidate" not in exported


def test_dual_retrieval_is_cross_project_and_uses_real_embedding_interface(tmp_path: Path) -> None:
    class FakeEmbeddingService:
        def __init__(self) -> None:
            self.calls: list[str] = []

        def embed(self, texts: list[str], purpose: str) -> list[object]:
            self.calls.append(purpose)
            return [deterministic_embedding(text, 16) for text in texts]

    unit = _target_unit()
    from proofplane_v0.kite.models import ReferenceTargetProfile

    profile = ReferenceTargetProfile(
        case_id="bugsinpy:eval-project:1",
        project="eval-project",
        reference_commit="fixed",
        units=[unit],
        content_hash="profile",
    )
    settings = Settings(
        database=tmp_path / "corpus.sqlite3",
        embedding_model="text-embedding-3-small",
        kite_retrieval_top_k=1,
    )
    store = CorpusStore(settings.database)
    service = FakeEmbeddingService()
    matches = retrieve(
        profile,
        [_mechanism("train-project"), _mechanism("eval-project")],
        service,  # type: ignore[arg-type]
        store,
        settings,
    )
    assert len(matches) == 1
    assert matches[0].mechanism.source_project == "train-project"
    assert matches[0].support_paths == ["semantic", "structural"]
    assert service.calls == [
        "kite-mechanism-dimension-embedding-v2",
        "kite-target-dimension-embedding-v2",
    ]


def test_local_injection_executor_preserves_reference_and_accepts_valid_plan(
    tmp_path: Path, monkeypatch
) -> None:
    reference = tmp_path / "reference"
    (reference / "pkg").mkdir(parents=True)
    source = "def allowed(value, limit):\n    return value <= limit\n"
    (reference / "pkg" / "rules.py").write_text(source, encoding="utf-8")
    unit = _target_unit()
    match = RetrievalMatch(
        rank=1,
        mechanism=_mechanism(),
        target_unit=unit,
        semantic_score=0.8,
        structural_score=0.7,
        fused_score=0.77,
    )
    plan = InjectionPlan(
        mechanism_to_reproduce="make the inclusive boundary exclusive",
        target_anchor_id="anchor-000",
        applicability_reason="the target contains the same comparison affordance",
        before="    return value <= limit",
        after="    return value < limit",
        expected_behavioral_change="the exact limit becomes rejected",
        trigger_recipe=["call allowed with value equal to limit"],
        mechanism_alignment=_alignment(),
        behavioral_probe=(
            "from pkg.rules import allowed\n\n"
            "def test_boundary():\n"
            "    assert allowed(10, 10)\n"
        ),
    )
    monkeypatch.setattr(
        "proofplane_v0.kite.injection.run_workspace_command",
        lambda *args, **kwargs: ExecutionOutcome(status="pass", return_code=0),
    )
    monkeypatch.setattr(
        "proofplane_v0.kite.injection.run_test",
        lambda workspace, *args, **kwargs: ExecutionOutcome(
            status="pass" if workspace == reference else "fail",
            return_code=0 if workspace == reference else 1,
        ),
    )
    settings = Settings(data_dir=tmp_path / "data", database=tmp_path / "corpus.sqlite3")
    store = CorpusStore(settings.database)
    validation = apply_and_validate(
        match=match,
        plan=plan,
        reference=reference,
        synthetic_root=tmp_path / "synthetic-root",
        image="reference-image",
        settings=settings,
        store=store,
    )
    assert validation.accepted
    assert (reference / unit.file).read_text(encoding="utf-8") == source
    assert "return value < limit" in (
        Path(validation.synthetic_workspace) / unit.file
    ).read_text(encoding="utf-8")
    assert validation.gates["behavioral_non_triviality"]
    assert validation.gates["behavioral_probe_no_source_introspection"]


def test_injection_rejects_parseable_variant_without_behavioral_distinction(
    tmp_path: Path, monkeypatch
) -> None:
    reference = tmp_path / "reference"
    (reference / "pkg").mkdir(parents=True)
    source = "def allowed(value, limit):\n    return value <= limit\n"
    (reference / "pkg" / "rules.py").write_text(source, encoding="utf-8")
    unit = _target_unit()
    match = RetrievalMatch(
        rank=1,
        mechanism=_mechanism(),
        target_unit=unit,
        semantic_score=0.8,
        structural_score=0.7,
        fused_score=0.77,
    )
    plan = InjectionPlan(
        mechanism_to_reproduce="boundary",
        target_anchor_id="anchor-000",
        applicability_reason="comparison",
        before="    return value <= limit",
        after="    return value < limit",
        expected_behavioral_change="boundary changes",
        trigger_recipe=["exercise boundary"],
        mechanism_alignment=_alignment(),
        behavioral_probe="def test_probe():\n    assert True\n",
    )
    monkeypatch.setattr(
        "proofplane_v0.kite.injection.run_workspace_command",
        lambda *args, **kwargs: ExecutionOutcome(status="pass", return_code=0),
    )
    monkeypatch.setattr(
        "proofplane_v0.kite.injection.run_test",
        lambda *args, **kwargs: ExecutionOutcome(status="pass", return_code=0),
    )
    validation = apply_and_validate(
        match=match,
        plan=plan,
        reference=reference,
        synthetic_root=tmp_path / "synthetic-root",
        image="reference-image",
        settings=Settings(data_dir=tmp_path / "data", database=tmp_path / "corpus.sqlite3"),
        store=CorpusStore(tmp_path / "corpus.sqlite3"),
    )
    assert not validation.accepted
    assert not validation.gates["behavioral_non_triviality"]


def test_behavioral_probe_cannot_win_by_reading_implementation() -> None:
    assert _behavioral_probe_is_black_box(
        "from pkg.rules import allowed\n\ndef test_behavior():\n    assert allowed(1, 1)\n"
    )
    assert not _behavioral_probe_is_black_box(
        "import inspect\nfrom pkg.rules import allowed\n\n"
        "def test_source():\n    assert '<=' in inspect.getsource(allowed)\n"
    )


def test_structural_fingerprint_rewards_exact_shape() -> None:
    left_source = "def f(x):\n    return x <= 1\n"
    right_source = "def g(value):\n    return value <= 99\n"
    different_source = "def h(value):\n    raise ValueError(value)\n"
    left = structural_unit("a.py", "f", ast.parse(left_source).body[0], left_source)
    right = structural_unit("b.py", "g", ast.parse(right_source).body[0], right_source)
    different = structural_unit(
        "c.py", "h", ast.parse(different_source).body[0], different_source
    )
    assert left.fingerprint == right.fingerprint
    assert structural_similarity(left, right) > structural_similarity(left, different)


def test_weighted_structural_similarity_does_not_saturate_on_common_shape() -> None:
    source = "def f(x):\n    return x <= 1\n"
    unit = structural_unit("a.py", "f", ast.parse(source).body[0], source)
    common = weighted_structural_similarity(
        unit,
        unit,
        fingerprint_frequency=Counter({unit.fingerprint: 10}),
        coarse_frequency=Counter({unit.coarse_key: 10}),
        signature_frequency=Counter({value: 10 for value in unit.signatures}),
        document_count=10,
    )
    rare = weighted_structural_similarity(
        unit,
        unit,
        fingerprint_frequency=Counter({unit.fingerprint: 1}),
        coarse_frequency=Counter({unit.coarse_key: 1}),
        signature_frequency=Counter({value: 1 for value in unit.signatures}),
        document_count=10,
    )
    assert 0.0 <= common < rare < 1.0


def test_control_header_replacement_is_checked_as_part_of_full_focal_unit(
    tmp_path: Path, monkeypatch
) -> None:
    reference = tmp_path / "reference"
    (reference / "pkg").mkdir(parents=True)
    source = (
        "def allowed(value, limit):\n"
        "    if value <= limit:\n"
        "        return True\n"
        "    return False\n"
    )
    (reference / "pkg" / "rules.py").write_text(source, encoding="utf-8")
    node = ast.parse(source).body[0]
    assert isinstance(node, ast.FunctionDef)
    unit = _target_unit().model_copy(
        update={
            "source": source,
            "structural": structural_unit("pkg/rules.py", "allowed", node, source),
            "structural_queries": structural_query_units("pkg/rules.py", "allowed", node),
        }
    )
    match = RetrievalMatch(
        rank=1,
        mechanism=_mechanism(),
        target_unit=unit,
        semantic_score=0.8,
        structural_score=0.7,
        fused_score=0.77,
    )
    header_index = next(
        index
        for index, anchor in enumerate(unit.structural_queries)
        if anchor.node_type == "If"
    )
    plan = InjectionPlan(
        mechanism_to_reproduce="make the inclusive boundary exclusive",
        target_anchor_id=f"anchor-{header_index:03d}",
        applicability_reason="the target contains the same boundary control header",
        before="    if value <= limit:",
        after="    if value < limit:",
        expected_behavioral_change="the exact limit becomes rejected",
        trigger_recipe=["call allowed at the limit"],
        mechanism_alignment=_alignment(),
        behavioral_probe=(
            "from pkg.rules import allowed\n\n"
            "def test_boundary():\n    assert allowed(10, 10)\n"
        ),
    )
    monkeypatch.setattr(
        "proofplane_v0.kite.injection.run_workspace_command",
        lambda *args, **kwargs: ExecutionOutcome(status="pass", return_code=0),
    )
    monkeypatch.setattr(
        "proofplane_v0.kite.injection.run_test",
        lambda workspace, *args, **kwargs: ExecutionOutcome(
            status="pass" if workspace == reference else "fail",
            return_code=0 if workspace == reference else 1,
        ),
    )
    validation = apply_and_validate(
        match=match,
        plan=plan,
        reference=reference,
        synthetic_root=tmp_path / "synthetic-root",
        image="reference-image",
        settings=Settings(data_dir=tmp_path / "data", database=tmp_path / "gate.sqlite3"),
        store=CorpusStore(tmp_path / "gate.sqlite3"),
    )
    assert validation.accepted
    assert validation.gates["syntax_validity"]


def test_local_anchor_relations_and_correct_to_buggy_direction_are_recorded() -> None:
    correct_source = (
        "def allowed(value, limit):\n"
        "    if value <= limit:\n"
        "        return True\n"
        "    return False\n"
    )
    buggy_source = correct_source.replace("<=", "<")
    correct_node = ast.parse(correct_source).body[0]
    buggy_node = ast.parse(buggy_source).body[0]
    assert isinstance(correct_node, ast.FunctionDef)
    assert isinstance(buggy_node, ast.FunctionDef)
    correct = structural_query_units("rules.py", "allowed", correct_node)
    buggy = structural_query_units("rules.py", "allowed", buggy_node)
    assert any(unit.relations for unit in correct)
    directional = _attach_buggy_direction(correct, buggy)
    assert all(unit.buggy_fingerprint for unit in directional)
    assert all("correct_to_buggy" in unit.edit_operations for unit in directional)


def test_injection_selection_deduplicates_and_keeps_distinct_variants(
    tmp_path: Path, monkeypatch
) -> None:
    unit = _target_unit()
    matches = [
        RetrievalMatch(
            rank=index,
            mechanism=_mechanism(f"train-{index}"),
            target_unit=unit,
            semantic_score=0.8,
            structural_score=0.7,
            fused_score=0.77,
        )
        for index in range(1, 4)
    ]
    plan = InjectionPlan(
        mechanism_to_reproduce="boundary",
        target_anchor_id="anchor-000",
        applicability_reason="comparison",
        before="    return value <= limit",
        after="    return value < limit",
        expected_behavioral_change="boundary",
        trigger_recipe=["boundary"],
        mechanism_alignment=_alignment(),
        behavioral_probe="def test_probe():\n    assert True\n",
    )
    monkeypatch.setattr("proofplane_v0.kite.runner.propose_injection", lambda *args: plan)
    monkeypatch.setattr(
        "proofplane_v0.kite.runner.injection_alignment_score", lambda *args: 1.0
    )
    monkeypatch.setattr(
        "proofplane_v0.kite.runner.apply_and_validate",
        lambda **kwargs: __import__(
            "proofplane_v0.kite.models", fromlist=["InjectionValidation"]
        ).InjectionValidation(
            accepted=True,
            gates={"accepted": True},
            patch=(
                "same-patch"
                if kwargs["match"].rank in {1, 2}
                else kwargs["match"].mechanism.mechanism_id
            ),
            synthetic_workspace=str(tmp_path / "synthetic"),
        ),
    )
    settings = Settings(
        data_dir=tmp_path,
        database=tmp_path / "corpus.sqlite3",
        kite_max_injection_attempts=3,
        kite_max_accepted_injections=2,
    )
    accepted, attempts = _select_injections(
        matches=matches,
        service=object(),  # type: ignore[arg-type]
        store=CorpusStore(settings.database),
        reference=tmp_path / "reference",
        reference_image="image",
        settings=settings,
        artifact=tmp_path / "artifact",
    )
    assert attempts == 3
    assert len(accepted) == 2
    duplicate = (
        tmp_path / "artifact" / "injections" / "attempt-2" / "selection.json"
    ).read_text(encoding="utf-8")
    assert '"reason": "duplicate_patch"' in duplicate


def test_feedback_arm_retains_base_suite_when_no_injection_is_valid(
    tmp_path: Path, monkeypatch
) -> None:
    unit = _target_unit()
    match = RetrievalMatch(
        rank=1,
        mechanism=_mechanism(),
        target_unit=unit,
        semantic_score=0.8,
        structural_score=0.7,
        fused_score=0.77,
    )
    draft = KiteTestDraft(
        rationale="exercise the visible boundary",
        relative_path="test_boundary.py",
        test_code="def test_boundary():\n    assert True\n",
        expected_behavior="the reference remains valid",
    )
    monkeypatch.setattr(
        "proofplane_v0.kite.runner._select_injections", lambda **kwargs: ([], 3)
    )
    monkeypatch.setattr(
        "proofplane_v0.kite.runner.generate_tests", lambda **kwargs: [draft]
    )
    monkeypatch.setattr(
        "proofplane_v0.kite.runner._reference_filter",
        lambda **kwargs: ([draft], []),
    )
    monkeypatch.setattr(
        "proofplane_v0.kite.runner._evaluate_hidden",
        lambda **kwargs: (1, 1, [True], [{"success": True}]),
    )
    case = DatasetCase(
        dataset="bugsinpy",
        project="eval-project",
        bug_id="1",
        buggy_source={},
        fixed_source={},
        pre_fix_tests={},
        hidden_tests={},
    )
    result = _run_feedback_arm(
        case=case,
        arm="retrieved_mechanism_sbf",
        matches=[match],
        service=object(),  # type: ignore[arg-type]
        store=CorpusStore(tmp_path / "fallback.sqlite3"),
        reference=tmp_path / "reference",
        reference_image="reference-image",
        hidden=HiddenEvaluationHandle(tmp_path / "hidden", "hidden-image"),
        settings=Settings(data_dir=tmp_path, database=tmp_path / "fallback.sqlite3"),
        k=1,
        case_dir=tmp_path / "case",
    )
    assert result.injection_accepted is False
    assert result.generated_count == 1
    assert result.real_bug_kills == 1
    assert result.success_at_k["success@1"]


def test_matched_control_and_sbf_consume_the_exact_same_prepared_suite(
    tmp_path: Path, monkeypatch
) -> None:
    unit = _target_unit()
    match = RetrievalMatch(
        rank=1,
        mechanism=_mechanism(),
        target_unit=unit,
        semantic_score=0.8,
        structural_score=0.7,
        fused_score=0.77,
    )
    draft = KiteTestDraft(
        rationale="shared reference-passing test",
        relative_path="test_shared.py",
        test_code="def test_shared():\n    assert True\n",
        expected_behavior="pass",
    )
    case = DatasetCase(
        dataset="bugsinpy",
        project="eval-project",
        bug_id="1",
        buggy_source={},
        fixed_source={},
        pre_fix_tests={},
        hidden_tests={},
    )
    observed: list[list[str]] = []
    monkeypatch.setattr(
        "proofplane_v0.kite.runner._build_reference_suite",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("suite was rebuilt")),
    )
    monkeypatch.setattr(
        "proofplane_v0.kite.runner._select_injections", lambda **kwargs: ([], 3)
    )
    monkeypatch.setattr(
        "proofplane_v0.kite.runner._evaluate_hidden",
        lambda **kwargs: (
            observed.append([value.test_code for value in kwargs["drafts"]]) or 1,
            0,
            [False],
            [{"success": False}],
        ),
    )
    shared = ([draft], 7)
    _run_repaired_direct_arm(
        case=case,
        match=match,
        service=object(),  # type: ignore[arg-type]
        reference=tmp_path / "reference",
        reference_image="reference-image",
        hidden=HiddenEvaluationHandle(tmp_path / "hidden", "hidden-image"),
        settings=Settings(data_dir=tmp_path),
        k=1,
        case_dir=tmp_path / "case",
        prepared_base_suite=shared,
    )
    _run_feedback_arm(
        case=case,
        arm="retrieved_mechanism_sbf",
        matches=[match],
        service=object(),  # type: ignore[arg-type]
        store=CorpusStore(tmp_path / "shared.sqlite3"),
        reference=tmp_path / "reference",
        reference_image="reference-image",
        hidden=HiddenEvaluationHandle(tmp_path / "hidden", "hidden-image"),
        settings=Settings(data_dir=tmp_path),
        k=1,
        case_dir=tmp_path / "case",
        prepared_base_suite=shared,
    )
    assert observed == [[draft.test_code], [draft.test_code]]


def test_conformance_gate_is_machine_decidable(tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest = FocalBenchmarkManifest(
        split="eval",
        max_focal_methods=1,
        cases=[
            FocalBenchmarkCase(
                case_id="bugsinpy:eval-project:1",
                project="eval-project",
                eligible=True,
                focal_methods=[
                    FocalMethodSpec(
                        file="pkg/rules.py",
                        qualname="allowed",
                        start_line=1,
                        end_line=2,
                        reference_source_hash="hash",
                    )
                ],
            )
        ],
        content_hash="manifest-hash",
    )
    manifest_path.write_text(manifest.model_dump_json(), encoding="utf-8")
    report = verify_conformance(
        settings=Settings(
            reasoning_model="gpt-5.4-nano-2026-03-17",
            kite_temperature=0.1,
            kite_max_enhancement_rounds=5,
        ),
        mechanisms=[_mechanism()],
        benchmark_manifest=manifest_path,
    )
    assert report.protocol_ready
    assert not report.exact_paper_replication
    assert {value.check_id for value in report.checks} >= {
        "benchmark.hidden_boundary",
        "injection.deterministic_gates",
        "feedback.canonical_suite",
    }


def test_explicit_mechanism_enrichment_refuses_eval_records() -> None:
    from proofplane_v0.models import (
        CausalRetrievalDescriptor,
        CorpusBugRecord,
        HiddenBugPayload,
    )

    record = CorpusBugRecord(
        dataset="bugsinpy",
        project="eval-project",
        bug_id="1",
        split="eval",
        visible_descriptor=CausalRetrievalDescriptor(
            visible_change_summary="summary",
            retrieval_text="text",
            confidence=1.0,
        ),
        hidden_payload=HiddenBugPayload(
            mechanism_summary="mechanism",
            portable_prompt_summary="portable mechanism",
            trigger_sequence=["trigger"],
            violated_invariant="invariant",
            observable_symptom="symptom",
            test_strategy=["strategy"],
            mechanism_primitives=["primitive"],
            bug_family="family",
            root_cause="cause",
            fix_strategy="fix",
            confidence=1.0,
        ),
    )
    try:
        enrich_explicit_fields(record, _mechanism(), object())  # type: ignore[arg-type]
    except ValueError as error:
        assert "train-only" in str(error)
    else:
        raise AssertionError("eval records must never enter explicit historical enrichment")


def test_semantic_grouping_uses_complete_link_not_chaining() -> None:
    mechanisms = [_mechanism(f"project-{index}") for index in range(3)]
    vectors = [
        np.asarray([1.0, 0.0]),
        np.asarray([0.8, 0.6]),
        np.asarray([0.28, 0.96]),
    ]
    groups = _semantic_groups(mechanisms, vectors, threshold=0.7, max_group_size=8)
    assert sorted(len(value) for value in groups) == [1, 2]


def test_group_synthesis_retains_all_project_provenance() -> None:
    review = ReviewedMechanismGroup(
        what="a retained state contract is violated",
        where="a state-bearing transition reconstructs an object",
        how="replace preservation with default reconstruction",
        semantic_group="state preservation",
        expected_buggy_behavior="the specialized state is lost",
        preconditions=["specialized state exists"],
        actions=["cross the reconstruction transition"],
        transition="reconstruction discards specialized state",
        violated_property="state survives reconstruction",
        observable="downstream behavior uses defaults",
        test_shape=["construct, transition, and assert preserved state"],
        representative_evidence=["a reconstruction path selects a generic constructor"],
        transferable=True,
        review_confidence=0.9,
    )
    settings = Settings(kite_knowledge_review_min_confidence=0.65)
    grouped = _synthesize_group([_mechanism("one"), _mechanism("two")], review, settings)
    assert grouped.derivation == "train_llm_grouped"
    assert grouped.review_status == "accepted"
    assert grouped.source_projects == ["one", "two"]
    assert len(grouped.supporting_bug_unit_ids) == 2
