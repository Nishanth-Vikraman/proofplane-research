from __future__ import annotations

import argparse
import json
from pathlib import Path

from proofplane_v0.config import Settings
from proofplane_v0.corpus.store import CorpusStore
from proofplane_v0.evaluation.docker_runner import build_workspace_image, remove_workspace_image
from proofplane_v0.kite.benchmark import load_focal_benchmark
from proofplane_v0.kite.injection import apply_and_validate, available_local_anchors
from proofplane_v0.kite.models import (
    InjectionPlan,
    MechanismAlignment,
    RetrievalMatch,
    TransferableMechanism,
)
from proofplane_v0.kite.target_profile import build_reference_profile
from proofplane_v0.workflow import adapter_for


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/full.yaml"))
    parser.add_argument("--root", type=Path, default=Path("BugsInPy"))
    parser.add_argument("--case-id", default="bugsinpy:fastapi:10")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/full/artifacts/kite-verification/docker-gate.json"),
    )
    args = parser.parse_args()
    settings = Settings.load(args.config)
    store = CorpusStore(settings.database)
    adapter = adapter_for("bugsinpy", args.root)
    case = next(value for value in adapter.list_cases() if value.case_id == args.case_id)
    benchmark = load_focal_benchmark(
        settings.artifacts_dir / "kite-benchmark" / "manifest.json"
    )
    benchmark_case = next(value for value in benchmark.cases if value.case_id == args.case_id)
    if not benchmark_case.eligible or len(benchmark_case.focal_methods) != 1:
        raise RuntimeError("Docker gate requires one eligible focal method")
    reference = settings.data_dir / "workspaces" / args.case_id.replace(":", "-") / "reference"
    profile = build_reference_profile(
        case,
        reference,
        store,
        settings.kite_max_target_units,
        settings.kite_max_unit_chars,
        focal_methods=benchmark_case.focal_methods,
    )
    unit = profile.units[0]
    before = "        return jsonable_encoder(response)"
    anchor = next(
        value
        for value in unit.structural_queries
        if value.start_line <= 65 <= value.end_line and value.node_type == "Return"
    )
    mechanism = TransferableMechanism(
        mechanism_id="verification:null-response",
        corpus_id="verification:train-only",
        source_project="verification-train",
        what="a response serialization contract returns an incorrect value",
        where="a terminal return serializes an untyped response",
        how="replace the terminal serialized value with an observably wrong value",
        transition="the terminal serialization result is discarded",
        violated_property="the response value is preserved by serialization",
        observable="the caller receives None instead of its serialized mapping",
        test_shape=["serialize a mapping and assert the mapping is preserved"],
    )
    match = RetrievalMatch(
        rank=1,
        mechanism=mechanism,
        target_unit=unit,
        semantic_score=1.0,
        structural_score=1.0,
        fused_score=1.0,
    )
    plan = InjectionPlan(
        mechanism_to_reproduce="discard the terminal serialization result",
        target_anchor_id=next(
            anchor_id
            for anchor_id, candidate in available_local_anchors(match)
            if candidate == anchor
        ),
        applicability_reason="the focal method has a terminal response-serialization return",
        before=before,
        after="        return None",
        expected_behavioral_change="mapping serialization returns None",
        trigger_recipe=["serialize a mapping without a response field"],
        mechanism_alignment=[
            MechanismAlignment(
                dimension="what",
                historical_element=mechanism.what,
                target_evidence="serialize_response returns jsonable_encoder(response)",
                edit_effect="the returned response value is discarded",
            ),
            MechanismAlignment(
                dimension="where",
                historical_element=mechanism.where,
                target_evidence="the selected anchor is the focal method's terminal Return",
                edit_effect="the edit remains local to that return",
            ),
            MechanismAlignment(
                dimension="how",
                historical_element=mechanism.how,
                target_evidence="the return value can be locally replaced",
                edit_effect="jsonable_encoder(response) becomes None",
            ),
        ],
        behavioral_probe=(
            "from fastapi.routing import serialize_response\n\n"
            "def test_kite_local_injection_gate():\n"
            "    assert serialize_response(field=None, response={'value': 1}) == {'value': 1}\n"
        ),
    )
    tag = "proofplane-kite-verification-fastapi10:latest"
    try:
        build_workspace_image(reference, tag, str(case.metadata.get("python_version", "3.8")))
        validation = apply_and_validate(
            match=match,
            plan=plan,
            reference=reference,
            synthetic_root=settings.data_dir / "kite-synthetic-verification",
            image=tag,
            settings=settings,
            store=store,
        )
    finally:
        remove_workspace_image(tag)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(validation.model_dump_json(indent=2), encoding="utf-8")
    print(json.dumps({"accepted": validation.accepted, "gates": validation.gates}, indent=2))
    return 0 if validation.accepted and all(validation.gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
