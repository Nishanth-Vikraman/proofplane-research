from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

from proofplane_v0.config import Settings
from proofplane_v0.context.bundle import build_bundle
from proofplane_v0.datasets.base import DatasetCase
from proofplane_v0.evaluation.docker_runner import (
    EnvironmentBuildError,
    build_workspace_image,
    remove_workspace_image,
    run_native_test,
)

VALIDATOR_VERSION = "bugsinpy-counterfactual-v2"


def _recorded_commands(case: DatasetCase) -> list[str]:
    return [
        line.strip()
        for line in str(case.metadata.get("run_test", "")).splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _visible_command_exists(command: str, workspace: Path, test_files: list[str]) -> bool:
    existing = [workspace / name for name in test_files if (workspace / name).is_file()]
    if not existing:
        return False
    selector = ""
    if "::" in command:
        selector = command.rsplit("::", 1)[-1].split()[0]
    elif "unittest" in command:
        values = [part for part in command.split() if ".test" in part]
        if values:
            selector = values[-1].rsplit(".", 1)[-1]
    if not selector:
        return True
    pattern = re.compile(rf"\b(?:async\s+)?def\s+{re.escape(selector)}\s*\(")
    return any(
        pattern.search(path.read_text(encoding="utf-8", errors="replace"))
        for path in existing
    )


def validate_counterfactual(case: DatasetCase, settings: Settings) -> dict[str, object]:
    root = settings.data_dir / "workspaces" / case.case_id.replace(":", "-")
    candidate, reference, oracle = root / "candidate", root / "reference", root / "oracle_tests"
    bundle = build_bundle(
        case.case_id,
        reference,
        candidate,
        settings.max_context_chars,
        set(),
        settings.max_nearby_tests,
        settings.max_test_snippet_chars,
        settings.max_repository_tree_entries,
    )
    test_files = list(case.metadata.get("test_files", []))
    result: dict[str, object] = {
        "validator_version": VALIDATOR_VERSION,
        "case_id": case.case_id,
        "production_diff_present": bool(bundle.changed_files),
        "visible_tests_identical": all(
            (candidate / name).exists() == (reference / name).exists()
            and (
                not (candidate / name).exists()
                or (candidate / name).read_bytes() == (reference / name).read_bytes()
            )
            for name in test_files
        ),
        "hidden_tests_absent": not (candidate / "oracle_tests").exists()
        and not (reference / "oracle_tests").exists(),
    }
    python_version = str(case.metadata.get("python_version", "3.12"))
    base_tag = re.sub(r"[^a-z0-9_.-]", "-", case.case_id.lower())
    candidate_image = f"proofplane-validate-{base_tag}-candidate:latest"
    reference_image = f"proofplane-validate-{base_tag}-reference:latest"
    oracle_candidate = f"proofplane-validate-{base_tag}-oracle-candidate:latest"
    oracle_reference = f"proofplane-validate-{base_tag}-oracle-reference:latest"
    images = (candidate_image, reference_image, oracle_candidate, oracle_reference)
    try:
        commands = _recorded_commands(case)
        visible_commands = [
            command
            for command in commands
            if _visible_command_exists(command, candidate, test_files)
        ]
        visible_pass = True
        if visible_commands:
            build_workspace_image(candidate, candidate_image, python_version)
            build_workspace_image(reference, reference_image, python_version)
            native_visible = "set -e\n" + "\n".join(visible_commands)
            visible_candidate = run_native_test(
                candidate, native_visible, settings.execution_timeout, image=candidate_image
            )
            visible_reference = run_native_test(
                reference, native_visible, settings.execution_timeout, image=reference_image
            )
            result["visible_asset_status"] = "present"
            result["visible_commands"] = visible_commands
            result["visible_candidate"] = visible_candidate.model_dump(mode="json")
            result["visible_reference"] = visible_reference.model_dump(mode="json")
            visible_pass = (
                visible_candidate.status == "pass" and visible_reference.status == "pass"
            )
        else:
            result["visible_asset_status"] = "absent_pre_fix"
            result["visible_commands"] = []

        backups: dict[tuple[Path, str], bytes | None] = {}
        try:
            missing_oracle = [name for name in test_files if not (oracle / name).is_file()]
            if missing_oracle:
                raise EnvironmentBuildError(
                    "fixed BugsInPy trigger asset could not be materialized: "
                    + ", ".join(missing_oracle)
                )
            for workspace in (candidate, reference):
                for name in test_files:
                    target, source = workspace / name, oracle / name
                    backups[(workspace, name)] = target.read_bytes() if target.exists() else None
                    if source.exists():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, target)
            build_workspace_image(candidate, oracle_candidate, python_version)
            build_workspace_image(reference, oracle_reference, python_version)
            if not commands:
                raise RuntimeError("BugsInPy case has no recorded run_test command")
            native_trigger = "set -e\n" + "\n".join(commands)
            trigger_candidate = run_native_test(
                candidate, native_trigger, settings.execution_timeout, image=oracle_candidate
            )
            trigger_reference = run_native_test(
                reference, native_trigger, settings.execution_timeout, image=oracle_reference
            )
            result["trigger_candidate"] = trigger_candidate.model_dump(mode="json")
            result["trigger_reference"] = trigger_reference.model_dump(mode="json")
            result["valid"] = bool(
                result["production_diff_present"]
                and result["visible_tests_identical"]
                and result["hidden_tests_absent"]
                and visible_pass
                and trigger_candidate.status == "fail"
                and trigger_reference.status == "pass"
            )
        finally:
            for (workspace, name), content in backups.items():
                target = workspace / name
                if content is None:
                    if target.exists():
                        target.unlink()
                else:
                    target.write_bytes(content)
        output = settings.artifacts_dir / "validations" / f"{case.case_id.replace(':', '-')}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result
    finally:
        for image in images:
            remove_workspace_image(image)
