from pathlib import Path
from proofplane_v0.evaluation.docker_runner import docker_command
from proofplane_v0.evaluation.scorer import classify, syntax_valid
from proofplane_v0.models import ExecutionOutcome
from proofplane_v0.evaluation.workspace import _recorded_commands, _visible_command_exists
from proofplane_v0.datasets.base import DatasetCase


def test_candidate_only_scoring():
    assert (
        classify(ExecutionOutcome(status="fail"), ExecutionOutcome(status="pass"))
        == "candidate-only failure"
    )
    assert syntax_valid("def test_x():\n assert True\n")
    assert (
        classify(ExecutionOutcome(status="pass"), ExecutionOutcome(status="fail"))
        == "reference-only failure"
    )


def test_docker_command_is_sandboxed():
    command = docker_command(Path("workspace"), "test_generated.py")
    assert "--network" in command and "none" in command and "--read-only" in command
    assert "--pids-limit" in command


def test_bugsinpy_native_commands_and_missing_prefix_asset(tmp_path: Path):
    case = DatasetCase(
        dataset="bugsinpy",
        project="sample",
        bug_id="1",
        buggy_source={},
        fixed_source={},
        pre_fix_tests={},
        hidden_tests={},
        metadata={
            "test_files": ["tests/test_added.py"],
            "run_test": "# recorded\npython -m unittest -q tests.test_added.T.test_new\n",
        },
    )
    commands = _recorded_commands(case)
    assert commands == ["python -m unittest -q tests.test_added.T.test_new"]
    assert not _visible_command_exists(commands[0], tmp_path, ["tests/test_added.py"])


def test_new_method_in_existing_file_is_not_a_pre_fix_trigger(tmp_path: Path):
    target = tmp_path / "tests/test_existing.py"
    target.parent.mkdir(parents=True)
    target.write_text("def test_old():\n    pass\n", encoding="utf-8")
    command = "pytest tests/test_existing.py::test_new"
    assert not _visible_command_exists(command, tmp_path, ["tests/test_existing.py"])
    target.write_text("def test_new():\n    pass\n", encoding="utf-8")
    assert _visible_command_exists(command, tmp_path, ["tests/test_existing.py"])
