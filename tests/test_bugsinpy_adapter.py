from __future__ import annotations

import subprocess
from pathlib import Path

from proofplane_v0.datasets.base import DatasetCase
from proofplane_v0.datasets.bugsinpy import BugsInPyAdapter


def test_git_materialization_enables_long_paths_and_uses_reset(tmp_path: Path, monkeypatch) -> None:
    bug_dir = tmp_path / "metadata"
    bug_dir.mkdir()
    case = DatasetCase(
        dataset="bugsinpy",
        project="project",
        bug_id="1",
        buggy_source={},
        fixed_source={},
        pre_fix_tests={},
        hidden_tests={},
        metadata={
            "buggy_commit_id": "buggy",
            "fixed_commit_id": "fixed",
            "bug_dir": str(bug_dir),
            "test_files": [],
        },
    )
    calls: list[list[str]] = []

    def fake_run(command: list[str], **kwargs):
        calls.append(command)
        if "clone" in command:
            target = Path(command[-1])
            if "--mirror" in command:
                target.mkdir(parents=True)
            else:
                (target / ".git").mkdir(parents=True)
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr("proofplane_v0.datasets.bugsinpy.subprocess.run", fake_run)
    adapter = BugsInPyAdapter(tmp_path)
    candidate, reference = adapter._materialize_with_git(
        case, tmp_path / "case", "https://example.invalid/project.git"
    )

    clones = [call for call in calls if "clone" in call]
    assert len(clones) == 3
    assert all(call[1:3] == ["-c", "core.longpaths=true"] for call in clones)
    assert sum("--mirror" in call for call in clones) == 1
    assert sum("--shared" in call for call in clones) == 2
    assert any("reset" in call and "buggy" in call for call in calls)
    assert any("reset" in call and "fixed" in call for call in calls)
    assert (candidate / ".proofplane-ready").exists()
    assert (reference / ".proofplane-ready").exists()
