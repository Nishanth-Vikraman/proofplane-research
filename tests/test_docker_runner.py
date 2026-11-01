from __future__ import annotations

import subprocess
from pathlib import Path

from proofplane_v0.evaluation import docker_runner


def test_workspace_image_installs_git_before_vcs_requirements(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / "bugsinpy_requirements.txt").write_bytes(
        "example @ git+https://example.invalid/repository.git\r\n".encode("utf-16")
    )
    (tmp_path / "bugsinpy_bug.info").write_text(
        'pythonpath="example/build/lib/;example/vendor/"\n', encoding="utf-8"
    )
    monkeypatch.setattr(docker_runner, "docker_executable", lambda: "docker")
    commands: list[list[str]] = []
    monkeypatch.setattr(
        docker_runner.subprocess,
        "run",
        lambda command, **kwargs: commands.append(command)
        or subprocess.CompletedProcess(command, 0, "", ""),
    )

    docker_runner.build_workspace_image(tmp_path, "proofplane-test:latest", "3.8")

    dockerfile = (tmp_path / ".proofplane.Dockerfile").read_text(encoding="utf-8")
    git_install = "apt-get install -y --no-install-recommends git"
    requirements_install = "python /tmp/proofplane_install_requirements.py"
    assert git_install in dockerfile
    assert dockerfile.index(git_install) < dockerfile.index(requirements_install)
    assert dockerfile.count(requirements_install) == 2
    assert "pip install --no-cache-dir pytest" not in dockerfile
    assert "pip install --no-cache-dir --no-build-isolation -e ." not in dockerfile
    assert dockerfile.index(requirements_install) < dockerfile.index("COPY . /workspace")
    assert 'ENV PYTHONPATH="/workspace/build/lib:/workspace/vendor:$PYTHONPATH"' in dockerfile
    normalized = (tmp_path / ".proofplane.requirements.utf8.txt").read_bytes()
    assert not normalized.startswith((b"\xff\xfe", b"\xfe\xff"))
    assert b"\x00" not in normalized
    assert normalized.decode("utf-8") == "example @ git+https://example.invalid/repository.git\n"
    assert commands == [
        [
            "docker",
            "buildx",
            "build",
            "--load",
            "--file",
            str(tmp_path / ".proofplane.Dockerfile"),
            "--tag",
            "proofplane-test:latest",
            str(tmp_path),
        ]
    ]


def test_remove_workspace_image_is_best_effort(monkeypatch) -> None:
    calls: list[list[str]] = []
    monkeypatch.setattr(docker_runner, "docker_executable", lambda: "docker")
    monkeypatch.setattr(
        docker_runner.subprocess,
        "run",
        lambda command, **kwargs: calls.append(command)
        or subprocess.CompletedProcess(command, 1, "", "storage unavailable"),
    )

    docker_runner.remove_workspace_image("temporary-validation:latest")

    assert calls == [
        ["docker", "image", "rm", "--force", "temporary-validation:latest"]
    ]


def test_native_runner_uses_recorded_command_without_read_only_mount(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(docker_runner, "docker_executable", lambda: "docker")
    commands: list[list[str]] = []
    monkeypatch.setattr(
        docker_runner.subprocess,
        "run",
        lambda command, **kwargs: commands.append(command)
        or subprocess.CompletedProcess(command, 0, "ok", ""),
    )

    outcome = docker_runner.run_native_test(
        tmp_path, "python -m unittest -q tests.test_x.TestX.test_it", image="case:image"
    )

    assert outcome.status == "pass"
    assert commands[0][-3:] == ["sh", "-lc", "python -m unittest -q tests.test_x.TestX.test_it"]
    assert "--read-only" not in commands[0]
    assert "-v" not in commands[0]
