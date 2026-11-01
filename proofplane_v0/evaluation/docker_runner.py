from __future__ import annotations

import shutil
import subprocess
import sys
import time
import os
import re
from pathlib import Path
from typing import Literal

from proofplane_v0.models import ExecutionOutcome


class ValidationInfrastructureError(RuntimeError):
    """The host/container service failed; this says nothing about a benchmark case."""


class EnvironmentBuildError(RuntimeError):
    """The historical case environment could not be reproduced by the validator."""


_INFRASTRUCTURE_MARKERS = (
    "dockerdesktop/wsl/execerror",
    "failed to connect to the docker api",
    "cannot connect to the docker daemon",
    "is the docker daemon running",
    "running wslexec",
    "no such host",
    "temporary failure in name resolution",
    "network is unreachable",
    "tls handshake timeout",
    "i/o timeout",
    "no space left on device",
    "read-only file system",
    "/var/lib/docker/buildkit/",
    "/var/lib/desktop-containerd/",
)

_REQUIREMENTS_INSTALLER = """from __future__ import annotations

import shlex
import subprocess
import sys
from pathlib import Path

requirements = Path(sys.argv[1])
failures = 0
for raw_line in requirements.read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if not line or line.startswith("#"):
        continue
    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-cache-dir",
        "--use-deprecated=legacy-resolver",
        *shlex.split(line),
    ]
    result = subprocess.run(command, check=False)
    if result.returncode:
        failures += 1
        print(
            f"[proofplane:pip-failure] exit={result.returncode} requirement={line}",
            file=sys.stderr,
            flush=True,
        )
print(f"[proofplane:pip-summary] failures={failures}", file=sys.stderr, flush=True)
"""


def is_infrastructure_failure(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in _INFRASTRUCTURE_MARKERS)


def docker_executable() -> str | None:
    configured = os.getenv("DOCKER_EXE")
    if configured:
        try:
            if Path(configured).exists():
                return configured
        except OSError:
            pass
    found = shutil.which("docker")
    if found:
        return found
    for path in (
        Path(r"C:\Program Files\Docker\Docker\resources\bin\docker.exe"),
        Path.home() / "AppData/Local/Programs/DockerDesktop/resources/bin/docker.exe",
        Path.home() / "AppData/Local/Docker/wsl/bin/docker.exe",
    ):
        try:
            if path.exists():
                return str(path)
        except OSError:
            continue
    return None


def docker_command(
    workspace: Path,
    test_path: str | list[str],
    timeout: int = 30,
    image: str = "proofplane-runner:py312",
) -> list[str]:
    return [
        docker_executable() or "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--cpus",
        "1",
        "--memory",
        "512m",
        "--pids-limit",
        "128",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=64m",
        "-v",
        f"{workspace.resolve()}:/workspace:ro",
        "-w",
        "/workspace",
        image,
        "python",
        "-m",
        "pytest",
        "-q",
        *([test_path] if isinstance(test_path, str) else test_path),
    ]


def run_test(
    workspace: Path,
    test_path: str | list[str],
    timeout: int = 30,
    unsafe_local_execution: bool = False,
    image: str = "proofplane-runner:py312",
) -> ExecutionOutcome:
    if not unsafe_local_execution:
        if not docker_executable():
            return ExecutionOutcome(
                status="error", stderr="Docker unavailable; local fallback not enabled"
            )
        command = docker_command(workspace, test_path, timeout, image)
    else:
        arguments = [test_path] if isinstance(test_path, str) else test_path
        command = [sys.executable, "-m", "pytest", "-q", *arguments]
    started = time.monotonic()
    try:
        result = subprocess.run(
            command,
            cwd=workspace if unsafe_local_execution else None,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        stdout = (
            error.stdout.decode(errors="replace")
            if isinstance(error.stdout, bytes)
            else (error.stdout or "")
        )
        stderr = (
            error.stderr.decode(errors="replace")
            if isinstance(error.stderr, bytes)
            else (error.stderr or "")
        )
        return ExecutionOutcome(
            status="timeout",
            duration_seconds=time.monotonic() - started,
            stdout=stdout,
            stderr=stderr,
        )
    status: Literal["pass", "fail", "import_setup"] = "pass" if result.returncode == 0 else "fail"
    if (
        "ImportError" in result.stdout + result.stderr
        or "ModuleNotFoundError" in result.stdout + result.stderr
    ):
        status = "import_setup"
    return ExecutionOutcome(
        status=status,
        return_code=result.returncode,
        duration_seconds=time.monotonic() - started,
        stdout=result.stdout[-5000:],
        stderr=result.stderr[-5000:],
    )


def run_workspace_command(
    workspace: Path,
    arguments: list[str],
    timeout: int = 30,
    image: str = "proofplane-runner:py312",
) -> ExecutionOutcome:
    """Run a bounded command against a mounted workspace using the normal Docker sandbox."""
    docker = docker_executable()
    if not docker:
        return ExecutionOutcome(status="error", stderr="Docker unavailable")
    command = [
        docker,
        "run",
        "--rm",
        "--network",
        "none",
        "--cpus",
        "1",
        "--memory",
        "512m",
        "--pids-limit",
        "128",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,noexec,nosuid,size=64m",
        "-v",
        f"{workspace.resolve()}:/workspace:ro",
        "-w",
        "/workspace",
        image,
        *arguments,
    ]
    started = time.monotonic()
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        return ExecutionOutcome(
            status="timeout",
            duration_seconds=time.monotonic() - started,
            stdout=(error.stdout or "") if isinstance(error.stdout, str) else "",
            stderr=(error.stderr or "") if isinstance(error.stderr, str) else "",
        )
    combined = result.stdout + result.stderr
    if is_infrastructure_failure(combined):
        raise ValidationInfrastructureError(combined[-5000:])
    status: Literal["pass", "fail", "import_setup"] = (
        "pass" if result.returncode == 0 else "fail"
    )
    if "ImportError" in combined or "ModuleNotFoundError" in combined:
        status = "import_setup"
    return ExecutionOutcome(
        status=status,
        return_code=result.returncode,
        duration_seconds=time.monotonic() - started,
        stdout=result.stdout[-5000:],
        stderr=result.stderr[-5000:],
    )


def run_native_test(
    workspace: Path,
    command_text: str,
    timeout: int = 30,
    image: str = "proofplane-runner:py312",
) -> ExecutionOutcome:
    """Run the benchmark's recorded command instead of coercing it into pytest."""
    if not docker_executable():
        raise ValidationInfrastructureError(
            "Docker unavailable; native BugsInPy validation requires Docker"
        )
    command = [
        docker_executable() or "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--cpus",
        "1",
        "--memory",
        "512m",
        "--pids-limit",
        "128",
        "--tmpfs",
        "/tmp:rw,nosuid,size=64m",
        "-w",
        "/workspace",
        image,
        "sh",
        "-lc",
        command_text,
    ]
    started = time.monotonic()
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except OSError as error:
        raise ValidationInfrastructureError(f"Docker execution failed: {error}") from error
    except subprocess.TimeoutExpired as error:
        return ExecutionOutcome(
            status="timeout",
            duration_seconds=time.monotonic() - started,
            stdout=(error.stdout or "") if isinstance(error.stdout, str) else "",
            stderr=(error.stderr or "") if isinstance(error.stderr, str) else "",
        )
    combined = result.stdout + result.stderr
    if is_infrastructure_failure(combined):
        raise ValidationInfrastructureError(combined[-5000:])
    status: Literal["pass", "fail", "import_setup"] = (
        "pass" if result.returncode == 0 else "fail"
    )
    if "ImportError" in combined or "ModuleNotFoundError" in combined:
        status = "import_setup"
    return ExecutionOutcome(
        status=status,
        return_code=result.returncode,
        duration_seconds=time.monotonic() - started,
        stdout=result.stdout[-5000:],
        stderr=result.stderr[-5000:],
    )


def build_workspace_image(
    workspace: Path, tag: str, python_version: str = "3.12", timeout: int = 900
) -> None:
    docker = docker_executable()
    if not docker:
        raise RuntimeError(
            "Docker CLI is not visible. Restart the terminal/Codex after Docker installation or set DOCKER_EXE."
        )
    version = ".".join(python_version.split(".")[:2])
    dockerfile = workspace / ".proofplane.Dockerfile"
    lines = [
        f"FROM python:{version}-slim",
        "WORKDIR /workspace",
        "RUN apt-get update && apt-get install -y --no-install-recommends git "
        "&& rm -rf /var/lib/apt/lists/*",
    ]
    info_path = workspace / "bugsinpy_bug.info"
    if info_path.exists():
        match = re.search(
            r'^pythonpath\s*=\s*["\'](.*?)["\']\s*$',
            info_path.read_text(encoding="utf-8", errors="replace"),
            flags=re.MULTILINE,
        )
        if match and match.group(1).strip():
            python_paths = []
            for value in match.group(1).split(";"):
                parts = Path(value.strip().replace("\\", "/")).parts
                relative = Path(*parts[1:]).as_posix() if len(parts) > 1 else ""
                python_paths.append("/workspace" + (f"/{relative}" if relative else ""))
            lines.append(f'ENV PYTHONPATH="{":".join(python_paths)}:$PYTHONPATH"')
    if (workspace / "bugsinpy_requirements.txt").exists():
        requirements_source = workspace / "bugsinpy_requirements.txt"
        requirements_bytes = requirements_source.read_bytes()
        if requirements_bytes.startswith((b"\xff\xfe", b"\xfe\xff")):
            requirements_text = requirements_bytes.decode("utf-16")
        else:
            requirements_text = requirements_bytes.decode("utf-8-sig", errors="replace")
        normalized_requirements = workspace / ".proofplane.requirements.utf8.txt"
        normalized_requirements.write_text(
            requirements_text.replace("\r\n", "\n").replace("\r", "\n"),
            encoding="utf-8",
            newline="\n",
        )
        installer = workspace / ".proofplane_install_requirements.py"
        installer.write_text(_REQUIREMENTS_INSTALLER, encoding="utf-8", newline="\n")
        lines.extend(
            [
                "COPY .proofplane.requirements.utf8.txt /tmp/bugsinpy_requirements.txt",
                "COPY .proofplane_install_requirements.py /tmp/proofplane_install_requirements.py",
                "RUN python /tmp/proofplane_install_requirements.py /tmp/bugsinpy_requirements.txt",
            ]
        )
    lines.append("COPY . /workspace")
    lines.append(
        "RUN for f in bugsinpy_setup.sh bugsinpy_run_test.sh; do "
        "if [ -f \"$f\" ]; then sed -i 's/\\r$//' \"$f\"; fi; done"
    )
    if (workspace / "bugsinpy_setup.sh").exists():
        lines.append("RUN sh bugsinpy_setup.sh")
    if (workspace / "bugsinpy_requirements.txt").exists():
        lines.append(
            "RUN python /tmp/proofplane_install_requirements.py /tmp/bugsinpy_requirements.txt"
        )
    lines.append(
        "RUN python -c 'import pytest' 2>/dev/null || "
        "python -m pip install --no-cache-dir 'pytest<6'"
    )
    dockerfile.write_text("\n".join(lines) + "\n", encoding="utf-8")
    command = [
        docker,
        "buildx",
        "build",
        "--load",
        "--file",
        str(dockerfile),
        "--tag",
        tag,
        str(workspace),
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValidationInfrastructureError(f"Docker image build did not complete: {error}") from error
    if result.returncode:
        message = (
            f"Docker image build failed ({command!r}):\n"
            f"{result.stdout[-3000:]}\n{result.stderr[-3000:]}"
        )
        if is_infrastructure_failure(message):
            raise ValidationInfrastructureError(message)
        raise EnvironmentBuildError(message)
    pip_failures = [
        line
        for line in (result.stdout + result.stderr).splitlines()
        if "[proofplane:pip-failure]" in line
    ]
    if pip_failures:
        print(
            f"Non-blocking pip failures while building {tag}:\n" + "\n".join(pip_failures),
            file=sys.stderr,
            flush=True,
        )


def remove_workspace_image(tag: str) -> None:
    """Best-effort cleanup for per-case images that are never reused after a run."""
    docker = docker_executable()
    if not docker:
        return
    subprocess.run(
        [docker, "image", "rm", "--force", tag],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
