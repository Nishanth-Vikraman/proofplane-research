from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from proofplane_v0.evaluation.docker_runner import docker_executable


def run(command: list[str], timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        check=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require-image", default="python:3.8-slim")
    args = parser.parse_args()
    docker = docker_executable()
    report: dict[str, object] = {
        "schema_version": "docker-readiness-v1",
        "docker_executable": docker or "",
        "required_image": args.require_image,
        "ready": False,
    }
    if not docker:
        report["error"] = "docker CLI is unavailable"
    else:
        try:
            info = run([docker, "info", "--format", "{{json .}}"], 30)
            report["info_exit_code"] = info.returncode
            if info.returncode:
                report["error"] = (info.stderr or info.stdout).strip()
            else:
                server = json.loads(info.stdout)
                report["server_version"] = server.get("ServerVersion", "")
                report["operating_system"] = server.get("OperatingSystem", "")
                report["os_type"] = server.get("OSType", "")
                if server.get("OSType") != "linux":
                    report["error"] = "Docker server is not using Linux containers"
                else:
                    image = run([docker, "image", "inspect", args.require_image], 30)
                    report["required_image_present"] = image.returncode == 0
                    if image.returncode:
                        report["error"] = f"required base image is not local: {args.require_image}"
                    else:
                        report["ready"] = True
        except (subprocess.TimeoutExpired, json.JSONDecodeError) as error:
            report["error"] = f"{type(error).__name__}: {error}"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
