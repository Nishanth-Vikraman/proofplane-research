from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from .base import DatasetAdapter, DatasetCase


class MaterializationInfrastructureError(RuntimeError):
    """A checkout/setup failure that must not be treated as a dataset exclusion."""


def _parse_info(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = re.match(r"\s*([A-Za-z_]+)\s*[=:]\s*['\"]?(.*?)['\"]?\s*$", line)
        if match:
            values[match.group(1)] = match.group(2)
    return values


class BugsInPyAdapter(DatasetAdapter):
    """Reads the upstream projects/<project>/bugs/<id> contract without guessing paths."""

    def __init__(self, root: Path):
        self.root = root

    def list_cases(self) -> list[DatasetCase]:
        cases: list[DatasetCase] = []
        for bug_dir in sorted(self.root.glob("projects/*/bugs/*")):
            info_path = bug_dir / "bug.info"
            project_info = bug_dir.parents[1] / "project.info"
            if not info_path.exists() or not project_info.exists():
                continue
            info = _parse_info(info_path)
            tests = [x.strip() for x in re.split(r"[,;]", info.get("test_file", "")) if x.strip()]
            run_test_path = bug_dir / "run_test.sh"
            cases.append(
                DatasetCase(
                    dataset="bugsinpy",
                    project=bug_dir.parents[1].name,
                    bug_id=bug_dir.name,
                    buggy_source={},
                    fixed_source={},
                    pre_fix_tests={},
                    hidden_tests={},
                    metadata={
                        **info,
                        "test_files": tests,
                        "bug_dir": str(bug_dir),
                        "run_test": run_test_path.read_text(encoding="utf-8", errors="replace")
                        if run_test_path.exists()
                        else "",
                        "project_info": _parse_info(project_info),
                    },
                )
            )
        if not cases:
            for manifest in sorted((self.root / "manifests").glob("*.json")):
                raw = json.loads(manifest.read_text(encoding="utf-8"))
                cases.append(
                    DatasetCase(
                        dataset="bugsinpy",
                        project=str(raw["project"]),
                        bug_id=str(raw["bug_id"]),
                        buggy_source=raw.get("buggy_source", {}),
                        fixed_source=raw.get("fixed_source", {}),
                        pre_fix_tests=raw.get("pre_fix_tests", {}),
                        hidden_tests=raw.get("hidden_tests", {}),
                        metadata={**raw.get("metadata", {}), "manifest": str(manifest)},
                    )
                )
        return cases

    def materialize(self, case: DatasetCase, root: Path) -> tuple[Path, Path]:
        candidate, reference = root / "candidate", root / "reference"
        if (candidate / ".proofplane-ready").exists() and (
            reference / ".proofplane-ready"
        ).exists():
            return candidate, reference
        github_url = case.metadata.get("project_info", {}).get("github_url")
        if github_url:
            return self._materialize_with_git(case, root, str(github_url))
        checkout = shutil.which("bugsinpy-checkout")
        if not checkout:
            source_candidate = case.metadata.get("candidate_workspace")
            source_reference = case.metadata.get("reference_workspace")
            if source_candidate and source_reference:
                shutil.copytree(Path(source_candidate), candidate, dirs_exist_ok=True)
                shutil.copytree(Path(source_reference), reference, dirs_exist_ok=True)
                return candidate, reference
            if case.buggy_source and case.fixed_source:
                for workspace, source in (
                    (candidate, case.buggy_source),
                    (reference, case.fixed_source),
                ):
                    workspace.mkdir(parents=True, exist_ok=True)
                    for name, content in {**source, **case.pre_fix_tests}.items():
                        path = workspace / name
                        path.parent.mkdir(parents=True, exist_ok=True)
                        path.write_text(content, encoding="utf-8")
                return candidate, reference
            raise RuntimeError(
                "BugsInPy CLI unavailable; manifest must contain source maps or candidate/reference workspaces"
            )
        for version, workspace in (("0", candidate), ("1", reference)):
            subprocess.run(
                [
                    checkout,
                    "-p",
                    case.project,
                    "-i",
                    case.bug_id,
                    "-v",
                    version,
                    "-w",
                    str(workspace),
                ],
                check=True,
            )
        return candidate, reference

    def _materialize_with_git(
        self, case: DatasetCase, root: Path, github_url: str
    ) -> tuple[Path, Path]:
        candidate, reference = root / "candidate", root / "reference"
        root.mkdir(parents=True, exist_ok=True)
        buggy = str(case.metadata["buggy_commit_id"])
        fixed = str(case.metadata["fixed_commit_id"])
        cache = root.parent.parent / "git-cache" / f"{self._safe_project(case.project)}.git"
        self._ensure_project_cache(case, cache, github_url, (buggy, fixed))
        for workspace, commit in ((candidate, buggy), (reference, fixed)):
            try:
                if workspace.exists() and not (workspace / ".git").exists():
                    # A failed Windows checkout commonly leaves a non-reusable partial tree.
                    self._remove_partial_workspace(workspace)
                if not (workspace / ".git").exists():
                    subprocess.run(
                        [
                            "git",
                            "-c",
                            "core.longpaths=true",
                            "clone",
                            "--no-checkout",
                            "--shared",
                            str(cache),
                            str(workspace),
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                subprocess.run(
                    ["git", "-C", str(workspace), "config", "core.longpaths", "true"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                present = subprocess.run(
                    ["git", "-C", str(workspace), "cat-file", "-e", f"{commit}^{{commit}}"],
                    capture_output=True,
                )
                if present.returncode:
                    subprocess.run(
                        ["git", "-C", str(workspace), "fetch", str(cache), commit],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                subprocess.run(
                    ["git", "-C", str(workspace), "reset", "--hard", commit],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                subprocess.run(
                    ["git", "-C", str(workspace), "clean", "-ffd"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except (OSError, subprocess.CalledProcessError) as error:
                stderr = getattr(error, "stderr", "") or ""
                raise MaterializationInfrastructureError(
                    f"Git materialization failed for {case.case_id} at {workspace}: "
                    f"{type(error).__name__}: {error}; {stderr[-2000:]}"
                ) from error

        self.refresh_test_separation(case, candidate, reference)
        bug_dir = Path(str(case.metadata["bug_dir"]))
        for workspace in (candidate, reference):
            for source_name, target_name in (
                ("requirements.txt", "bugsinpy_requirements.txt"),
                ("run_test.sh", "bugsinpy_run_test.sh"),
                ("setup.sh", "bugsinpy_setup.sh"),
                ("bug.info", "bugsinpy_bug.info"),
            ):
                source = bug_dir / source_name
                if source.exists():
                    shutil.copy2(source, workspace / target_name)
            (workspace / ".proofplane-ready").write_text(case.case_id, encoding="utf-8")
        return candidate, reference

    @staticmethod
    def _safe_project(project: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]", "-", project)

    @staticmethod
    def _remove_partial_workspace(workspace: Path) -> None:
        def writable_then_retry(function, path, _error_info) -> None:
            os.chmod(path, 0o700)
            function(path)

        shutil.rmtree(workspace, onerror=writable_then_retry)

    def _ensure_project_cache(
        self,
        case: DatasetCase,
        cache: Path,
        github_url: str,
        commits: tuple[str, str],
    ) -> None:
        try:
            if cache.exists():
                valid = subprocess.run(
                    ["git", "--git-dir", str(cache), "rev-parse", "--is-bare-repository"],
                    capture_output=True,
                    text=True,
                )
                if valid.returncode or valid.stdout.strip() != "true":
                    self._remove_partial_workspace(cache)
            if not cache.exists():
                cache.parent.mkdir(parents=True, exist_ok=True)
                local_seed = next(
                    (
                        candidate
                        for candidate in cache.parent.parent.joinpath("workspaces").glob(
                            f"bugsinpy-{self._safe_project(case.project)}-*/candidate"
                        )
                        if (candidate / ".git").exists()
                        and (candidate / ".proofplane-ready").exists()
                    ),
                    None,
                )
                subprocess.run(
                    [
                        "git",
                        "-c",
                        "core.longpaths=true",
                        "clone",
                        "--mirror",
                        str(local_seed) if local_seed else github_url,
                        str(cache),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                if local_seed:
                    subprocess.run(
                        [
                            "git",
                            "--git-dir",
                            str(cache),
                            "remote",
                            "set-url",
                            "origin",
                            github_url,
                        ],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
            subprocess.run(
                ["git", "--git-dir", str(cache), "config", "core.longpaths", "true"],
                check=True,
                capture_output=True,
                text=True,
            )
            missing = [
                commit
                for commit in commits
                if subprocess.run(
                    ["git", "--git-dir", str(cache), "cat-file", "-e", f"{commit}^{{commit}}"],
                    capture_output=True,
                ).returncode
            ]
            if missing:
                subprocess.run(
                    ["git", "--git-dir", str(cache), "fetch", "--prune", "origin"],
                    check=True,
                    capture_output=True,
                    text=True,
                )
            for commit in missing:
                present = subprocess.run(
                    ["git", "--git-dir", str(cache), "cat-file", "-e", f"{commit}^{{commit}}"],
                    capture_output=True,
                )
                if present.returncode:
                    raise RuntimeError(f"commit {commit} is absent after refreshing the project mirror")
        except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
            stderr = getattr(error, "stderr", "") or ""
            raise MaterializationInfrastructureError(
                f"Git project cache failed for {case.case_id} at {cache}: "
                f"{type(error).__name__}: {error}; {stderr[-2000:]}"
            ) from error

    def refresh_test_separation(self, case: DatasetCase, candidate: Path, reference: Path) -> None:
        """Restore fixed oracle tests separately and buggy/pre-fix tests to both workspaces."""
        root = candidate.parent
        oracle = root / "oracle_tests"
        oracle.mkdir(exist_ok=True)
        buggy = str(case.metadata["buggy_commit_id"])
        fixed = str(case.metadata["fixed_commit_id"])
        for test_name in case.metadata.get("test_files", []):
            fixed_test = subprocess.run(
                ["git", "-c", "core.longpaths=true", "-C", str(reference), "show", f"{fixed}:{test_name}"],
                capture_output=True,
            )
            if fixed_test.returncode == 0:
                oracle_path = oracle / test_name
                oracle_path.parent.mkdir(parents=True, exist_ok=True)
                oracle_path.write_bytes(fixed_test.stdout)
            buggy_test = subprocess.run(
                ["git", "-c", "core.longpaths=true", "-C", str(candidate), "show", f"{buggy}:{test_name}"],
                capture_output=True,
            )
            for workspace in (candidate, reference):
                target = workspace / test_name
                if buggy_test.returncode == 0:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(buggy_test.stdout)
                elif target.exists():
                    target.unlink()
