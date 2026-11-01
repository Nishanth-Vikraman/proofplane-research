from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetCase:
    dataset: str
    project: str
    bug_id: str
    buggy_source: dict[str, str]
    fixed_source: dict[str, str]
    pre_fix_tests: dict[str, str]
    hidden_tests: dict[str, str]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def case_id(self) -> str:
        return f"{self.dataset}:{self.project}:{self.bug_id}"


class DatasetAdapter(ABC):
    @abstractmethod
    def list_cases(self) -> list[DatasetCase]: ...

    def get_case(self, case_id: str) -> DatasetCase:
        return next(case for case in self.list_cases() if case.case_id == case_id)

    @abstractmethod
    def materialize(self, case: DatasetCase, root: Path) -> tuple[Path, Path]: ...
