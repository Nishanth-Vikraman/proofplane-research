from __future__ import annotations

import random
from proofplane_v0.datasets.base import DatasetCase


def project_split(cases: list[DatasetCase], seed: int = 17) -> dict[str, str]:
    projects = sorted({case.project for case in cases})
    random.Random(seed).shuffle(projects)
    train_end = max(1, int(len(projects) * 0.6))
    dev_end = max(train_end, int(len(projects) * 0.8))
    mapping = {p: "train" for p in projects[:train_end]}
    mapping.update({p: "dev" for p in projects[train_end:dev_end]})
    mapping.update({p: "eval" for p in projects[dev_end:]})
    return {case.case_id: mapping[case.project] for case in cases}


def assert_isolated(cases: list[DatasetCase], splits: dict[str, str]) -> None:
    by_split: dict[str, set[str]] = {}
    for case in cases:
        by_split.setdefault(splits[case.case_id], set()).add(case.project)
    names = list(by_split)
    for i, name in enumerate(names):
        for other in names[i + 1 :]:
            if by_split[name] & by_split[other]:
                raise AssertionError("project leakage across splits")
