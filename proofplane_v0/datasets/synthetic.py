from __future__ import annotations

from pathlib import Path

from .base import DatasetAdapter, DatasetCase

CASES = [
    DatasetCase(
        dataset="synthetic",
        project="cachebox",
        bug_id="1",
        fixed_source={
            "cachebox.py": "class Box:\n    def __init__(self): self.items=[]; self.count=0\n    def add(self, x): self.items.append(x); self.count=len(self.items)\n    def size(self): return self.count\n"
        },
        buggy_source={
            "cachebox.py": "class Box:\n    def __init__(self): self.items=[]; self.count=0\n    def add(self, x): self.items.append(x)\n    def size(self): return self.count\n"
        },
        pre_fix_tests={
            "test_existing.py": "from cachebox import Box\ndef test_empty(): assert Box().size() == 0\n"
        },
        hidden_tests={
            "test_regression.py": "from cachebox import Box\ndef test_add_updates_size():\n b=Box(); b.add(1); assert b.size()==1\n"
        },
        metadata={
            "family": "stale-derived-state",
            "primitives": ["mutation", "derived-read"],
            "summary": "A mutation updates authoritative state but leaves a derived read path stale.",
            "strategy": ["write then read through derived accessor"],
        },
    ),
    DatasetCase(
        dataset="synthetic",
        project="threshold",
        bug_id="1",
        fixed_source={"threshold.py": "def allowed(value, limit):\n    return value <= limit\n"},
        buggy_source={"threshold.py": "def allowed(value, limit):\n    return value < limit\n"},
        pre_fix_tests={
            "test_existing.py": "from threshold import allowed\ndef test_inside(): assert allowed(2,3)\n"
        },
        hidden_tests={
            "test_regression.py": "from threshold import allowed\ndef test_equal_boundary(): assert allowed(3,3)\n"
        },
        metadata={
            "family": "equality-boundary",
            "primitives": ["comparison", "boundary"],
            "summary": "A strict comparison rejects the equality boundary promised by the interface.",
            "strategy": ["exercise values below, at, and above a boundary"],
        },
    ),
    DatasetCase(
        dataset="synthetic",
        project="ledger",
        bug_id="1",
        fixed_source={
            "ledger.py": "class Ledger:\n def __init__(self): self.rows=[]\n def add_pair(self,a,b):\n  if b < 0: raise ValueError('negative')\n  self.rows.extend([a,b])\n"
        },
        buggy_source={
            "ledger.py": "class Ledger:\n def __init__(self): self.rows=[]\n def add_pair(self,a,b):\n  self.rows.append(a)\n  if b < 0: raise ValueError('negative')\n  self.rows.append(b)\n"
        },
        pre_fix_tests={
            "test_existing.py": "from ledger import Ledger\ndef test_valid():\n x=Ledger(); x.add_pair(1,2); assert x.rows==[1,2]\n"
        },
        hidden_tests={
            "test_regression.py": "import pytest\nfrom ledger import Ledger\ndef test_atomic_failure():\n x=Ledger()\n with pytest.raises(ValueError): x.add_pair(1,-1)\n assert x.rows==[]\n"
        },
        metadata={
            "family": "partial-mutation",
            "primitives": ["exception", "mutation", "atomicity"],
            "summary": "An operation mutates state before validation, leaving partial state after an exception.",
            "strategy": ["force an exception and verify state remains unchanged"],
        },
    ),
]


class SyntheticAdapter(DatasetAdapter):
    def list_cases(self) -> list[DatasetCase]:
        return CASES

    def materialize(self, case: DatasetCase, root: Path) -> tuple[Path, Path]:
        candidate, reference = root / "candidate", root / "reference"
        for workspace, source in ((candidate, case.buggy_source), (reference, case.fixed_source)):
            workspace.mkdir(parents=True, exist_ok=True)
            for name, content in {**source, **case.pre_fix_tests}.items():
                path = workspace / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")
        return candidate, reference
