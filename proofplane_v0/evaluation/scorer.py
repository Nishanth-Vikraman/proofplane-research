from __future__ import annotations

import ast

from proofplane_v0.models import ExecutionOutcome


def syntax_valid(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def classify(candidate: ExecutionOutcome, reference: ExecutionOutcome) -> str:
    if candidate.status == "timeout" or reference.status == "timeout":
        return "timeout"
    if candidate.status in {"error", "import_setup"} or reference.status in {
        "error",
        "import_setup",
    }:
        return "import/setup failure"
    if candidate.status == "fail" and reference.status == "pass":
        return "candidate-only failure"
    if candidate.status == "pass" and reference.status == "fail":
        return "reference-only failure"
    if candidate.status == "fail" and reference.status == "fail":
        return "failure on both versions"
    if candidate.status == "pass" and reference.status == "pass":
        return "pass on both versions"
    return "other"
