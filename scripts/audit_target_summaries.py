from __future__ import annotations

import argparse
import ast
import json
import re
import sqlite3
from pathlib import Path


PATH_PATTERN = re.compile(r"(?:[A-Za-z]:\\|[/\\]|\b[\w.-]+\.py\b)", re.IGNORECASE)
IDENTIFIER_PATTERN = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
NUMBER_PATTERN = re.compile(r"(?<![A-Za-z_])\d+(?:\.\d+)?(?![A-Za-z_])")
FIX_LANGUAGE = re.compile(
    r"\b(?:the fix|fix ensures|fix by|ensure that|must (?:initialize|propagate|preserve|carry|run|create))\b",
    re.IGNORECASE,
)
GENERIC_IDENTIFIERS = {
    "api",
    "json",
    "model",
    "models",
    "response",
    "request",
    "schema",
    "schemas",
    "field",
    "fields",
    "route",
    "routes",
}


def code_identifiers(text: str) -> set[str]:
    values = set(IDENTIFIER_PATTERN.findall(text))
    return {
        value.casefold()
        for value in values
        if "_" in value or any(char.isupper() for char in value[1:])
    }


def string_literals(text: str) -> set[str]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return set()
    return {
        node.value.casefold()
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and len(node.value.strip()) >= 4
        and (
            not node.value.strip().isalpha()
            or " " in node.value.strip()
            or "_" in node.value.strip()
            or any(char.isupper() for char in node.value.strip()[1:])
        )
    }


def audit(database: Path, data_dir: Path) -> dict[str, object]:
    connection = sqlite3.connect(database)
    rows = connection.execute(
        "SELECT id,hidden_json FROM records WHERE split='eval' ORDER BY id"
    ).fetchall()
    results: list[dict[str, object]] = []
    for case_id, hidden_json in rows:
        hidden = json.loads(hidden_json)
        summary = str(hidden["portable_prompt_summary"])
        workspace = data_dir / "workspaces" / case_id.replace(":", "-")
        oracle_text = "\n".join(
            path.read_text(encoding="utf-8", errors="replace")
            for path in (workspace / "oracle_tests").rglob("*.py")
        ) if (workspace / "oracle_tests").exists() else ""
        context_path = data_dir / "artifacts" / "contexts" / f"{case_id.replace(':', '-')}.json"
        context = json.loads(context_path.read_text(encoding="utf-8")) if context_path.exists() else {}
        validation_path = (
            data_dir / "artifacts" / "validations" / f"{case_id.replace(':', '-')}.json"
        )
        validation = (
            json.loads(validation_path.read_text(encoding="utf-8"))
            if validation_path.exists()
            else {}
        )
        visible_diff = str(context.get("visible_diff", ""))
        sensitive_text = visible_diff + "\n" + oracle_text
        sensitive_ids = code_identifiers(sensitive_text) - GENERIC_IDENTIFIERS
        summary_ids = {item.casefold() for item in IDENTIFIER_PATTERN.findall(summary)}
        identifier_overlap = sorted(sensitive_ids & summary_ids)
        sensitive_literals = string_literals(oracle_text)
        literal_overlap = sorted(value for value in sensitive_literals if value in summary.casefold())
        source_numbers = set(NUMBER_PATTERN.findall(sensitive_text)) - {"0", "1", "2"}
        number_overlap = sorted(set(NUMBER_PATTERN.findall(summary)) & source_numbers)
        hard_findings: list[str] = []
        if PATH_PATTERN.search(summary):
            hard_findings.append("path_or_filename")
        if identifier_overlap:
            hard_findings.append("code_identifier_overlap")
        if literal_overlap:
            hard_findings.append("hidden_test_literal_overlap")
        if number_overlap:
            hard_findings.append("exact_numeric_literal_overlap")
        if not validation.get("valid"):
            hard_findings.append("counterfactual_validation_missing_or_invalid")
        if not (workspace / "candidate").exists() or not (workspace / "reference").exists():
            hard_findings.append("workspace_missing")
        advisories: list[str] = []
        if FIX_LANGUAGE.search(summary):
            advisories.append("prescriptive_wording")
        results.append(
            {
                "case_id": case_id,
                "status": "pass" if not hard_findings else "fail",
                "summary": summary,
                "findings": hard_findings,
                "advisories": advisories,
                "identifier_overlap": identifier_overlap,
                "hidden_test_literal_overlap": literal_overlap,
                "numeric_literal_overlap": number_overlap,
            }
        )
    return {
        "schema_version": "target-summary-audit-v1",
        "cases": len(results),
        "passed": sum(item["status"] == "pass" for item in results),
        "failed": sum(item["status"] == "fail" for item in results),
        "advisories": sum(len(item["advisories"]) for item in results),
        "results": results,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, default=Path("data/full/proofplane.sqlite3"))
    parser.add_argument("--data-dir", type=Path, default=Path("data/full"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.database, args.data_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Portable-summary audit: {report['passed']}/{report['cases']} passed; {report['failed']} failed")
    print(f"Report: {args.output}")
    return 0 if report["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
