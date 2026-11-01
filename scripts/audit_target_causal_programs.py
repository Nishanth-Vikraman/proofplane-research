from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

from audit_target_summaries import (
    GENERIC_IDENTIFIERS,
    IDENTIFIER_PATTERN,
    NUMBER_PATTERN,
    PATH_PATTERN,
    code_identifiers,
    string_literals,
)
from proofplane_v0.experiments.runner import target_causal_program_evidence
from proofplane_v0.models import CorpusBugRecord


def audit(database: Path, data_dir: Path) -> dict[str, object]:
    connection = sqlite3.connect(database)
    rows = connection.execute(
        "SELECT id,record_json FROM records WHERE split='eval' ORDER BY id"
    ).fetchall()
    results: list[dict[str, object]] = []
    for case_id, record_json in rows:
        record = CorpusBugRecord.model_validate_json(record_json)
        evidence = target_causal_program_evidence(record)[0]
        program = evidence.causal_program
        assert program is not None
        serialized = json.dumps(
            program.model_dump(mode="json"), sort_keys=True, ensure_ascii=False
        )
        workspace = data_dir / "workspaces" / case_id.replace(":", "-")
        oracle_text = (
            "\n".join(
                path.read_text(encoding="utf-8", errors="replace")
                for path in (workspace / "oracle_tests").rglob("*.py")
            )
            if (workspace / "oracle_tests").exists()
            else ""
        )
        context_path = (
            data_dir / "artifacts" / "contexts" / f"{case_id.replace(':', '-')}.json"
        )
        context = (
            json.loads(context_path.read_text(encoding="utf-8"))
            if context_path.exists()
            else {}
        )
        validation_path = (
            data_dir / "artifacts" / "validations" / f"{case_id.replace(':', '-')}.json"
        )
        validation = (
            json.loads(validation_path.read_text(encoding="utf-8"))
            if validation_path.exists()
            else {}
        )
        sensitive_text = str(context.get("visible_diff", "")) + "\n" + oracle_text
        sensitive_ids = code_identifiers(sensitive_text) - GENERIC_IDENTIFIERS
        program_ids = {item.casefold() for item in IDENTIFIER_PATTERN.findall(serialized)}
        identifier_overlap = sorted(sensitive_ids & program_ids)
        sensitive_literals = string_literals(oracle_text)
        literal_overlap = sorted(
            value for value in sensitive_literals if value in serialized.casefold()
        )
        source_numbers = set(NUMBER_PATTERN.findall(sensitive_text)) - {"0", "1", "2"}
        number_overlap = sorted(set(NUMBER_PATTERN.findall(serialized)) & source_numbers)
        findings: list[str] = []
        if PATH_PATTERN.search(serialized):
            findings.append("path_or_filename")
        if identifier_overlap:
            findings.append("code_identifier_overlap")
        if literal_overlap:
            findings.append("hidden_test_literal_overlap")
        if number_overlap:
            findings.append("exact_numeric_literal_overlap")
        if not validation.get("valid"):
            findings.append("counterfactual_validation_missing_or_invalid")
        if not (workspace / "candidate").exists() or not (workspace / "reference").exists():
            findings.append("workspace_missing")
        results.append(
            {
                "case_id": case_id,
                "status": "pass" if not findings else "fail",
                "causal_program": program.model_dump(mode="json"),
                "findings": findings,
                "identifier_overlap": identifier_overlap,
                "hidden_test_literal_overlap": literal_overlap,
                "numeric_literal_overlap": number_overlap,
            }
        )
    return {
        "schema_version": "target-causal-program-audit-v1",
        "derivation": "deterministic-existing-enrichment-v1",
        "enrichment_api_calls": 0,
        "cases": len(results),
        "passed": sum(item["status"] == "pass" for item in results),
        "failed": sum(item["status"] == "fail" for item in results),
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
    print(
        f"Causal-program audit: {report['passed']}/{report['cases']} passed; "
        f"{report['failed']} failed; enrichment API calls: 0"
    )
    print(f"Report: {args.output}")
    return 0 if report["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
