from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--to-split", choices=("train", "dev", "eval"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    connection = sqlite3.connect(args.database)
    before = dict(
        connection.execute(
            """SELECT s.case_id,s.split FROM splits s
               JOIN dataset_cases d ON d.id=s.case_id
               WHERE d.dataset='bugsinpy' AND d.project=?""",
            (args.project,),
        )
    )
    if not before:
        raise RuntimeError(f"no BugsInPy split rows found for project {args.project}")

    eval_projects_before = {
        row[0]
        for row in connection.execute(
            """SELECT DISTINCT d.project FROM splits s
               JOIN dataset_cases d ON d.id=s.case_id
               WHERE d.dataset='bugsinpy' AND s.split='eval'"""
        )
    }
    connection.execute(
        """UPDATE splits SET split=? WHERE case_id IN
           (SELECT id FROM dataset_cases WHERE dataset='bugsinpy' AND project=?)""",
        (args.to_split, args.project),
    )

    migrated_records = 0
    for record_id, raw in connection.execute(
        "SELECT id,record_json FROM records WHERE project=?", (args.project,)
    ).fetchall():
        value = json.loads(raw)
        value["split"] = args.to_split
        connection.execute(
            "UPDATE records SET split=?,record_json=? WHERE id=?",
            (args.to_split, json.dumps(value, separators=(",", ":")), record_id),
        )
        migrated_records += 1
    connection.commit()

    project_leaks = list(
        connection.execute(
            """SELECT d.project,COUNT(DISTINCT s.split) FROM splits s
               JOIN dataset_cases d ON d.id=s.case_id
               WHERE d.dataset='bugsinpy'
               GROUP BY d.project HAVING COUNT(DISTINCT s.split)>1"""
        )
    )
    eval_projects_after = {
        row[0]
        for row in connection.execute(
            """SELECT DISTINCT d.project FROM splits s
               JOIN dataset_cases d ON d.id=s.case_id
               WHERE d.dataset='bugsinpy' AND s.split='eval'"""
        )
    }
    if project_leaks:
        raise RuntimeError(f"project isolation violated: {project_leaks}")
    if eval_projects_after != eval_projects_before:
        raise RuntimeError("held-out eval projects changed unexpectedly")

    valid_by_split = {"train": 0, "dev": 0, "eval": 0}
    for validation in (args.artifacts / "validations").glob("bugsinpy-*.json"):
        try:
            result = json.loads(validation.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if result.get("validator_version") != "bugsinpy-counterfactual-v2" or not result.get(
            "valid"
        ):
            continue
        row = connection.execute(
            "SELECT split FROM splits WHERE case_id=?", (result["case_id"],)
        ).fetchone()
        if row:
            valid_by_split[str(row[0])] += 1

    output = {
        "changed_utc": datetime.now(UTC).isoformat(),
        "project": args.project,
        "from_splits": sorted(set(before.values())),
        "to_split": args.to_split,
        "case_rows_changed": len(before),
        "records_migrated": migrated_records,
        "eval_projects_unchanged": sorted(eval_projects_after),
        "project_isolation_verified": True,
        "valid_pool": {**valid_by_split, "total": sum(valid_by_split.values())},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
