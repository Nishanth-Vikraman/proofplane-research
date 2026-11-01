from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from proofplane_v0.models import CorpusBugRecord

SCHEMA = """
CREATE TABLE IF NOT EXISTS records (id TEXT PRIMARY KEY, project TEXT, split TEXT, visible_json TEXT, hidden_json TEXT, record_json TEXT);
CREATE TABLE IF NOT EXISTS embeddings (record_id TEXT, kind TEXT, vector BLOB, dimensions INTEGER, PRIMARY KEY(record_id, kind));
CREATE TABLE IF NOT EXISTS llm_cache (cache_key TEXT PRIMARY KEY, response_json TEXT, error TEXT);
CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, status TEXT, metadata_json TEXT);
CREATE TABLE IF NOT EXISTS api_usage (id INTEGER PRIMARY KEY AUTOINCREMENT, model TEXT, purpose TEXT, input_tokens INTEGER DEFAULT 0, output_tokens INTEGER DEFAULT 0, estimated_cost REAL, created_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS dataset_cases (id TEXT PRIMARY KEY, dataset TEXT, project TEXT, bug_id TEXT, metadata_json TEXT, prepared INTEGER DEFAULT 0, exclusion TEXT DEFAULT '');
CREATE TABLE IF NOT EXISTS splits (case_id TEXT PRIMARY KEY, split TEXT);
CREATE TABLE IF NOT EXISTS vocab (field TEXT, slug TEXT, canonical TEXT, aliases_json TEXT, status TEXT, definition TEXT, parent TEXT, reason TEXT, PRIMARY KEY(field, slug));
CREATE TABLE IF NOT EXISTS kite_cache (cache_key TEXT PRIMARY KEY, stage TEXT, payload_json TEXT, status TEXT, updated_at TEXT DEFAULT CURRENT_TIMESTAMP);
"""


class CorpusStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.executescript(SCHEMA)

    def put(self, record: CorpusBugRecord) -> None:
        visible = record.visible_descriptor.model_dump_json()
        hidden = record.hidden_payload.model_dump_json()
        self.connection.execute(
            "INSERT OR REPLACE INTO records VALUES (?,?,?,?,?,?)",
            (
                record.corpus_id,
                record.project,
                record.split,
                visible,
                hidden,
                record.model_dump_json(),
            ),
        )
        self.connection.commit()

    def records(self, split: str | None = None) -> list[CorpusBugRecord]:
        sql = "SELECT record_json FROM records"
        args: tuple[object, ...] = ()
        if split:
            sql, args = sql + " WHERE split=?", (split,)
        return [
            CorpusBugRecord.model_validate_json(row[0])
            for row in self.connection.execute(sql, args)
        ]

    def eligible_records(self, split: str | None = None) -> list[CorpusBugRecord]:
        records = self.records(split)
        status = {
            row[0]: (bool(row[1]), str(row[2]))
            for row in self.connection.execute("SELECT id,prepared,exclusion FROM dataset_cases")
        }
        return [
            record
            for record in records
            if record.dataset != "bugsinpy"
            or (
                status.get(record.corpus_id, (False, "missing"))[0]
                and not status.get(record.corpus_id, (False, "missing"))[1]
            )
        ]

    def primary_index_export(self, split: str | None = None) -> list[dict[str, object]]:
        """Deliberately excludes hidden_json; this is the auditable causal index boundary."""
        return [
            {"id": record.corpus_id, "visible_descriptor": record.visible_descriptor.model_dump()}
            for record in self.eligible_records(split)
        ]

    def put_case(
        self,
        case_id: str,
        dataset: str,
        project: str,
        bug_id: str,
        metadata: dict[str, object],
        prepared: bool = False,
        exclusion: str = "",
    ) -> None:
        self.connection.execute(
            """INSERT INTO dataset_cases VALUES (?,?,?,?,?,?,?)
               ON CONFLICT(id) DO UPDATE SET
                 dataset=excluded.dataset, project=excluded.project,
                 bug_id=excluded.bug_id, metadata_json=excluded.metadata_json""",
            (case_id, dataset, project, bug_id, json.dumps(metadata), int(prepared), exclusion),
        )
        self.connection.commit()

    def mark_prepared(self, case_id: str, prepared: bool = True, exclusion: str = "") -> None:
        self.connection.execute(
            "UPDATE dataset_cases SET prepared=?, exclusion=? WHERE id=?",
            (int(prepared), exclusion, case_id),
        )
        self.connection.commit()

    def put_splits(self, values: dict[str, str]) -> None:
        self.connection.executemany("INSERT OR REPLACE INTO splits VALUES (?,?)", values.items())
        self.connection.commit()

    def splits(self) -> dict[str, str]:
        return dict(self.connection.execute("SELECT case_id,split FROM splits"))

    def cache_get(self, key: str) -> str | None:
        row = self.connection.execute(
            "SELECT response_json FROM llm_cache WHERE cache_key=? AND error IS NULL", (key,)
        ).fetchone()
        return row[0] if row else None

    def cache_put(self, key: str, response_json: str, error: str | None = None) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO llm_cache VALUES (?,?,?)", (key, response_json, error)
        )
        self.connection.commit()

    def put_embedding(self, record_id: str, kind: str, vector: bytes, dimensions: int) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO embeddings VALUES (?,?,?,?)",
            (record_id, kind, vector, dimensions),
        )
        self.connection.commit()

    def get_embedding(self, record_id: str, kind: str) -> tuple[bytes, int] | None:
        row = self.connection.execute(
            "SELECT vector,dimensions FROM embeddings WHERE record_id=? AND kind=?",
            (record_id, kind),
        ).fetchone()
        return (bytes(row[0]), int(row[1])) if row else None

    def kite_get(self, key: str) -> tuple[str, str] | None:
        row = self.connection.execute(
            "SELECT payload_json,status FROM kite_cache WHERE cache_key=?", (key,)
        ).fetchone()
        return (str(row[0]), str(row[1])) if row else None

    def kite_put(self, key: str, stage: str, payload_json: str, status: str = "complete") -> None:
        self.connection.execute(
            """INSERT INTO kite_cache(cache_key,stage,payload_json,status,updated_at)
               VALUES (?,?,?,?,CURRENT_TIMESTAMP)
               ON CONFLICT(cache_key) DO UPDATE SET
                 stage=excluded.stage,payload_json=excluded.payload_json,
                 status=excluded.status,updated_at=CURRENT_TIMESTAMP""",
            (key, stage, payload_json, status),
        )
        self.connection.commit()

    def put_run(self, run_id: str, status: str, metadata: dict[str, object]) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO runs VALUES (?,?,?)",
            (run_id, status, json.dumps(metadata, sort_keys=True)),
        )
        self.connection.commit()

    def reserve_api_call(
        self,
        model: str,
        purpose: str,
        budget: float,
        reserve: float,
        allow_overrun: bool = False,
        scope_prefix: str | None = None,
    ) -> int:
        if scope_prefix:
            spent = float(
                self.connection.execute(
                    "SELECT COALESCE(SUM(estimated_cost),0) FROM api_usage WHERE purpose LIKE ?",
                    (scope_prefix,),
                ).fetchone()[0]
            )
        else:
            spent = float(
                self.connection.execute(
                    "SELECT COALESCE(SUM(estimated_cost),0) FROM api_usage"
                ).fetchone()[0]
            )
        if not allow_overrun and spent + reserve > budget:
            raise RuntimeError(
                f"API spend guard: ${spent:.2f} reserved; next ${reserve:.2f} call exceeds ${budget:.2f} budget"
            )
        cursor = self.connection.execute(
            "INSERT INTO api_usage(model,purpose,estimated_cost) VALUES (?,?,?)",
            (model, purpose, reserve),
        )
        self.connection.commit()
        if cursor.lastrowid is None:
            raise RuntimeError("failed to reserve API usage row")
        return int(cursor.lastrowid)

    def update_api_tokens(self, usage_id: int, input_tokens: int, output_tokens: int) -> None:
        self.connection.execute(
            "UPDATE api_usage SET input_tokens=?,output_tokens=? WHERE id=?",
            (input_tokens, output_tokens, usage_id),
        )
        self.connection.commit()

    def update_api_usage(
        self, usage_id: int, input_tokens: int, output_tokens: int, estimated_cost: float
    ) -> None:
        self.connection.execute(
            "UPDATE api_usage SET input_tokens=?,output_tokens=?,estimated_cost=? WHERE id=?",
            (input_tokens, output_tokens, estimated_cost, usage_id),
        )
        self.connection.commit()

    def api_usage(self) -> dict[str, float | int]:
        row = self.connection.execute(
            "SELECT COALESCE(SUM(input_tokens),0),COALESCE(SUM(output_tokens),0),COALESCE(SUM(estimated_cost),0) FROM api_usage"
        ).fetchone()
        return {
            "input_tokens": int(row[0]),
            "output_tokens": int(row[1]),
            "reserved_cost_usd": float(row[2]),
        }


def cache_key(model: str, prompt_version: str, schema_version: str, normalized_input: str) -> str:
    import hashlib

    return hashlib.sha256(
        "\0".join((model, prompt_version, schema_version, normalized_input)).encode()
    ).hexdigest()
