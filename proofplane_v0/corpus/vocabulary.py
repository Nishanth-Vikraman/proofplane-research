from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Sequence
from typing import Any


REGISTRY_FIELDS = ("bug_family", "mechanism_primitive")
PREFIXES = {"bug_family": "bf", "mechanism_primitive": "mp"}


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "unknown"


class VocabularyRegistry:
    def __init__(self, connection: sqlite3.Connection):
        self.db = connection

    def propose(
        self,
        field: str,
        value: str,
        parent: str = "",
        definition: str = "",
        nearest: str = "",
        reason: str = "",
    ) -> str:
        slug = slugify(value)
        self.db.execute(
            "INSERT OR REPLACE INTO vocab VALUES (?,?,?,?,?,?,?,?)",
            (
                field,
                slug,
                value,
                json.dumps([]),
                "proposed",
                definition,
                parent,
                f"nearest={nearest}; {reason}",
            ),
        )
        self.db.commit()
        return slug

    def approve(self, slug: str, field: str | None = None) -> None:
        sql = "UPDATE vocab SET status='canonical' WHERE slug=?"
        args: tuple[object, ...] = (slug,)
        if field:
            sql += " AND field=?"
            args += (field,)
        self.db.execute(sql, args)
        self.db.commit()

    def merge(self, source: str, target: str, field: str | None = None) -> None:
        suffix = " AND field=?" if field else ""
        target_args: tuple[object, ...] = (target,) + ((field,) if field else ())
        source_args: tuple[object, ...] = (source,) + ((field,) if field else ())
        row = self.db.execute(
            "SELECT canonical, aliases_json FROM vocab WHERE slug=?" + suffix, target_args
        ).fetchone()
        source_row = self.db.execute(
            "SELECT canonical FROM vocab WHERE slug=?" + suffix, source_args
        ).fetchone()
        if not row or not source_row:
            raise KeyError("unknown vocabulary slug")
        aliases = json.loads(row[1])
        aliases.append(source_row[0])
        update_suffix = " AND field=?" if field else ""
        self.db.execute(
            "UPDATE vocab SET aliases_json=? WHERE slug=?" + update_suffix,
            (json.dumps(aliases), target) + ((field,) if field else ()),
        )
        self.db.execute(
            "UPDATE vocab SET status='merged', parent=? WHERE slug=?" + update_suffix,
            (target, source) + ((field,) if field else ()),
        )
        self.db.commit()

    def list(self, status: str | None = None) -> Sequence[tuple[Any, ...]]:
        return list(
            self.db.execute(
                "SELECT field,slug,canonical,status FROM vocab"
                + (" WHERE status=?" if status else ""),
                (status,) if status else (),
            )
        )

    def entries(self, field: str) -> Sequence[dict[str, object]]:
        rows = self.db.execute(
            """SELECT slug,canonical,aliases_json,definition
               FROM vocab WHERE field=? AND status='canonical' ORDER BY slug""",
            (field,),
        )
        return [
            {
                "id": row[0],
                "name": row[1],
                "aliases": json.loads(row[2]),
                "definition": row[3],
            }
            for row in rows
        ]

    def snapshot(self) -> dict[str, Sequence[dict[str, object]]]:
        return {field: self.entries(field) for field in REGISTRY_FIELDS}

    def clear_generated(self) -> None:
        self.db.executemany("DELETE FROM vocab WHERE field=?", [(x,) for x in REGISTRY_FIELDS])
        self.db.commit()

    def _next_id(self, field: str) -> str:
        prefix = PREFIXES[field]
        ids = [str(item[0]) for item in self.db.execute("SELECT slug FROM vocab WHERE field=?", (field,))]
        numbers = [int(value.removeprefix(f"{prefix}-")) for value in ids if re.fullmatch(rf"{prefix}-\d+", value)]
        return f"{prefix}-{max(numbers, default=0) + 1:04d}"

    def resolve(
        self,
        field: str,
        value: str,
        *,
        allow_expand: bool,
        definition: str = "",
        reason: str = "",
    ) -> str:
        """Resolve a model label/ID; only training enrichment may expand the registry."""
        cleaned = value.strip()
        lowered = cleaned.casefold()
        for item in self.entries(field):
            aliases = item["aliases"]
            alias_names = (
                [str(alias) for alias in aliases]
                if isinstance(aliases, Sequence) and not isinstance(aliases, str)
                else []
            )
            names = [str(item["id"]), str(item["name"]), *alias_names]
            if lowered in {name.casefold() for name in names}:
                return str(item["id"])
        label = re.sub(r"^(?:new|unseen)\s*:\s*", "", cleaned, flags=re.IGNORECASE).strip()
        if not allow_expand:
            return f"unseen:{slugify(label)}"
        entry_id = self._next_id(field)
        self.db.execute(
            "INSERT INTO vocab VALUES (?,?,?,?,?,?,?,?)",
            (field, entry_id, label, "[]", "canonical", definition, "", reason),
        )
        self.db.commit()
        return entry_id
