from __future__ import annotations

import ast
import difflib
import hashlib
import re
from pathlib import Path

from pydantic import BaseModel, Field


class ContextBundle(BaseModel):
    case_id: str
    visible_diff: str
    changed_files: list[str]
    changed_symbols: list[str]
    nearby_tests: dict[str, str] = Field(default_factory=dict)
    repository_tree: list[str] = Field(default_factory=list)
    content_hash: str


def _symbols(text: str) -> list[str]:
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return []
    return [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))]


def _test_relevance(
    name: str, text: str, changed_files: list[str], changed_symbols: list[str]
) -> int:
    searchable = f"{name}\n{text}"
    generic_symbols = {
        "__init__",
        "decorator",
        "delete",
        "get",
        "head",
        "openapi",
        "options",
        "patch",
        "post",
        "put",
        "setup",
        "trace",
        "websocket",
    }
    useful_symbols = {
        symbol
        for symbol in changed_symbols
        if len(symbol) >= 4 and symbol.lower() not in generic_symbols
    }
    module_terms = {
        part.removesuffix("s")
        for changed in changed_files
        for part in Path(changed).with_suffix("").parts
        if len(part) >= 4 and part not in {"src", "test", "tests", "python", "utils"}
    }
    symbol_hits = sum(
        bool(re.search(rf"\b{re.escape(symbol)}\b", searchable)) for symbol in useful_symbols
    )
    module_hits = sum(
        bool(re.search(rf"\b{re.escape(term)}\w*\b", searchable)) for term in module_terms
    )
    direct_module_match = sum(term in Path(name).stem.lower() for term in module_terms)
    return direct_module_match * 100 + symbol_hits * 10 + module_hits * 3


def _bundle_size(bundle: ContextBundle) -> int:
    return len(bundle.model_dump_json())


def build_bundle(
    case_id: str,
    base: Path,
    candidate: Path,
    max_chars: int = 30000,
    hidden_test_names: set[str] | None = None,
    max_nearby_tests: int = 8,
    max_test_snippet_chars: int = 2500,
    max_repository_tree_entries: int = 100,
) -> ContextBundle:
    if max_chars < 2000:
        raise ValueError("max_chars must be at least 2000")
    hidden = hidden_test_names or set()
    ignored_parts = {".git", "proofplane_generated", "oracle_tests", ".venv", "env"}
    files = sorted(
        name
        for name in (
            {p.relative_to(base).as_posix() for p in base.rglob("*.py")}
            | {p.relative_to(candidate).as_posix() for p in candidate.rglob("*.py")}
        )
        if not ignored_parts.intersection(Path(name).parts)
    )
    diffs: list[str] = []
    changed: list[str] = []
    symbols: list[str] = []
    test_candidates: dict[str, str] = {}
    for name in files:
        if name in hidden:
            continue
        before = (base / name).read_text(encoding="utf-8") if (base / name).exists() else ""
        after = (
            (candidate / name).read_text(encoding="utf-8") if (candidate / name).exists() else ""
        )
        if name.startswith("test") or "/test" in name:
            test_candidates[name] = before
        elif before != after:
            changed.append(name)
            symbols.extend(sorted(set(_symbols(before) + _symbols(after))))
            diffs.extend(
                difflib.unified_diff(
                    before.splitlines(),
                    after.splitlines(),
                    fromfile=f"base/{name}",
                    tofile=f"candidate/{name}",
                    lineterm="",
                )
            )
    full_diff = "\n".join(diffs)
    ranked_tests = sorted(
        (
            (_test_relevance(name, source, changed, symbols), name, source)
            for name, source in test_candidates.items()
        ),
        key=lambda item: (-item[0], item[1]),
    )
    ranked_tests = [item for item in ranked_tests if item[0] > 0][:max_nearby_tests]

    test_budget = int(max_chars * 0.30)
    tests: dict[str, str] = {}
    for _score, name, source in ranked_tests:
        if test_budget <= 0:
            break
        snippet = source[: min(max_test_snippet_chars, test_budget)]
        if snippet:
            tests[name] = snippet
            test_budget -= len(snippet)

    tree_budget = int(max_chars * 0.08)
    tree: list[str] = []
    for name in (x for x in files if x not in hidden):
        if len(tree) >= max_repository_tree_entries or len(name) > tree_budget:
            continue
        tree.append(name)
        tree_budget -= len(name)

    text = full_diff[: int(max_chars * 0.55)]
    bundle = ContextBundle(
        case_id=case_id,
        visible_diff=text,
        changed_files=changed,
        changed_symbols=sorted(set(symbols)),
        nearby_tests=tests,
        repository_tree=tree,
        content_hash="0" * 64,
    )
    while _bundle_size(bundle) > max_chars:
        overflow = _bundle_size(bundle) - max_chars
        if len(bundle.visible_diff) > 4000:
            remove = min(max(overflow, 1), len(bundle.visible_diff) - 4000)
            bundle.visible_diff = bundle.visible_diff[:-remove]
        elif bundle.nearby_tests:
            largest = max(bundle.nearby_tests, key=lambda key: len(bundle.nearby_tests[key]))
            if len(bundle.nearby_tests[largest]) > 500:
                remove = min(max(overflow, 1), len(bundle.nearby_tests[largest]) - 500)
                bundle.nearby_tests[largest] = bundle.nearby_tests[largest][:-remove]
            else:
                del bundle.nearby_tests[largest]
        elif bundle.repository_tree:
            bundle.repository_tree.pop()
        else:
            raise ValueError("context metadata exceeds max_chars")

    bundle.content_hash = hashlib.sha256(
        bundle.model_dump_json(exclude={"content_hash"}).encode()
    ).hexdigest()
    return bundle
