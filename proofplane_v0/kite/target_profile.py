from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from proofplane_v0.corpus.store import CorpusStore, cache_key
from proofplane_v0.datasets.base import DatasetCase
from proofplane_v0.kite.api import OpenAIService
from proofplane_v0.kite.ast_units import (
    function_nodes,
    production_python_files,
    source_segment,
    structural_query_units,
    structural_unit,
    test_python_files,
)
from proofplane_v0.kite.models import (
    FocalMethodSpec,
    ReferenceTargetProfile,
    RepositoryContext,
    TargetCodeUnit,
    TargetSemanticFields,
)

PROFILE_VERSION = "kite-reference-profile-v4-api-grounded-test-context"
SEMANTIC_PROFILE_VERSION = "kite-target-semantic-what-where-how-v1"


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    try:
        return f"{prefix} {node.name}{ast.unparse(node.args)}"
    except (ValueError, AttributeError):
        return f"{prefix} {node.name}(...)"


def _read_tests(reference: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in test_python_files(reference):
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        result[path.relative_to(reference).as_posix()] = source
    return result


def _related_tests(
    qualname: str,
    focal_file: str,
    calls: list[str],
    tests: dict[str, str],
    limit: int = 3,
    snippet_chars: int = 2400,
) -> tuple[dict[str, str], int]:
    """Find API examples even when a test does not name the private focal method."""
    symbol = qualname.rsplit(".", 1)[-1]
    class_name = qualname.rsplit(".", 1)[0] if "." in qualname else ""
    module = Path(focal_file).with_suffix("").as_posix().replace("/", ".")
    module_parts = [part for part in module.split(".") if part != "__init__"]
    module_candidates = {
        ".".join(module_parts[index:])
        for index in range(len(module_parts))
        if module_parts[index:]
    }
    call_names = {
        value for value in calls if len(value) >= 4 and not value.startswith("_")
    }
    signals = ({symbol} if len(symbol) >= 4 else set()) | call_names
    if class_name:
        signals.add(class_name.rsplit(".", 1)[-1])

    ranked: list[tuple[int, int, str, str]] = []
    for name, source in tests.items():
        exact_hits = source.count(symbol) if len(symbol) >= 4 else 0
        class_hits = source.count(class_name.rsplit(".", 1)[-1]) if class_name else 0
        module_hits = sum(source.count(value) for value in module_candidates)
        collaborator_hits = sum(source.count(value) for value in call_names)
        path_stem = Path(focal_file).stem
        path_hits = source.count(path_stem) if len(path_stem) >= 4 else 0
        score = (
            exact_hits * 12
            + class_hits * 7
            + module_hits * 6
            + collaborator_hits * 2
            + path_hits * 3
        )
        if score:
            ranked.append((score, sum(source.count(value) for value in signals), name, source))
    ranked = sorted(
        ranked,
        key=lambda value: (-value[0], -value[1], value[2]),
    )[:limit]
    return (
        {name: source[:snippet_chars] for _score, _hits, name, source in ranked},
        sum(value[1] for value in ranked),
    )


def _call_names(node: ast.AST) -> list[str]:
    names: list[str] = []
    for value in ast.walk(node):
        if not isinstance(value, ast.Call):
            continue
        if isinstance(value.func, ast.Name):
            names.append(value.func.id)
        elif isinstance(value.func, ast.Attribute):
            names.append(value.func.attr)
    return list(dict.fromkeys(names))


def _type_constraints(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    values: list[str] = []
    for argument in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]:
        if argument.annotation is not None:
            values.append(f"{argument.arg}: {ast.unparse(argument.annotation)}")
    if node.returns is not None:
        values.append(f"returns: {ast.unparse(node.returns)}")
    return values


def _same_class_helpers(
    full_source: str, qualname: str, focal_name: str, limit: int = 4
) -> dict[str, str]:
    if "." not in qualname:
        return {}
    prefix = qualname.rsplit(".", 1)[0] + "."
    values: dict[str, str] = {}
    for name, node in function_nodes(full_source):
        if name != focal_name and name.startswith(prefix):
            values[name] = source_segment(full_source, node)[:1800]
            if len(values) >= limit:
                break
    return values


def _called_dependencies(
    reference: Path,
    calls: list[str],
    focal_file: str,
    focal_qualname: str,
    limit: int = 6,
) -> dict[str, str]:
    wanted = set(calls)
    values: dict[str, str] = {}
    if not wanted:
        return values
    for path in production_python_files(reference):
        relative = path.relative_to(reference).as_posix()
        try:
            source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for qualname, node in function_nodes(source):
            if qualname.rsplit(".", 1)[-1] not in wanted:
                continue
            if relative == focal_file and qualname == focal_qualname:
                continue
            values[f"{relative}::{qualname}"] = source_segment(source, node)[:1800]
            if len(values) >= limit:
                return values
    return values


def build_reference_profile(
    case: DatasetCase,
    reference: Path,
    store: CorpusStore,
    max_units: int,
    max_unit_chars: int,
    focal_methods: list[FocalMethodSpec] | None = None,
) -> ReferenceTargetProfile:
    """Build a target profile from the fixed/reference checkout only.

    This function intentionally has no candidate/buggy workspace parameter.
    """
    commit = str(case.metadata.get("fixed_commit_id", ""))
    focal_identity = json.dumps(
        [value.model_dump(mode="json") for value in (focal_methods or [])], sort_keys=True
    )
    key = cache_key(
        "deterministic",
        PROFILE_VERSION,
        "ReferenceTargetProfile",
        f"{case.case_id}:{commit}:{max_units}:{max_unit_chars}:{focal_identity}",
    )
    cached = store.kite_get(key)
    if cached and cached[1] == "complete":
        return ReferenceTargetProfile.model_validate_json(cached[0])

    tests = _read_tests(reference)
    candidates: list[TargetCodeUnit] = []
    repository_files = [path.relative_to(reference).as_posix() for path in production_python_files(reference)]
    focal_by_file: dict[str, dict[str, FocalMethodSpec]] | None = None
    if focal_methods is not None:
        if not focal_methods:
            raise ValueError(f"focal benchmark contains no methods for {case.case_id}")
        focal_by_file = {}
        for value in focal_methods:
            focal_by_file.setdefault(value.file, {})[value.qualname] = value
        input_paths = [reference / value for value in sorted(focal_by_file)]
    else:
        input_paths = production_python_files(reference)
    for path in input_paths:
        resolved = path.resolve()
        if not resolved.is_relative_to(reference.resolve()) or not path.is_file():
            raise ValueError(f"focal file is absent or outside reference workspace: {path}")
        relative = path.relative_to(reference).as_posix()
        try:
            full_source = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for qualname, node in function_nodes(full_source):
            focal = focal_by_file.get(relative, {}).get(qualname) if focal_by_file else None
            if focal_by_file is not None and focal is None:
                continue
            source = source_segment(full_source, node)
            if len(source) < 40 or len(source) > max_unit_chars:
                if focal is not None:
                    raise ValueError(
                        f"focal method {relative}::{qualname} is outside the configured size limit"
                    )
                continue
            source_hash = hashlib.sha256(source.encode()).hexdigest()
            if focal is not None and (
                source_hash != focal.reference_source_hash
                or int(getattr(node, "lineno", 1)) != focal.start_line
                or int(getattr(node, "end_lineno", getattr(node, "lineno", 1))) != focal.end_line
            ):
                raise ValueError(
                    f"focal benchmark integrity mismatch for {relative}::{qualname}; rebuild benchmark"
                )
            docstring = ast.get_docstring(node, clean=True) or ""
            calls = _call_names(node)
            related, test_hits = _related_tests(qualname, relative, calls, tests)
            public = not qualname.rsplit(".", 1)[-1].startswith("_")
            control_nodes = sum(
                isinstance(item, (ast.If, ast.For, ast.While, ast.Try, ast.Compare, ast.Call))
                for item in ast.walk(node)
            )
            deterministic_score = (
                min(test_hits, 20) * 3.0
                + (8.0 if public else 0.0)
                + (5.0 if docstring else 0.0)
                + min(control_nodes, 20) * 0.5
            )
            structural = structural_unit(relative, qualname, node, source)
            structural_queries = structural_query_units(relative, qualname, node)
            types = _type_constraints(node)
            helpers = _same_class_helpers(full_source, qualname, qualname)
            dependencies = _called_dependencies(reference, calls, relative, qualname)
            return_shapes = sorted(
                {
                    type(value.value).__name__ if value.value is not None else "None"
                    for value in ast.walk(node)
                    if isinstance(value, ast.Return)
                }
            )
            controls = sorted(
                {
                    type(value).__name__
                    for value in ast.walk(node)
                    if isinstance(value, (ast.If, ast.For, ast.While, ast.Try, ast.Compare))
                }
            )
            semantic_what = (
                docstring[:1000]
                or f"Calls {', '.join(calls[:8]) or 'no collaborators'} and returns "
                f"{', '.join(return_shapes) or 'implicitly'}."
            )
            semantic_where = (
                f"Signature {_signature(node)}. Type constraints: {'; '.join(types) or 'implicit'}. "
                f"Control shapes: {', '.join(controls) or 'straight-line'}."
            )
            affordances = sorted(
                {
                    f"{value.anchor_role}:{value.node_type}"
                    for value in structural_queries
                }
            )
            semantic_how = (
                "Local executable edit affordances: "
                + (", ".join(affordances[:12]) or "return/statement behavior")
            )
            content_hash = hashlib.sha256(f"{relative}\0{qualname}\0{source}".encode()).hexdigest()
            candidates.append(
                TargetCodeUnit(
                    unit_id=f"{case.case_id}:{content_hash[:16]}",
                    file=relative,
                    qualname=qualname,
                    signature=_signature(node),
                    docstring=docstring[:1600],
                    source=source,
                    semantic_text=(
                        f"WHAT: {semantic_what}\nWHERE: {semantic_where}\nHOW: {semantic_how}"
                    ),
                    semantic_what=semantic_what,
                    semantic_where=semantic_where,
                    semantic_how=semantic_how,
                    structural=structural,
                    structural_queries=structural_queries,
                    related_tests=related,
                    repository_context=RepositoryContext(
                        same_class_helpers=helpers,
                        called_dependencies=dependencies,
                        related_tests=related,
                        documentation=[docstring[:1600]] if docstring else [],
                        type_constraints=types,
                    ),
                    deterministic_score=deterministic_score,
                    content_hash=content_hash,
                )
            )
    candidates.sort(key=lambda unit: (-unit.deterministic_score, unit.file, unit.qualname))
    selected = candidates if focal_by_file is not None else candidates[:max_units]
    if focal_by_file is not None and len(selected) != len(focal_methods or []):
        found = {(value.file, value.qualname) for value in selected}
        expected = {(value.file, value.qualname) for value in (focal_methods or [])}
        missing = sorted(expected - found)
        raise ValueError(f"focal methods are absent from reference checkout: {missing}")
    digest = hashlib.sha256(
        json.dumps(
            [(unit.unit_id, unit.content_hash) for unit in selected],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    profile = ReferenceTargetProfile(
        case_id=case.case_id,
        project=case.project,
        reference_commit=commit,
        repository_files=repository_files[:400],
        units=selected,
        content_hash=digest,
    )
    store.kite_put(key, "target_profile", profile.model_dump_json())
    return profile


def enrich_target_semantics(
    profile: ReferenceTargetProfile, service: OpenAIService
) -> ReferenceTargetProfile:
    """Create paper-aligned target WHAT/WHERE/HOW using reference-visible evidence only."""
    enriched: list[TargetCodeUnit] = []
    for unit in profile.units:
        fields = service.parse(
            schema=TargetSemanticFields,
            purpose="kite-target-semantic-profile",
            version=SEMANTIC_PROFILE_VERSION,
            system=(
                "Profile the supplied REFERENCE focal method without hypothesizing or using any real bug. "
                "Summarize: mechanism_bearing_behavior (WHAT semantic behavior could carry a defect), "
                "applicability_context (WHERE local shapes, types, collaborators, and constraints apply), "
                "and edit_affordance (HOW local executable edits could perturb that behavior). Stay grounded "
                "in supplied visible code and repository context. Do not claim that any defect exists."
            ),
            payload={
                "case_id": profile.case_id,
                "project": profile.project,
                "focal_method": {
                    "file": unit.file,
                    "qualname": unit.qualname,
                    "signature": unit.signature,
                    "docstring": unit.docstring,
                    "source": unit.source,
                },
                "repository_context": unit.repository_context.model_dump(mode="json"),
                "local_structural_affordances": [
                    {
                        "anchor_role": value.anchor_role,
                        "node_type": value.node_type,
                        "relations": value.relations,
                    }
                    for value in unit.structural_queries
                ],
            },
        )
        enriched.append(
            unit.model_copy(
                update={
                    "semantic_what": fields.mechanism_bearing_behavior,
                    "semantic_where": fields.applicability_context,
                    "semantic_how": fields.edit_affordance,
                    "semantic_text": (
                        f"WHAT: {fields.mechanism_bearing_behavior}\n"
                        f"WHERE: {fields.applicability_context}\n"
                        f"HOW: {fields.edit_affordance}"
                    ),
                }
            )
        )
    digest = hashlib.sha256(
        json.dumps(
            [(value.unit_id, value.semantic_text) for value in enriched],
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    return profile.model_copy(
        update={
            "units": enriched,
            "content_hash": digest,
            "schema_version": SEMANTIC_PROFILE_VERSION,
        }
    )
