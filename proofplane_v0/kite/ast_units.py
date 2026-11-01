from __future__ import annotations

import ast
import hashlib
import math
from collections import Counter
from pathlib import Path

from proofplane_v0.kite.models import StructuralUnit

ANCHOR_ROLES: tuple[tuple[type[ast.AST], str], ...] = (
    (ast.If, "control_header"),
    (ast.While, "control_header"),
    (ast.For, "control_header"),
    (ast.Try, "control_header"),
    (ast.Compare, "expression_center"),
    (ast.Call, "expression_center"),
    (ast.BoolOp, "expression_center"),
    (ast.BinOp, "expression_center"),
    (ast.Assign, "statement_center"),
    (ast.AnnAssign, "statement_center"),
    (ast.AugAssign, "statement_center"),
    (ast.Return, "statement_center"),
    (ast.Raise, "statement_center"),
)


class _Normalizer(ast.NodeTransformer):
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        updated = self.generic_visit(node)
        assert isinstance(updated, ast.FunctionDef)
        updated.name = "FUNCTION"
        return updated

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        updated = self.generic_visit(node)
        assert isinstance(updated, ast.AsyncFunctionDef)
        updated.name = "FUNCTION"
        return updated

    def visit_ClassDef(self, node: ast.ClassDef) -> ast.AST:
        updated = self.generic_visit(node)
        assert isinstance(updated, ast.ClassDef)
        updated.name = "CLASS"
        return updated

    def visit_Name(self, node: ast.Name) -> ast.AST:
        return ast.copy_location(ast.Name(id="ID", ctx=node.ctx), node)

    def visit_arg(self, node: ast.arg) -> ast.AST:
        annotation = self.visit(node.annotation) if node.annotation is not None else None
        return ast.copy_location(ast.arg(arg="ARG", annotation=annotation), node)

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        return ast.copy_location(
            ast.Attribute(value=self.visit(node.value), attr="ATTR", ctx=node.ctx), node
        )

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        kind = type(node.value).__name__.upper()
        return ast.copy_location(ast.Constant(value=f"<{kind}>"), node)


def normalized_dump(node: ast.AST) -> str:
    normalized = _Normalizer().visit(ast.fix_missing_locations(ast.parse(ast.unparse(node))))
    return ast.dump(normalized, annotate_fields=True, include_attributes=False)


def _role(node: ast.AST) -> str:
    for node_type, role in ANCHOR_ROLES:
        if isinstance(node, node_type):
            return role
    return "statement_center"


def _signatures(node: ast.AST) -> list[str]:
    signatures: set[str] = set()
    for child in ast.walk(node):
        role = _role(child)
        signatures.add(f"node:{type(child).__name__}")
        if isinstance(child, ast.Compare):
            signatures.update(f"op:{type(op).__name__}" for op in child.ops)
        elif isinstance(child, (ast.BoolOp, ast.BinOp, ast.UnaryOp)):
            signatures.add(f"op:{type(child.op).__name__}")
        elif isinstance(child, ast.Call):
            signatures.add(f"call_arity:{len(child.args)}:{len(child.keywords)}")
        if isinstance(child, ast.stmt):
            signatures.add(f"role:{role}")
    return sorted(signatures)


def structural_unit(file: str, qualname: str, node: ast.AST, source: str) -> StructuralUnit:
    normalized = normalized_dump(node)
    role = _role(node)
    node_type = type(node).__name__
    signatures = _signatures(node)
    coarse = f"python:{role}:{node_type}"
    fingerprint = hashlib.sha1(normalized.encode()).hexdigest()[:16]
    return StructuralUnit(
        file=file,
        qualname=qualname,
        anchor_role=role,
        node_type=node_type,
        coarse_key=coarse,
        fingerprint=fingerprint,
        normalized_tree=normalized[:12000],
        signatures=signatures,
        relations=[],
        anchor_count=1,
        relation_count=0,
        start_line=int(getattr(node, "lineno", 1)),
        end_line=int(getattr(node, "end_lineno", getattr(node, "lineno", 1))),
        source_hash=hashlib.sha256(source.encode()).hexdigest(),
    )


def _is_anchor(node: ast.AST) -> bool:
    return any(isinstance(node, node_type) for node_type, _role_name in ANCHOR_ROLES)


def _contains(root: ast.AST, target: ast.AST) -> bool:
    return any(value is target for value in ast.walk(root))


def _statement_neighbors(root: ast.AST) -> dict[ast.stmt, tuple[ast.stmt | None, ast.stmt | None]]:
    neighbors: dict[ast.stmt, tuple[ast.stmt | None, ast.stmt | None]] = {}
    for parent in ast.walk(root):
        for _field, value in ast.iter_fields(parent):
            if not isinstance(value, list):
                continue
            statements = [item for item in value if isinstance(item, ast.stmt)]
            for index, statement in enumerate(statements):
                neighbors[statement] = (
                    statements[index - 1] if index else None,
                    statements[index + 1] if index + 1 < len(statements) else None,
                )
    return neighbors


def _nearest_anchor(node: ast.AST) -> ast.AST | None:
    if _is_anchor(node):
        return node
    return next((value for value in ast.walk(node) if _is_anchor(value)), None)


def structural_query_units(
    file: str,
    qualname: str,
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    *,
    seed_lines: set[int] | None = None,
    limit: int = 48,
) -> list[StructuralUnit]:
    """Build KITE-style local anchor-relation trees around behavior-bearing AST nodes."""
    anchors = [value for value in ast.walk(function) if _is_anchor(value)]
    if seed_lines:
        selected = [
            value
            for value in anchors
            if any(
                int(getattr(value, "lineno", 0)) <= line <= int(getattr(value, "end_lineno", 0))
                for line in seed_lines
            )
        ]
        anchors = selected or anchors
    neighbors = _statement_neighbors(function)
    units: list[StructuralUnit] = []
    seen: set[str] = set()
    for root in anchors[:limit]:
        local: list[ast.AST] = [root]
        raw_relations: list[tuple[int, int, str]] = []

        def add(
            value: ast.AST | None,
            relation: str,
            current_root: ast.AST = root,
            current_local: list[ast.AST] = local,
            current_relations: list[tuple[int, int, str]] = raw_relations,
        ) -> None:
            if value is None or value is current_root:
                return
            if value not in current_local:
                current_local.append(value)
            current_relations.append((0, current_local.index(value), relation))

        if isinstance(root, (ast.If, ast.While)):
            add(_nearest_anchor(root.test), "condition_of")
            for value in list(root.body[:1]) + list(root.orelse[:1]):
                add(_nearest_anchor(value), "guard_to_body")
        elif isinstance(root, ast.For):
            add(_nearest_anchor(root.iter), "condition_of")
            for value in root.body[:1]:
                add(_nearest_anchor(value), "guard_to_body")
        elif isinstance(root, ast.Try):
            for value in root.body[:1]:
                add(_nearest_anchor(value), "guard_to_body")
        elif isinstance(root, ast.stmt):
            expression = next(
                (
                    value
                    for value in ast.walk(root)
                    if value is not root and _is_anchor(value) and not isinstance(value, ast.stmt)
                ),
                None,
            )
            add(expression, "expr_in_stmt")

        if isinstance(root, ast.stmt) and root in neighbors:
            previous, following = neighbors[root]
            add(_nearest_anchor(previous) if previous else None, "adjacent_stmt")
            add(_nearest_anchor(following) if following else None, "adjacent_stmt")
            if previous is not None and following is not None:
                left = _nearest_anchor(previous)
                right = _nearest_anchor(following)
                if left in local and right in local:
                    raw_relations.append((local.index(left), local.index(right), "same_block"))

        anchor_signatures = [
            f"A{index}:{_role(value)}:{type(value).__name__}:{normalized_dump(value)}"
            for index, value in enumerate(local)
        ]
        relations = sorted(
            f"A{left}-A{right}:{relation}" for left, right, relation in raw_relations
        )
        canonical = "\n".join(anchor_signatures + relations)
        fingerprint = hashlib.sha1(canonical.encode()).hexdigest()[:16]
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        roles = sorted(_role(value) for value in local)
        roots = sorted(type(value).__name__ for value in local)
        relation_types = sorted(relation for _left, _right, relation in raw_relations)
        coarse = (
            f"python:a{len(local)}:r{len(relations)}:"
            f"roles={','.join(roles)}:roots={','.join(roots)}:rels={','.join(relation_types)}"
        )
        signatures = sorted(
            [f"anchor:{_role(value)}:{type(value).__name__}" for value in local]
            + [f"relation:{value.rsplit(':', 1)[-1]}" for value in relations]
        )
        units.append(
            StructuralUnit(
                file=file,
                qualname=qualname,
                anchor_role=_role(root),
                node_type=type(root).__name__,
                coarse_key=coarse,
                fingerprint=fingerprint,
                normalized_tree=canonical[:12000],
                signatures=signatures,
                relations=relations,
                anchor_count=len(local),
                relation_count=len(relations),
                start_line=int(getattr(root, "lineno", 1)),
                end_line=int(getattr(root, "end_lineno", getattr(root, "lineno", 1))),
                source_hash=hashlib.sha256(normalized_dump(root).encode()).hexdigest(),
            )
        )
    return units


def function_nodes(source: str) -> list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]]:
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return []
    result: list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef]] = []

    def walk(body: list[ast.stmt], prefix: str = "") -> None:
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{prefix}.{node.name}" if prefix else node.name
                result.append((name, node))
            elif isinstance(node, ast.ClassDef):
                name = f"{prefix}.{node.name}" if prefix else node.name
                walk(node.body, name)

    walk(tree.body)
    return result


def source_segment(source: str, node: ast.AST) -> str:
    return ast.get_source_segment(source, node) or ""


def production_python_files(root: Path) -> list[Path]:
    ignored = {
        ".git",
        ".venv",
        "venv",
        "env",
        "build",
        "dist",
        "proofplane_generated",
        "oracle_tests",
        "site-packages",
    }
    values: list[Path] = []
    for path in root.rglob("*.py"):
        relative = path.relative_to(root)
        lowered = [part.lower() for part in relative.parts]
        if ignored.intersection(relative.parts):
            continue
        if any(part in {"test", "tests", "testing"} for part in lowered):
            continue
        if path.name.startswith("test_") or path.name.endswith("_test.py"):
            continue
        if path.stat().st_size > 160_000:
            continue
        values.append(path)
    return sorted(values)


def test_python_files(root: Path) -> list[Path]:
    ignored = {".git", ".venv", "venv", "env", "oracle_tests", "site-packages"}
    values: list[Path] = []
    for path in root.rglob("*.py"):
        relative = path.relative_to(root)
        lowered = [part.lower() for part in relative.parts]
        if ignored.intersection(relative.parts):
            continue
        is_test = (
            path.name.startswith("test_")
            or path.name.endswith("_test.py")
            or any(part in {"test", "tests", "testing"} for part in lowered)
        )
        if is_test and path.stat().st_size <= 160_000:
            values.append(path)
    return sorted(values)


def structural_similarity(left: StructuralUnit, right: StructuralUnit) -> float:
    if left.fingerprint == right.fingerprint:
        return 1.0
    left_values, right_values = Counter(left.signatures), Counter(right.signatures)
    keys = left_values.keys() | right_values.keys()
    intersection = sum(min(left_values[key], right_values[key]) for key in keys)
    union = sum(max(left_values[key], right_values[key]) for key in keys)
    jaccard = intersection / union if union else 0.0
    coarse = 1.0 if left.coarse_key == right.coarse_key else 0.0
    role = 1.0 if left.anchor_role == right.anchor_role else 0.0
    count_similarity = 1.0 - min(
        1.0,
        (
            abs(left.anchor_count - right.anchor_count)
            + abs(left.relation_count - right.relation_count)
        )
        / max(1, left.anchor_count + left.relation_count, right.anchor_count + right.relation_count),
    )
    return 0.55 * jaccard + 0.2 * coarse + 0.15 * role + 0.1 * count_similarity


def weighted_structural_similarity(
    left: StructuralUnit,
    right: StructuralUnit,
    *,
    fingerprint_frequency: Counter[str],
    coarse_frequency: Counter[str],
    signature_frequency: Counter[str],
    document_count: int,
) -> float:
    """Reward rare transferable structure instead of ubiquitous exact fragments."""

    def idf(frequency: int) -> float:
        if document_count <= 1:
            return 1.0
        return math.log((document_count + 1) / (frequency + 1)) / math.log(
            document_count + 1
        )

    left_values, right_values = Counter(left.signatures), Counter(right.signatures)
    keys = left_values.keys() | right_values.keys()
    intersection = sum(
        min(left_values[key], right_values[key]) * idf(signature_frequency[key])
        for key in keys
    )
    union = sum(
        max(left_values[key], right_values[key]) * idf(signature_frequency[key])
        for key in keys
    )
    weighted_jaccard = intersection / union if union else 0.0
    shared = left_values.keys() & right_values.keys()
    specificity = (
        sum(idf(signature_frequency[key]) for key in shared) / len(shared)
        if shared
        else 0.0
    )
    exact = (
        idf(fingerprint_frequency[left.fingerprint])
        if left.fingerprint == right.fingerprint
        else 0.0
    )
    coarse = (
        idf(coarse_frequency[left.coarse_key])
        if left.coarse_key == right.coarse_key
        else 0.0
    )
    count_similarity = 1.0 - min(
        1.0,
        (
            abs(left.anchor_count - right.anchor_count)
            + abs(left.relation_count - right.relation_count)
        )
        / max(1, left.anchor_count + left.relation_count, right.anchor_count + right.relation_count),
    )
    raw = 0.5 * weighted_jaccard + 0.25 * exact + 0.15 * coarse + 0.1 * count_similarity
    return raw * (0.2 + 0.8 * specificity)
