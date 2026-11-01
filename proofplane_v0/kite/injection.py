from __future__ import annotations

import ast
import difflib
import hashlib
import json
import os
import shutil
import textwrap
from pathlib import Path

from proofplane_v0.config import Settings
from proofplane_v0.corpus.store import CorpusStore, cache_key
from proofplane_v0.evaluation.docker_runner import run_test, run_workspace_command
from proofplane_v0.kite.api import OpenAIService
from proofplane_v0.kite.models import (
    InjectionPlan,
    InjectionValidation,
    RetrievalMatch,
    StructuralUnit,
)
from proofplane_v0.retrieval.embeddings import cosine

PLAN_VERSION = "kite-injection-plan-v4-anchor-id-alignment"
VALIDATOR_VERSION = "kite-injection-validator-v4-full-unit-fragments"
REQUIRED_INJECTION_GATES = (
    "target_locality",
    "anchor_integrity",
    "fragment_anchor_alignment",
    "exact_replacement",
    "syntax_validity",
    "non_trivial_change",
    "no_import_or_signature_change",
    "behavioral_probe_present",
    "behavioral_probe_no_source_introspection",
    "syntax_and_compilation_validity",
    "repository_compatibility",
    "executable_variant",
    "behavioral_non_triviality",
)


def available_local_anchors(match: RetrievalMatch) -> list[tuple[str, StructuralUnit]]:
    units = match.target_unit.structural_queries or [match.target_unit.structural]
    return [(f"anchor-{index:03d}", value) for index, value in enumerate(units)]


def injection_alignment_score(
    match: RetrievalMatch, plan: InjectionPlan, service: OpenAIService
) -> float:
    """Score explicit WHAT/WHERE/HOW transfer without any hidden target information."""
    by_dimension: dict[str, list[str]] = {"what": [], "where": [], "how": []}
    for value in plan.mechanism_alignment:
        if value.dimension in by_dimension:
            by_dimension[value.dimension].append(
                f"{value.historical_element}\n{value.target_evidence}\n{value.edit_effect}"
            )
    if any(not by_dimension[value] for value in ("what", "where", "how")):
        return 0.0
    mechanism = match.mechanism
    historical = [mechanism.what, mechanism.where, mechanism.how]
    proposed = [
        "\n".join(by_dimension[dimension])
        for dimension in ("what", "where", "how")
    ]
    vectors = service.embed(
        [*historical, *proposed], purpose="kite-injection-alignment-v1"
    )
    scores = [max(0.0, cosine(vectors[index], vectors[index + 3])) for index in range(3)]
    return 0.4 * scores[0] + 0.25 * scores[1] + 0.35 * scores[2]


def propose_injection(match: RetrievalMatch, service: OpenAIService) -> InjectionPlan:
    unit, mechanism = match.target_unit, match.mechanism
    anchors = available_local_anchors(match)
    payload: dict[str, object] = {
        "historical_transferable_mechanism": {
            "mechanism_id": mechanism.mechanism_id,
            "what": mechanism.what,
            "where": mechanism.where,
            "how": mechanism.how,
            "expected_buggy_behavior": mechanism.expected_buggy_behavior,
            "representative_evidence": mechanism.representative_evidence,
            "supporting_bug_unit_count": len(mechanism.supporting_bug_unit_ids),
            "causal_program": {
                "preconditions": mechanism.preconditions,
                "actions": mechanism.actions,
                "transition": mechanism.transition,
                "violated_property": mechanism.violated_property,
                "observable": mechanism.observable,
                "test_shape": mechanism.test_shape,
            },
            "semantic_group": mechanism.semantic_group,
            "structural_edit_directions": [
                {
                    "correct_fingerprint": value.fingerprint,
                    "buggy_fingerprint": value.buggy_fingerprint,
                    "edit_operations": value.edit_operations,
                    "relations": value.relations,
                }
                for value in mechanism.structural_units
            ],
        },
        "reference_target_unit": {
            "file": unit.file,
            "qualname": unit.qualname,
            "focal_start_line": unit.structural.start_line,
            "focal_end_line": unit.structural.end_line,
            "signature": unit.signature,
            "docstring": unit.docstring,
            "source": unit.source,
            "related_pre_existing_tests": unit.related_tests,
            "semantic_profile": {
                "what": unit.semantic_what,
                "where": unit.semantic_where,
                "how": unit.semantic_how,
            },
            "repository_context": unit.repository_context.model_dump(mode="json"),
            "available_local_anchors": [
                {
                    "anchor_id": anchor_id,
                    "start_line": value.start_line,
                    "end_line": value.end_line,
                    "coarse_key": value.coarse_key,
                    "anchor_role": value.anchor_role,
                    "node_type": value.node_type,
                    "relations": value.relations,
                }
                for anchor_id, value in anchors
            ],
        },
    }
    return service.parse(
        schema=InjectionPlan,
        purpose="kite-injection-plan",
        version=PLAN_VERSION,
        system=(
            "Propose one target-local synthetic Python bug that instantiates the historical causal "
            "mechanism in the supplied REFERENCE target unit. The historical mechanism is analogy, "
            "not a fact about the target. Return a local replacement plan only, never a rewritten "
            "file or repository. Select exactly one available_local_anchor and return only its opaque "
            "anchor_id in target_anchor_id; never reproduce hashes or invent an anchor. Set before to the "
            "smallest useful "
            "contiguous statement or expression fragment copied byte-for-byte from the supplied focal "
            "method; it must occur exactly once in that method. Set after to its local replacement with "
            "matching indentation. Change only this focal method, preserve its public signature, imports, "
            "target-local names, and surrounding APIs. Do not edit tests, dependencies, configuration, "
            "or add imports. The edit "
            "must be syntactically valid after substitution and non-trivial. A control header fragment such "
            "as `if condition:` is allowed when both before and after include the colon and retain the existing "
            "body. mechanism_alignment must contain separate WHAT, WHERE, and HOW entries that quote the "
            "historical abstraction, cite concrete target evidence, and explain how this exact edit preserves "
            "that causal meaning; do not substitute a generic mutation. trigger_recipe must explain how a test can make "
            "the reference pass and this synthetic variant fail. behavioral_probe must be one raw, "
            "self-contained executable pytest file implementing that distinction with visible target APIs: "
            "it must PASS on the supplied reference and FAIL on the proposed synthetic rewrite. Do not use "
            "Markdown, edit production code, inspect Git history, source text, files, ASTs, bytecode, or "
            "function implementation metadata, and do not refer to any hidden real bug. The probe must "
            "distinguish observable behavior through public or repository-visible APIs. Only import and call "
            "APIs demonstrated by the focal source, repository_context, or related_pre_existing_tests; do not "
            "assume modern package re-exports exist. Prefer the exact import paths and setup patterns shown."
        ),
        payload=payload,
    )


def _safe_copy(reference: Path, target: Path, synthetic_root: Path) -> None:
    synthetic_root.mkdir(parents=True, exist_ok=True)
    resolved_target = target.resolve()
    resolved_root = synthetic_root.resolve()
    if not resolved_target.is_relative_to(resolved_root) or resolved_target == resolved_root:
        raise ValueError("synthetic workspace target escaped its dedicated root")
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    def linked_copy(source: str, destination: str) -> str:
        try:
            os.link(source, destination)
            return destination
        except OSError:
            return shutil.copy2(source, destination)

    shutil.copytree(
        reference,
        target,
        copy_function=linked_copy,
        ignore=shutil.ignore_patterns(
            ".git", "__pycache__", ".pytest_cache", "proofplane_generated", "*.pyc"
        ),
    )


def _ast_changed(before: str, after: str) -> bool:
    try:
        return ast.dump(ast.parse(textwrap.dedent(before)), include_attributes=False) != ast.dump(
            ast.parse(textwrap.dedent(after)), include_attributes=False
        )
    except SyntaxError:
        return False


def _behavioral_probe_is_black_box(source: str) -> bool:
    """Reject probes that distinguish variants by reading implementation text."""
    tree = ast.parse(source)
    forbidden_modules = {"ast", "dis", "inspect", "subprocess", "tokenize"}
    forbidden_names = {"compile", "eval", "exec", "open"}
    forbidden_attributes = {
        "__code__",
        "__file__",
        "co_code",
        "co_consts",
        "getsource",
        "getsourcefile",
        "read_bytes",
        "read_text",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import) and any(
            value.name.split(".", 1)[0] in forbidden_modules for value in node.names
        ):
            return False
        if isinstance(node, ast.ImportFrom) and (
            node.module or ""
        ).split(".", 1)[0] in forbidden_modules:
            return False
        if isinstance(node, ast.Name) and node.id in forbidden_names:
            return False
        if isinstance(node, ast.Attribute) and node.attr in forbidden_attributes:
            return False
    return True


def _module_candidates(file: str) -> list[str]:
    path = Path(file)
    parts = list(path.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    values: list[str] = []
    for index in range(len(parts)):
        candidate = parts[index:]
        if candidate and all(part.isidentifier() for part in candidate):
            values.append(".".join(candidate))
    return values


def _repository_import_probe(file: str) -> list[str]:
    modules = _module_candidates(file)
    script = (
        "import importlib,json,sys\n"
        f"modules=json.loads({json.dumps(json.dumps(modules))})\n"
        "errors=[]\n"
        "for name in modules:\n"
        "    try:\n"
        "        importlib.import_module(name)\n"
        "        print('kite-import-ok:'+name)\n"
        "        raise SystemExit(0)\n"
        "    except Exception as error:\n"
        "        errors.append(name+':'+type(error).__name__+':'+str(error))\n"
        "print('\\n'.join(errors), file=sys.stderr)\n"
        "raise SystemExit(1)\n"
    )
    return ["python", "-c", script]


def _compilation_probe(file: str) -> list[str]:
    script = (
        "import pathlib; "
        f"p=pathlib.Path({json.dumps(file)}); "
        "compile(p.read_text(encoding='utf-8'), str(p), 'exec'); "
        "print('kite-compile-ok')"
    )
    return ["python", "-c", script]


def apply_and_validate(
    *,
    match: RetrievalMatch,
    plan: InjectionPlan,
    reference: Path,
    synthetic_root: Path,
    image: str,
    settings: Settings,
    store: CorpusStore,
) -> InjectionValidation:
    unit = match.target_unit
    plan_identity = hashlib.sha256(plan.model_dump_json().encode()).hexdigest()
    workspace = synthetic_root / unit.unit_id.split(":", 1)[0].replace(":", "-") / plan_identity[:20]
    key = cache_key(
        "deterministic",
        VALIDATOR_VERSION,
        "InjectionValidation",
        f"{unit.content_hash}:{plan.model_dump_json()}:{image}",
    )
    cached = store.kite_get(key)
    if cached and cached[1] == "complete":
        validation = InjectionValidation.model_validate_json(cached[0])
        if not validation.accepted or (Path(validation.synthetic_workspace) / ".kite-ready").exists():
            return validation

    gates = dict.fromkeys(REQUIRED_INJECTION_GATES, False)
    reasons: list[str] = []
    anchor_map = dict(available_local_anchors(match))
    anchor = anchor_map.get(plan.target_anchor_id)
    gates["target_locality"] = (
        anchor is not None
        and not Path(unit.file).is_absolute()
        and ".." not in Path(unit.file).parts
        and not any(part.lower() in {"test", "tests"} for part in Path(unit.file).parts)
    )
    gates["anchor_integrity"] = anchor is not None
    gates["exact_replacement"] = (
        plan.before != plan.after and unit.source.count(plan.before) == 1
    )
    if gates["exact_replacement"] and anchor is not None:
        relative_start = unit.source.index(plan.before)
        before_start_line = unit.structural.start_line + unit.source[:relative_start].count("\n")
        before_end_line = before_start_line + plan.before.count("\n")
        gates["fragment_anchor_alignment"] = not (
            before_end_line < anchor.start_line or before_start_line > anchor.end_line
        )
    gates["behavioral_probe_present"] = bool(plan.behavioral_probe.strip())
    gates["behavioral_probe_no_source_introspection"] = _behavioral_probe_is_black_box(
        plan.behavioral_probe
    )
    try:
        modified_unit = unit.source.replace(plan.before, plan.after, 1)
        original_unit_tree = ast.parse(textwrap.dedent(unit.source))
        modified_unit_tree = ast.parse(textwrap.dedent(modified_unit))
        gates["syntax_validity"] = True
        gates["non_trivial_change"] = _ast_changed(unit.source, modified_unit)
        before_function = next(
            (
                node
                for node in original_unit_tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            ),
            None,
        )
        after_function = next(
            (
                node
                for node in modified_unit_tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            ),
            None,
        )
        gates["no_import_or_signature_change"] = bool(
            before_function
            and after_function
            and before_function.name == after_function.name
            and ast.dump(before_function.args, include_attributes=False)
            == ast.dump(after_function.args, include_attributes=False)
            and [
                ast.dump(node, include_attributes=False)
                for node in ast.walk(original_unit_tree)
                if isinstance(node, (ast.Import, ast.ImportFrom))
            ]
            == [
                ast.dump(node, include_attributes=False)
                for node in ast.walk(modified_unit_tree)
                if isinstance(node, (ast.Import, ast.ImportFrom))
            ]
        )
    except SyntaxError as error:
        reasons.append(f"modified focal unit syntax error: {error}")

    static_gates = {
        "target_locality",
        "anchor_integrity",
        "fragment_anchor_alignment",
        "exact_replacement",
        "syntax_validity",
        "non_trivial_change",
        "no_import_or_signature_change",
        "behavioral_probe_present",
        "behavioral_probe_no_source_introspection",
    }
    if not all(gates[name] for name in static_gates):
        reasons.extend(name for name, passed in gates.items() if not passed)
        validation = InjectionValidation(accepted=False, gates=gates, reasons=sorted(set(reasons)))
        store.kite_put(key, "injection_validation", validation.model_dump_json())
        return validation

    target_file = reference / unit.file
    if not target_file.is_file():
        reasons.append("target file is absent from reference workspace")
    else:
        original = target_file.read_text(encoding="utf-8", errors="replace")
        if original.count(unit.source) != 1:
            reasons.append("focal method source does not occur exactly once in target file")
        else:
            modified_unit = unit.source.replace(plan.before, plan.after, 1)
            modified = original.replace(unit.source, modified_unit, 1)
            try:
                ast.parse(modified)
                _safe_copy(reference, workspace, synthetic_root)
                synthetic_target = workspace / unit.file
                # Break the hard link before writing so the reference checkout is immutable.
                synthetic_target.unlink()
                synthetic_target.write_text(modified, encoding="utf-8")
                reference_compilation = run_workspace_command(
                    reference,
                    _compilation_probe(unit.file),
                    timeout=settings.execution_timeout,
                    image=image,
                )
                synthetic_compilation = run_workspace_command(
                    workspace,
                    _compilation_probe(unit.file),
                    timeout=settings.execution_timeout,
                    image=image,
                )
                gates["syntax_and_compilation_validity"] = (
                    reference_compilation.status == "pass"
                    and synthetic_compilation.status == "pass"
                )
                reference_repository = run_workspace_command(
                    reference,
                    _repository_import_probe(unit.file),
                    timeout=settings.execution_timeout,
                    image=image,
                )
                synthetic_repository = run_workspace_command(
                    workspace,
                    _repository_import_probe(unit.file),
                    timeout=settings.execution_timeout,
                    image=image,
                )
                gates["repository_compatibility"] = (
                    reference_repository.status == "pass"
                    and synthetic_repository.status == "pass"
                )
                gates["executable_variant"] = (
                    synthetic_compilation.status == "pass"
                    and synthetic_repository.status == "pass"
                )
                probe_relative = (
                    "proofplane_generated/kite/injection_probe/"
                    f"test_{plan_identity[:16]}.py"
                )
                probe_paths: list[Path] = []
                for probe_workspace in (reference, workspace):
                    probe_path = probe_workspace / probe_relative
                    probe_path.parent.mkdir(parents=True, exist_ok=True)
                    probe_path.write_text(plan.behavioral_probe, encoding="utf-8")
                    probe_paths.append(probe_path)
                try:
                    behavioral_reference = run_test(
                        reference,
                        probe_relative,
                        settings.execution_timeout,
                        image=image,
                    )
                    behavioral_synthetic = run_test(
                        workspace,
                        probe_relative,
                        settings.execution_timeout,
                        image=image,
                    )
                finally:
                    for probe_path in probe_paths:
                        probe_path.unlink(missing_ok=True)
                gates["behavioral_non_triviality"] = (
                    behavioral_reference.status == "pass"
                    and behavioral_synthetic.status == "fail"
                )
                if not gates["syntax_and_compilation_validity"]:
                    reasons.append(
                        "compilation gate failed: "
                        f"reference={reference_compilation.status}, "
                        f"synthetic={synthetic_compilation.status}"
                    )
                if not gates["repository_compatibility"]:
                    reasons.append(
                        "repository import gate failed: "
                        f"reference={reference_repository.status}, "
                        f"synthetic={synthetic_repository.status}"
                    )
                if not gates["behavioral_non_triviality"]:
                    reasons.append(
                        "behavioral probe did not distinguish variants: "
                        f"reference={behavioral_reference.status}, "
                        f"synthetic={behavioral_synthetic.status}"
                    )
                patch = "\n".join(
                    difflib.unified_diff(
                        original.splitlines(),
                        modified.splitlines(),
                        fromfile=f"reference/{unit.file}",
                        tofile=f"synthetic/{unit.file}",
                        lineterm="",
                    )
                )
                accepted = all(gates.values())
                if accepted:
                    (workspace / ".kite-ready").write_text(plan_identity, encoding="utf-8")
                validation = InjectionValidation(
                    accepted=accepted,
                    gates=gates,
                    reasons=reasons,
                    patch=patch,
                    synthetic_workspace=str(workspace),
                    execution=synthetic_repository,
                    reference_compilation=reference_compilation,
                    synthetic_compilation=synthetic_compilation,
                    reference_repository_probe=reference_repository,
                    synthetic_repository_probe=synthetic_repository,
                    behavioral_reference=behavioral_reference,
                    behavioral_synthetic=behavioral_synthetic,
                )
                store.kite_put(key, "injection_validation", validation.model_dump_json())
                return validation
            except SyntaxError as error:
                reasons.append(f"full-file syntax error: {error}")

    validation = InjectionValidation(accepted=False, gates=gates, reasons=reasons)
    store.kite_put(key, "injection_validation", validation.model_dump_json())
    return validation
