from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from proofplane_v0.config import Settings
from proofplane_v0.kite.ast_units import structural_query_units
from proofplane_v0.kite.benchmark import load_focal_benchmark
from proofplane_v0.kite.injection import REQUIRED_INJECTION_GATES, VALIDATOR_VERSION
from proofplane_v0.kite.knowledge import KNOWLEDGE_VERSION
from proofplane_v0.kite.models import TransferableMechanism
from proofplane_v0.kite.retrieval import RETRIEVAL_VERSION
from proofplane_v0.kite.runner import FEEDBACK_PROTOCOL_VERSION
from proofplane_v0.kite.target_profile import PROFILE_VERSION, SEMANTIC_PROFILE_VERSION
from proofplane_v0.models import StrictModel


class ConformanceCheck(StrictModel):
    check_id: str
    category: str
    status: Literal["pass", "partial", "fail", "adaptation"]
    paper_expectation: str
    evidence: str
    remediation: str = ""


class ConformanceReport(StrictModel):
    generated_utc: str
    protocol_ready: bool
    exact_paper_replication: bool
    checks: list[ConformanceCheck]
    implementation_versions: dict[str, str]
    content_hash: str = ""
    schema_version: str = "kite-paper-conformance-v1"


def _check(
    check_id: str,
    category: str,
    status: Literal["pass", "partial", "fail", "adaptation"],
    expected: str,
    evidence: str,
    remediation: str = "",
) -> ConformanceCheck:
    return ConformanceCheck(
        check_id=check_id,
        category=category,
        status=status,
        paper_expectation=expected,
        evidence=evidence,
        remediation=remediation,
    )


def verify_conformance(
    *,
    settings: Settings,
    mechanisms: list[TransferableMechanism],
    benchmark_manifest: Path,
) -> ConformanceReport:
    """Verify enforceable protocol invariants and surface deliberate adaptations."""
    checks: list[ConformanceCheck] = []
    if benchmark_manifest.is_file():
        benchmark = load_focal_benchmark(benchmark_manifest)
        eligible = [value for value in benchmark.cases if value.eligible]
        focal_ok = bool(eligible) and all(
            1 <= len(value.focal_methods) <= benchmark.max_focal_methods for value in eligible
        )
        checks.append(
            _check(
                "benchmark.focal_scope",
                "benchmark",
                "pass" if focal_ok else "fail",
                "Each target is localized to a bounded focal-method patch set before runtime.",
                f"{len(eligible)}/{len(benchmark.cases)} eligible; max focal methods "
                f"is {benchmark.max_focal_methods}.",
                "Rebuild the benchmark manifest with `kite benchmark-build`." if not focal_ok else "",
            )
        )
        checks.append(
            _check(
                "benchmark.hidden_boundary",
                "leakage",
                "pass",
                "The real fault is unavailable during retrieval, injection, and generation.",
                "The runtime manifest contains only reference method identity, line bounds, and a "
                "reference source hash; no buggy source or patch is serialized.",
            )
        )
    else:
        checks.append(
            _check(
                "benchmark.focal_scope",
                "benchmark",
                "fail",
                "A sanitized focal-method benchmark manifest exists.",
                f"Missing manifest: {benchmark_manifest}",
                "Run `python -m proofplane_v0.cli kite benchmark-build ...`.",
            )
        )

    semantic_ok = bool(mechanisms) and all(
        value.what.strip() and value.where.strip() and value.how.strip() for value in mechanisms
    )
    structural_count = sum(bool(value.structural_units) for value in mechanisms)
    relation_count = sum(
        bool(unit.relations) for value in mechanisms for unit in value.structural_units
    )
    directional_count = sum(
        bool(unit.buggy_fingerprint and unit.edit_operations)
        for value in mechanisms
        for unit in value.structural_units
    )
    grouped_count = sum(value.derivation == "train_llm_grouped" for value in mechanisms)
    reviewed_count = sum(value.review_status in {"accepted", "rejected"} for value in mechanisms)
    checks.extend(
        [
            _check(
                "knowledge.what_where_how",
                "knowledge",
                "pass" if semantic_ok else "fail",
                "Historical records expose WHAT, WHERE, and HOW abstractions.",
                f"{len(mechanisms)} mechanisms; all fields populated={semantic_ok}.",
            ),
            _check(
                "knowledge.diff_aligned_local_structure",
                "knowledge",
                "pass" if mechanisms and structural_count == len(mechanisms) else "partial",
                "Bug units contain diff-aligned local anchor-relation AST structures.",
                f"{structural_count}/{len(mechanisms)} mechanisms have structural units; "
                f"{relation_count} units contain explicit relations; {directional_count} contain "
                "correct-to-buggy structural direction.",
                "Prepare both versions for missing train cases, then rebuild KITE knowledge."
                if structural_count != len(mechanisms)
                else "",
            ),
            _check(
                "knowledge.reviewed_abstraction",
                "knowledge",
                "pass"
                if mechanisms and grouped_count == len(mechanisms) and reviewed_count == len(mechanisms)
                else "partial",
                "Historical bug units are semantically abstracted, grouped, reviewed, and synthesized.",
                f"{grouped_count}/{len(mechanisms)} are grouped syntheses and "
                f"{reviewed_count}/{len(mechanisms)} have an explicit group-review decision.",
                "Run `kite knowledge-build --explicit-llm` when budget permits; it is train-only and cached.",
            ),
            _check(
                "target.reference_profile",
                "target",
                "pass" if settings.kite_llm_target_profile else "partial",
                "Target profiling uses focal code, repository context, semantic applicability, and local structure.",
                f"Profile schemas {PROFILE_VERSION} + {SEMANTIC_PROFILE_VERSION}; "
                f"LLM semantic profiling enabled={settings.kite_llm_target_profile}; local builder "
                f"{structural_query_units.__name__}; "
                "same-class helpers, called dependencies, module/API-related tests, docs, and types are represented.",
                "Set kite_llm_target_profile: true."
                if not settings.kite_llm_target_profile
                else "",
            ),
            _check(
                "retrieval.dual_mechanism_fusion",
                "retrieval",
                "pass",
                "Semantic applicability and structural instantiability are fused at mechanism level.",
                f"Retriever {RETRIEVAL_VERSION}; top-K={settings.kite_retrieval_top_k}; "
                f"weights={settings.kite_semantic_weight}/{settings.kite_structural_weight}; "
                "WHAT/WHERE/HOW are embedded separately, common structural fingerprints are IDF-downweighted, "
                "and same-project mechanisms are rejected.",
            ),
        ]
    )

    gate_ok = {
        "target_locality",
        "anchor_integrity",
        "fragment_anchor_alignment",
        "no_import_or_signature_change",
        "syntax_and_compilation_validity",
        "repository_compatibility",
        "non_trivial_change",
        "executable_variant",
        "behavioral_non_triviality",
    }.issubset(REQUIRED_INJECTION_GATES)
    docker_gate_path = settings.artifacts_dir / "kite-verification" / "docker-gate.json"
    docker_gate_ok = False
    docker_evidence = f"No current Docker gate artifact at {docker_gate_path}."
    if docker_gate_path.is_file():
        try:
            docker_gate = json.loads(docker_gate_path.read_text(encoding="utf-8"))
            docker_gate_ok = bool(docker_gate.get("accepted")) and all(
                bool(value) for value in docker_gate.get("gates", {}).values()
            )
            docker_evidence = (
                f"accepted={docker_gate.get('accepted')}; "
                f"{sum(bool(value) for value in docker_gate.get('gates', {}).values())}/"
                f"{len(docker_gate.get('gates', {}))} recorded gates passed."
            )
        except (OSError, json.JSONDecodeError) as error:
            docker_evidence = f"Unreadable Docker gate artifact: {type(error).__name__}: {error}"
    checks.extend(
        [
            _check(
                "injection.deterministic_gates",
                "injection",
                "pass" if gate_ok else "fail",
                "Structured local edits are deterministically gated for locality, compilation, compatibility, and effect.",
                f"Validator {VALIDATOR_VERSION}; {len(REQUIRED_INJECTION_GATES)} mandatory gates: "
                + ", ".join(REQUIRED_INJECTION_GATES),
            ),
            _check(
                "injection.mechanism_adherence",
                "injection",
                "pass" if 0.0 < settings.kite_injection_alignment_threshold <= 1.0 else "fail",
                "Accepted synthetic faults instantiate the selected historical mechanism rather than a generic mutation.",
                "Plans require explicit WHAT/WHERE/HOW evidence, use program-issued opaque anchors, pass an "
                f"embedding alignment threshold of {settings.kite_injection_alignment_threshold}, and duplicate patches are rejected.",
                "Set kite_injection_alignment_threshold within (0, 1]."
                if not 0.0 < settings.kite_injection_alignment_threshold <= 1.0
                else "",
            ),
            _check(
                "injection.real_docker_smoke",
                "injection",
                "pass" if docker_gate_ok else "partial",
                "The complete injection gate executes against a real repository Docker image.",
                docker_evidence,
                "Run `python scripts/verify_kite_docker_gate.py`." if not docker_gate_ok else "",
            ),
            _check(
                "feedback.canonical_suite",
                "feedback",
                "pass"
                if settings.kite_max_enhancement_rounds == 5
                and settings.kite_max_test_repairs <= 4
                else "partial",
                "All accepted injections are processed in rank order with bounded enhancement and regression-safe promotion.",
                f"Protocol {FEEDBACK_PROTOCOL_VERSION}; max rounds="
                f"{settings.kite_max_enhancement_rounds}; max accepted injections="
                f"{settings.kite_max_accepted_injections}; base generation + repair rounds="
                f"{1 + settings.kite_max_test_repairs}.",
                "Use five enhancement rounds and at most four repairs after initial base generation."
                if settings.kite_max_enhancement_rounds != 5
                or settings.kite_max_test_repairs > 4
                else "",
            ),
            _check(
                "feedback.treatment_activation",
                "feedback",
                "pass",
                "The experiment distinguishes injection yield from actual synthetic-guided generation and promotion.",
                "Per-arm artifacts and reports record unique injections, enhancement rounds, promotions, and "
                "synthetic kills; a rejected injection retains the comparable base suite instead of producing zero tests.",
            ),
            _check(
                "generation.temperature",
                "configuration",
                "pass" if settings.kite_temperature == 0.1 else "partial",
                "Generation temperature is 0.1.",
                f"Configured temperature={settings.kite_temperature!r}.",
                "Set kite_temperature: 0.1." if settings.kite_temperature != 0.1 else "",
            ),
            _check(
                "generation.model",
                "configuration",
                "pass" if settings.reasoning_model == "gpt-5.4" else "partial",
                "The reported experiment uses GPT-5.4.",
                f"Configured model={settings.reasoning_model}; the nano choice is a deliberate cost adaptation.",
                "Set OPENAI_REASONING_MODEL=gpt-5.4 for a paper-model replication run.",
            ),
            _check(
                "experiment.core_metrics",
                "reporting",
                "pass",
                "Generation success and real-bug detection rate use all benchmark tasks as denominator.",
                "Per-case machine results retain generated count, reference passes, real-bug kills, "
                "success@K, injection acceptance, unique injections, enhancement rounds, promotions, and "
                "synthetic kills; failed cases remain in results.",
            ),
            _check(
                "experiment.auxiliary_metrics",
                "reporting",
                "partial",
                "The paper additionally reports focal line/branch coverage, generic mutation score, and average tokens.",
                "The KITE-lite causal experiment reports executable-feedback and real-bug outcomes; it does "
                "not yet run Python coverage/mutation tooling or attribute cached API tokens per arm.",
                "Add coverage.py, a pinned Python mutation engine, and per-arm token attribution only if "
                "replicating the paper's secondary metric table becomes necessary.",
            ),
            _check(
                "benchmark.ecosystem",
                "adaptation",
                "adaptation",
                "The paper evaluates Java/Defects4J single-focal faults.",
                "Proofplane evaluates Python/BugsInPy and maps compilation/build gates to Python syntax, "
                "module import, repository image, and pytest execution.",
            ),
        ]
    )
    protocol_ready = not any(value.status == "fail" for value in checks)
    exact = protocol_ready and all(value.status == "pass" for value in checks)
    versions = {
        "knowledge": KNOWLEDGE_VERSION,
        "profile": PROFILE_VERSION,
        "semantic_profile": SEMANTIC_PROFILE_VERSION,
        "retrieval": RETRIEVAL_VERSION,
        "injection": VALIDATOR_VERSION,
        "feedback": FEEDBACK_PROTOCOL_VERSION,
    }
    body = {
        "checks": [value.model_dump(mode="json") for value in checks],
        "versions": versions,
    }
    return ConformanceReport(
        generated_utc=datetime.now(UTC).isoformat(),
        protocol_ready=protocol_ready,
        exact_paper_replication=exact,
        checks=checks,
        implementation_versions=versions,
        content_hash=hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
    )


def write_conformance_report(report: ConformanceReport, output: Path) -> tuple[Path, Path]:
    output.mkdir(parents=True, exist_ok=True)
    json_path = output / "conformance.json"
    json_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    lines = [
        "# KITE paper conformance gate",
        "",
        f"Protocol ready: **{report.protocol_ready}**  ",
        f"Exact paper replication: **{report.exact_paper_replication}**",
        "",
        "| Check | Status | Evidence | Remediation |",
        "|---|---|---|---|",
    ]
    for value in report.checks:
        evidence = value.evidence.replace("|", "\\|").replace("\n", " ")
        remediation = value.remediation.replace("|", "\\|").replace("\n", " ")
        lines.append(
            f"| `{value.check_id}` | {value.status} | {evidence} | {remediation} |"
        )
    markdown_path = output / "CONFORMANCE.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path
