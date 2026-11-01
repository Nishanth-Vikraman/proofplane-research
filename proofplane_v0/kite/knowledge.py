from __future__ import annotations

import difflib
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from proofplane_v0.config import Settings
from proofplane_v0.corpus.store import CorpusStore, cache_key
from proofplane_v0.kite.api import OpenAIService
from proofplane_v0.kite.ast_units import (
    function_nodes,
    production_python_files,
    source_segment,
    structural_query_units,
    structural_similarity,
)
from proofplane_v0.kite.models import (
    BugUnitSemanticAbstraction,
    ReviewedMechanismGroup,
    StructuralUnit,
    TransferableMechanism,
)
from proofplane_v0.models import CorpusBugRecord
from proofplane_v0.retrieval.embeddings import cosine

KNOWLEDGE_VERSION = "kite-mechanism-derived-v4-group-reviewed"
UNIT_ABSTRACTION_VERSION = "kite-bug-unit-semantic-v1-paper-what-where-how"
GROUPING_VERSION = "kite-semantic-grouping-v1-complete-link"
GROUP_REVIEW_VERSION = "kite-mechanism-group-review-v1-paper-synthesis"
EXPLICIT_VERSION = UNIT_ABSTRACTION_VERSION


def _changed_reference_lines(before: str, after: str, start_line: int) -> set[int]:
    lines: set[int] = set()
    matcher = difflib.SequenceMatcher(a=before.splitlines(), b=after.splitlines())
    for tag, left_start, left_end, _right_start, _right_end in matcher.get_opcodes():
        if tag == "equal":
            continue
        if left_start == left_end:
            lines.add(start_line + max(0, left_start - 1))
        else:
            lines.update(range(start_line + left_start, start_line + left_end))
    return lines


def _attach_buggy_direction(
    reference_units: list[StructuralUnit], buggy_units: list[StructuralUnit]
) -> list[StructuralUnit]:
    paired: list[StructuralUnit] = []
    for reference_unit in reference_units:
        if not buggy_units:
            paired.append(reference_unit)
            continue
        buggy_unit = max(
            buggy_units,
            key=lambda value: structural_similarity(reference_unit, value),
        )
        paired.append(
            reference_unit.model_copy(
                update={
                    "buggy_fingerprint": buggy_unit.fingerprint,
                    "buggy_normalized_tree": buggy_unit.normalized_tree,
                    "edit_operations": [
                        "correct_to_buggy",
                        (
                            f"{reference_unit.anchor_role}:{reference_unit.node_type}"
                            f"->{buggy_unit.anchor_role}:{buggy_unit.node_type}"
                        ),
                        f"fingerprint:{reference_unit.fingerprint}->{buggy_unit.fingerprint}",
                    ],
                }
            )
        )
    return paired


def _changed_structural_units(reference: Path, candidate: Path, limit: int = 12) -> list[StructuralUnit]:
    units: list[StructuralUnit] = []
    for reference_path in production_python_files(reference):
        relative = reference_path.relative_to(reference)
        candidate_path = candidate / relative
        if not candidate_path.exists():
            continue
        before = reference_path.read_text(encoding="utf-8", errors="replace")
        after = candidate_path.read_text(encoding="utf-8", errors="replace")
        if before == after:
            continue
        reference_nodes = {name: node for name, node in function_nodes(before)}
        candidate_nodes = {name: node for name, node in function_nodes(after)}
        for qualname in sorted(reference_nodes):
            reference_source = source_segment(before, reference_nodes[qualname])
            candidate_node = candidate_nodes.get(qualname)
            candidate_source = source_segment(after, candidate_node) if candidate_node else ""
            if reference_source == candidate_source:
                continue
            reference_node = reference_nodes[qualname]
            seeds = _changed_reference_lines(
                reference_source,
                candidate_source,
                int(getattr(reference_node, "lineno", 1)),
            )
            reference_units = structural_query_units(
                relative.as_posix(),
                qualname,
                reference_node,
                seed_lines=seeds,
                limit=max(1, limit - len(units)),
            )
            buggy_units = (
                structural_query_units(
                    relative.as_posix(),
                    qualname,
                    candidate_node,
                    limit=max(12, len(reference_units)),
                )
                if candidate_node is not None
                else []
            )
            units.extend(_attach_buggy_direction(reference_units, buggy_units))
            if len(units) >= limit:
                return units[:limit]
    return units


def derive_mechanism(record: CorpusBugRecord, settings: Settings, store: CorpusStore) -> TransferableMechanism:
    key = cache_key(
        "deterministic",
        KNOWLEDGE_VERSION,
        "TransferableMechanism",
        record.model_dump_json(),
    )
    cached = store.kite_get(key)
    if cached and cached[1] == "complete":
        return TransferableMechanism.model_validate_json(cached[0])
    root = settings.data_dir / "workspaces" / record.corpus_id.replace(":", "-")
    reference, candidate = root / "reference", root / "candidate"
    structures = (
        _changed_structural_units(reference, candidate)
        if reference.exists() and candidate.exists()
        else []
    )
    hidden, visible = record.hidden_payload, record.visible_descriptor
    where_parts = (
        visible.component_roles
        + visible.operations_changed
        + visible.control_flow_features
        + visible.boundary_surfaces
        + visible.lifecycle_stages
    )
    where = "; ".join(dict.fromkeys(where_parts)) or "Python behavior matching the causal preconditions"
    what = f"{hidden.violated_invariant}. Observable consequence: {hidden.observable_symptom}"
    how = (
        "Inject the portable failure direction described by this mechanism: "
        f"{hidden.portable_prompt_summary}. Preserve target-local names and APIs."
    )
    mechanism = TransferableMechanism(
        mechanism_id=f"mechanism:{record.corpus_id}",
        corpus_id=record.corpus_id,
        source_project=record.project,
        what=what,
        where=where,
        how=how,
        preconditions=hidden.trigger_sequence[:1],
        actions=hidden.trigger_sequence[1:] or hidden.trigger_sequence,
        transition=hidden.mechanism_summary,
        violated_property=hidden.violated_invariant,
        observable=hidden.observable_symptom,
        test_shape=hidden.test_strategy,
        structural_units=structures,
    )
    store.kite_put(key, "historical_mechanism", mechanism.model_dump_json())
    return mechanism


def build_mechanisms(settings: Settings, store: CorpusStore) -> list[TransferableMechanism]:
    records = store.eligible_records("train")
    mechanisms: list[TransferableMechanism] = []
    for record in records:
        mechanism = derive_mechanism(record, settings, store)
        explicit_key = cache_key(
            settings.reasoning_model,
            EXPLICIT_VERSION,
            "TransferableMechanism",
            record.corpus_id,
        )
        explicit = store.kite_get(explicit_key)
        if explicit and explicit[1] == "complete":
            mechanism = TransferableMechanism.model_validate_json(explicit[0])
        mechanisms.append(mechanism)
    grouped_key = _grouped_cache_key(mechanisms, settings)
    grouped = store.kite_get(grouped_key)
    if grouped and grouped[1] == "complete":
        mechanisms = [
            TransferableMechanism.model_validate(value)
            for value in json.loads(grouped[0])
        ]
    if any(mechanism.source_project == "FastAPI" for mechanism in mechanisms):
        raise RuntimeError("held-out FastAPI project leaked into the train mechanism corpus")
    return mechanisms


def _local_bug_unit_evidence(
    record: CorpusBugRecord,
    mechanism: TransferableMechanism,
    settings: Settings,
    *,
    limit: int = 4,
) -> list[dict[str, object]]:
    """Return local train-pair evidence; callers must never persist it as runtime context."""
    root = settings.data_dir / "workspaces" / record.corpus_id.replace(":", "-")
    reference, candidate = root / "reference", root / "candidate"
    evidence: list[dict[str, object]] = []
    seen: set[tuple[str, str]] = set()
    for unit in mechanism.structural_units:
        identity = (unit.file, unit.qualname)
        if identity in seen:
            continue
        seen.add(identity)
        reference_path, candidate_path = reference / unit.file, candidate / unit.file
        if not reference_path.is_file() or not candidate_path.is_file():
            continue
        correct_text = reference_path.read_text(encoding="utf-8", errors="replace")
        buggy_text = candidate_path.read_text(encoding="utf-8", errors="replace")
        correct_nodes = dict(function_nodes(correct_text))
        buggy_nodes = dict(function_nodes(buggy_text))
        correct_node, buggy_node = correct_nodes.get(unit.qualname), buggy_nodes.get(unit.qualname)
        if correct_node is None:
            continue
        correct = source_segment(correct_text, correct_node)
        buggy = source_segment(buggy_text, buggy_node) if buggy_node is not None else ""
        if correct == buggy:
            continue
        evidence.append(
            {
                "opaque_unit_id": f"unit-{len(evidence) + 1:03d}",
                "correct_local_code": correct[:7000],
                "buggy_local_code": buggy[:7000],
                "normalized_structure": {
                    "coarse_key": unit.coarse_key,
                    "signatures": unit.signatures,
                    "relations": unit.relations,
                    "correct_tree": unit.normalized_tree[:4000],
                    "buggy_tree": unit.buggy_normalized_tree[:4000],
                    "edit_operations": unit.edit_operations,
                },
            }
        )
        if len(evidence) >= limit:
            break
    return evidence


def enrich_explicit_fields(
    record: CorpusBugRecord,
    mechanism: TransferableMechanism,
    service: OpenAIService,
) -> TransferableMechanism:
    """Create one paper-style semantic abstraction for a historical TRAIN bug unit."""
    if record.split != "train":
        raise ValueError("explicit WHAT/WHERE/HOW enrichment is train-only")
    hidden = record.hidden_payload
    payload: dict[str, object] = {
        "existing_portable_summary": hidden.portable_prompt_summary,
        "mechanism_summary": hidden.mechanism_summary,
        "violated_invariant": hidden.violated_invariant,
        "observable_symptom": hidden.observable_symptom,
        "trigger_sequence": hidden.trigger_sequence,
        "test_strategy": hidden.test_strategy,
        "fix_strategy": hidden.fix_strategy,
        "local_correct_buggy_evidence": _local_bug_unit_evidence(
            record, mechanism, service.settings
        ),
        "normalized_structural_evidence": [
            {
                "coarse_key": unit.coarse_key,
                "fingerprint": unit.fingerprint,
                "signatures": unit.signatures,
                "relations": unit.relations,
                "correct_fingerprint": unit.fingerprint,
                "buggy_fingerprint": unit.buggy_fingerprint,
                "edit_operations": unit.edit_operations,
            }
            for unit in mechanism.structural_units
        ],
    }
    fields = service.parse(
        schema=BugUnitSemanticAbstraction,
        purpose="kite-train-bug-unit-abstraction",
        version=UNIT_ABSTRACTION_VERSION,
        system=(
            "This is KITE-style offline semantic abstraction of one historical TRAIN bug unit. "
            "Use the local correct/buggy pair and normalized structure to produce: WHAT (the violated "
            "semantic constraint or behavioral deviation), WHERE (local applicability conditions, code "
            "shapes, and constraints), and HOW (an abstract correct-to-buggy injectable edit direction). "
            "Also state the expected buggy behavior, applicability signals, injection constraints, and "
            "representative evidence. This stage abstracts one unit; do not group, rank, or decide final "
            "transferability. Never emit repository, package, module, path, filename, commit, class, "
            "function, variable, endpoint, or exact constant names. Preserve causal and executable detail "
            "while replacing concrete identifiers with roles."
        ),
        payload=payload,
    )
    enriched = mechanism.model_copy(
        update={
            "what": fields.what,
            "where": fields.where,
            "how": fields.how,
            "expected_buggy_behavior": fields.expected_buggy_behavior,
            "representative_evidence": fields.representative_evidence,
            "derivation": "train_llm_explicit",
            "enrichment_version": UNIT_ABSTRACTION_VERSION,
            "review_status": "unreviewed",
            "review_confidence": fields.abstraction_confidence,
            "semantic_group": "",
            "rejection_reason": "",
            "supporting_bug_unit_ids": [f"{record.corpus_id}::unit-001"],
            "source_projects": [record.project],
        }
    )
    key = cache_key(
        service.settings.reasoning_model,
        EXPLICIT_VERSION,
        "TransferableMechanism",
        record.corpus_id,
    )
    service.store.kite_put(key, "historical_mechanism_explicit", enriched.model_dump_json())
    return enriched


def _semantic_groups(
    mechanisms: list[TransferableMechanism],
    vectors: list[Any],
    threshold: float,
    max_group_size: int,
) -> list[list[int]]:
    """Deterministic complete-link clustering; no member joins an incoherent group."""
    clusters: list[list[int]] = [[index] for index in range(len(mechanisms))]
    pair_scores = {
        (left, right): max(0.0, cosine(vectors[left], vectors[right]))
        for left in range(len(vectors))
        for right in range(left + 1, len(vectors))
    }
    while True:
        candidates: list[tuple[float, tuple[str, ...], int, int]] = []
        for left in range(len(clusters)):
            for right in range(left + 1, len(clusters)):
                merged = clusters[left] + clusters[right]
                if len(merged) > max_group_size:
                    continue
                cross = [
                    pair_scores[(min(a, b), max(a, b))]
                    for a in clusters[left]
                    for b in clusters[right]
                ]
                complete_link = min(cross)
                if complete_link >= threshold:
                    identity = tuple(sorted(mechanisms[index].mechanism_id for index in merged))
                    candidates.append((complete_link, identity, left, right))
        if not candidates:
            break
        _score, _identity, left, right = max(
            candidates, key=lambda value: (value[0], tuple(reversed(value[1])))
        )
        clusters[left] = sorted(clusters[left] + clusters[right])
        del clusters[right]
    return sorted(
        clusters,
        key=lambda group: tuple(mechanisms[index].mechanism_id for index in group),
    )


def _grouped_cache_key(
    mechanisms: list[TransferableMechanism], settings: Settings
) -> str:
    identity = json.dumps(
        {
            "unit_mechanisms": [value.model_dump(mode="json") for value in mechanisms],
            "model": settings.reasoning_model,
            "embedding_model": settings.embedding_model,
            "threshold": settings.kite_knowledge_group_similarity_threshold,
            "max_group_size": settings.kite_knowledge_max_group_size,
            "min_review_confidence": settings.kite_knowledge_review_min_confidence,
            "grouping_version": GROUPING_VERSION,
            "review_version": GROUP_REVIEW_VERSION,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return cache_key(
        settings.reasoning_model,
        KNOWLEDGE_VERSION,
        "TransferableMechanismGroupList",
        identity,
    )


def _group_payload(members: list[TransferableMechanism]) -> dict[str, object]:
    return {
        "historical_bug_units": [
            {
                "opaque_unit_id": f"unit-{index:03d}",
                "what": member.what,
                "where": member.where,
                "how": member.how,
                "expected_buggy_behavior": member.expected_buggy_behavior,
                "preconditions": member.preconditions,
                "actions": member.actions,
                "transition": member.transition,
                "violated_property": member.violated_property,
                "observable": member.observable,
                "test_shape": member.test_shape,
                "representative_evidence": member.representative_evidence,
                "normalized_structural_evidence": [
                    {
                        "coarse_key": unit.coarse_key,
                        "signatures": unit.signatures,
                        "relations": unit.relations,
                        "edit_operations": unit.edit_operations,
                    }
                    for unit in member.structural_units[:8]
                ],
            }
            for index, member in enumerate(members, 1)
        ]
    }


def _review_group(
    members: list[TransferableMechanism],
    service: OpenAIService,
) -> ReviewedMechanismGroup:
    return service.parse(
        schema=ReviewedMechanismGroup,
        purpose="kite-train-mechanism-group-review",
        version=GROUP_REVIEW_VERSION,
        system=(
            "You are performing KITE-style offline review of a deterministically formed semantic "
            "group of historical TRAIN bug units. Review the group as a whole. If its members support "
            "one coherent, functional, target-independent, and injectably specific mechanism, synthesize "
            "a canonical knowledge-base record: WHAT, WHERE, HOW, expected buggy behavior, causal program, "
            "test shape, semantic group, and representative evidence. Otherwise mark it non-transferable "
            "and explain whether it is unclear, non-functional, overly project-specific, or semantically "
            "incoherent. A singleton may be accepted when its evidence is independently clear. Do not add "
            "facts absent from the supplied units. Never emit concrete repository, package, module, path, "
            "filename, commit, class, function, variable, endpoint, or exact constant names."
        ),
        payload=_group_payload(members),
    )


def _synthesize_group(
    members: list[TransferableMechanism],
    review: ReviewedMechanismGroup,
    settings: Settings,
) -> TransferableMechanism:
    member_ids = sorted(member.mechanism_id for member in members)
    digest = hashlib.sha256("\n".join(member_ids).encode()).hexdigest()[:20]
    projects = sorted(
        {
            project
            for member in members
            for project in (member.source_projects or [member.source_project])
        }
    )
    supporting = [
        unit_id
        for member in members
        for unit_id in (
            member.supporting_bug_unit_ids or [f"{member.corpus_id}::unit-001"]
        )
    ]
    accepted = bool(
        review.transferable
        and review.review_confidence >= settings.kite_knowledge_review_min_confidence
    )
    structures: list[StructuralUnit] = []
    seen: set[tuple[str, str, str]] = set()
    for member in members:
        for unit in member.structural_units:
            identity = (unit.coarse_key, unit.fingerprint, unit.buggy_fingerprint)
            if identity in seen:
                continue
            seen.add(identity)
            structures.append(unit)
            if len(structures) >= 24:
                break
        if len(structures) >= 24:
            break
    return TransferableMechanism(
        mechanism_id=f"mechanism-group:{digest}",
        corpus_id=f"kite-group:{digest}",
        source_project=projects[0] if len(projects) == 1 else "multiple-train-projects",
        source_projects=projects,
        what=review.what,
        where=review.where,
        how=review.how,
        expected_buggy_behavior=review.expected_buggy_behavior,
        preconditions=review.preconditions,
        actions=review.actions,
        transition=review.transition,
        violated_property=review.violated_property,
        observable=review.observable,
        test_shape=review.test_shape,
        representative_evidence=review.representative_evidence,
        structural_units=structures,
        derivation="train_llm_grouped",
        enrichment_version=GROUP_REVIEW_VERSION,
        review_status="accepted" if accepted else "rejected",
        review_confidence=review.review_confidence,
        semantic_group=review.semantic_group,
        rejection_reason=(
            "" if accepted else review.rejection_reason or "review confidence below threshold"
        ),
        supporting_bug_unit_ids=supporting,
    )


def _write_knowledge_heartbeat(settings: Settings, payload: dict[str, object]) -> None:
    path = settings.artifacts_dir / "kite-knowledge-heartbeat.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({**payload, "updated_utc": datetime.now(UTC).isoformat()}, indent=2),
        encoding="utf-8",
    )


def build_reviewed_mechanisms(
    settings: Settings,
    store: CorpusStore,
    service: OpenAIService,
) -> list[TransferableMechanism]:
    """Build the cached paper-style unit -> group -> reviewed mechanism knowledge base."""
    records = store.eligible_records("train")
    enriched: list[TransferableMechanism] = []
    for index, record in enumerate(records, 1):
        mechanism = derive_mechanism(record, settings, store)
        key = cache_key(
            settings.reasoning_model,
            UNIT_ABSTRACTION_VERSION,
            "TransferableMechanism",
            record.corpus_id,
        )
        cached = store.kite_get(key)
        if cached and cached[1] == "complete":
            value = TransferableMechanism.model_validate_json(cached[0])
        else:
            value = enrich_explicit_fields(record, mechanism, service)
        enriched.append(value)
        _write_knowledge_heartbeat(
            settings,
            {
                "status": "unit_abstraction",
                "completed_units": index,
                "total_units": len(records),
            },
        )
    if any(value.source_project == "FastAPI" for value in enriched):
        raise RuntimeError("held-out FastAPI project leaked into knowledge construction")
    grouped_key = _grouped_cache_key(enriched, settings)
    cached_groups = store.kite_get(grouped_key)
    if cached_groups and cached_groups[1] == "complete":
        return [
            TransferableMechanism.model_validate(value)
            for value in json.loads(cached_groups[0])
        ]
    vectors = service.embed(
        [value.semantic_text for value in enriched],
        purpose="kite-train-mechanism-grouping-v1",
    )
    groups = _semantic_groups(
        enriched,
        vectors,
        settings.kite_knowledge_group_similarity_threshold,
        settings.kite_knowledge_max_group_size,
    )
    reviewed: list[TransferableMechanism] = []
    for index, group in enumerate(groups, 1):
        members = [enriched[value] for value in group]
        review = _review_group(members, service)
        reviewed.append(_synthesize_group(members, review, settings))
        _write_knowledge_heartbeat(
            settings,
            {
                "status": "group_review",
                "completed_groups": index,
                "total_groups": len(groups),
                "accepted_groups": sum(value.review_status == "accepted" for value in reviewed),
                "rejected_groups": sum(value.review_status == "rejected" for value in reviewed),
            },
        )
    store.kite_put(
        grouped_key,
        "historical_mechanism_group_list",
        json.dumps([value.model_dump(mode="json") for value in reviewed], sort_keys=True),
    )
    _write_knowledge_heartbeat(
        settings,
        {
            "status": "complete",
            "completed_units": len(enriched),
            "total_units": len(enriched),
            "completed_groups": len(reviewed),
            "total_groups": len(reviewed),
            "accepted_groups": sum(value.review_status == "accepted" for value in reviewed),
            "rejected_groups": sum(value.review_status == "rejected" for value in reviewed),
        },
    )
    return reviewed


def knowledge_manifest(mechanisms: list[TransferableMechanism]) -> dict[str, object]:
    serialized = [mechanism.model_dump(mode="json") for mechanism in mechanisms]
    return {
        "schema_version": KNOWLEDGE_VERSION,
        "count": len(mechanisms),
        "mechanism_ids": [mechanism.mechanism_id for mechanism in mechanisms],
        "content_hash": hashlib.sha256(
            json.dumps(serialized, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "unit_abstraction_version": UNIT_ABSTRACTION_VERSION,
        "grouping_version": GROUPING_VERSION,
        "group_review_version": GROUP_REVIEW_VERSION,
        "supporting_bug_units": len(
            {value for mechanism in mechanisms for value in mechanism.supporting_bug_unit_ids}
        ),
        "accepted_groups": sum(value.review_status == "accepted" for value in mechanisms),
        "rejected_groups": sum(value.review_status == "rejected" for value in mechanisms),
        "explicit_llm_enrichment_available": True,
        "explicit_llm_enrichment_used": sum(
            mechanism.derivation in {"train_llm_explicit", "train_llm_grouped"}
            for mechanism in mechanisms
        ),
        "group_review_used": sum(
            mechanism.derivation == "train_llm_grouped" for mechanism in mechanisms
        ),
    }
