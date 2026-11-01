from __future__ import annotations

import json
from collections import Counter
from typing import Literal

from proofplane_v0.config import Settings
from proofplane_v0.corpus.store import CorpusStore, cache_key
from proofplane_v0.kite.api import OpenAIService
from proofplane_v0.kite.ast_units import weighted_structural_similarity
from proofplane_v0.kite.models import (
    ReferenceTargetProfile,
    RetrievalMatch,
    TransferableMechanism,
)
from proofplane_v0.retrieval.embeddings import cosine

RETRIEVAL_VERSION = "kite-dual-retrieval-v4-idf-dimension-aligned"


def retrieve(
    profile: ReferenceTargetProfile,
    mechanisms: list[TransferableMechanism],
    service: OpenAIService,
    store: CorpusStore,
    settings: Settings,
    top_k: int | None = None,
) -> list[RetrievalMatch]:
    if not profile.units:
        raise RuntimeError(f"reference-only profiler found no Python units for {profile.case_id}")
    eligible = [
        m
        for m in mechanisms
        if profile.project not in (m.source_projects or [m.source_project])
        and m.review_status != "rejected"
    ]
    if not eligible:
        raise RuntimeError("no cross-project historical mechanisms are available")
    count = top_k or settings.kite_retrieval_top_k
    identity = json.dumps(
        {
            "profile": profile.content_hash,
            "mechanisms": [m.mechanism_id + ":" + m.enrichment_version for m in eligible],
            "embedding_model": settings.embedding_model,
            "semantic_weight": settings.kite_semantic_weight,
            "structural_weight": settings.kite_structural_weight,
            "top_k": count,
        },
        sort_keys=True,
    )
    key = cache_key(settings.embedding_model, RETRIEVAL_VERSION, "RetrievalMatchList", identity)
    cached = store.kite_get(key)
    if cached and cached[1] == "complete":
        values = json.loads(cached[0])
        return [RetrievalMatch.model_validate(value) for value in values]

    mechanism_vectors = service.embed(
        [
            value
            for mechanism in eligible
            for value in (mechanism.what, mechanism.where, mechanism.how)
        ],
        purpose="kite-mechanism-dimension-embedding-v2",
    )
    target_vectors = service.embed(
        [
            value
            for unit in profile.units
            for value in (unit.semantic_what, unit.semantic_where, unit.semantic_how)
        ],
        purpose="kite-target-dimension-embedding-v2",
    )
    fingerprint_frequency: Counter[str] = Counter()
    coarse_frequency: Counter[str] = Counter()
    signature_frequency: Counter[str] = Counter()
    for mechanism in eligible:
        fingerprint_frequency.update({value.fingerprint for value in mechanism.structural_units})
        coarse_frequency.update({value.coarse_key for value in mechanism.structural_units})
        signature_frequency.update(
            {signature for value in mechanism.structural_units for signature in value.signatures}
        )
    scored: list[tuple[float, float, float, int, int]] = []
    for mechanism_index, mechanism in enumerate(eligible):
        mechanism_dimensions = mechanism_vectors[
            mechanism_index * 3 : mechanism_index * 3 + 3
        ]
        for unit_index, unit in enumerate(profile.units):
            target_dimensions = target_vectors[unit_index * 3 : unit_index * 3 + 3]
            dimension_scores = [
                max(0.0, cosine(left, right))
                for left, right in zip(
                    mechanism_dimensions, target_dimensions, strict=True
                )
            ]
            semantic = (
                0.45 * dimension_scores[0]
                + 0.3 * dimension_scores[1]
                + 0.25 * dimension_scores[2]
            )
            query_structures = unit.structural_queries or [unit.structural]
            structural = max(
                (
                    weighted_structural_similarity(
                        source_unit,
                        query_unit,
                        fingerprint_frequency=fingerprint_frequency,
                        coarse_frequency=coarse_frequency,
                        signature_frequency=signature_frequency,
                        document_count=len(eligible),
                    )
                    for source_unit in mechanism.structural_units
                    for query_unit in query_structures
                ),
                default=0.0,
            )
            fused = (
                settings.kite_semantic_weight * semantic
                + settings.kite_structural_weight * structural
            )
            scored.append((fused, semantic, structural, mechanism_index, unit_index))
    rows_by_mechanism: dict[int, list[tuple[float, float, float, int, int]]] = {}
    for value in scored:
        rows_by_mechanism.setdefault(value[3], []).append(value)
    best_by_mechanism = {
        index: max(values, key=lambda value: (value[0], value[1], value[2]))
        for index, values in rows_by_mechanism.items()
    }
    semantic_best = {
        index: max(values, key=lambda value: (value[1], value[2], value[0]))
        for index, values in rows_by_mechanism.items()
    }
    structural_best = {
        index: max(values, key=lambda value: (value[2], value[1], value[0]))
        for index, values in rows_by_mechanism.items()
    }
    semantic_order = sorted(
        semantic_best.values(), key=lambda value: (-value[1], -value[2], value[3])
    )
    structural_order = sorted(
        structural_best.values(), key=lambda value: (-value[2], -value[1], value[3])
    )
    semantic_ranks = {value[3]: rank for rank, value in enumerate(semantic_order, 1)}
    structural_ranks = {value[3]: rank for rank, value in enumerate(structural_order, 1)}
    recalled = {
        value[3] for value in semantic_order[:count]
    } | {value[3] for value in structural_order[:count]}
    fused_candidates = sorted(
        (value for index, value in best_by_mechanism.items() if index in recalled),
        key=lambda value: (-value[0], -value[1], -value[2], value[3]),
    )
    selected: list[RetrievalMatch] = []
    for fused, semantic, structural, mechanism_index, unit_index in fused_candidates:
        mechanism = eligible[mechanism_index]
        paths: list[Literal["semantic", "structural"]] = []
        if semantic_ranks[mechanism_index] <= count:
            paths.append("semantic")
        if structural_ranks[mechanism_index] <= count:
            paths.append("structural")
        selected.append(
            RetrievalMatch(
                rank=len(selected) + 1,
                mechanism=mechanism,
                target_unit=profile.units[unit_index],
                semantic_score=semantic,
                structural_score=structural,
                fused_score=fused,
                semantic_rank=semantic_ranks[mechanism_index],
                structural_rank=structural_ranks[mechanism_index],
                support_paths=paths,
            )
        )
        if len(selected) >= count:
            break
    store.kite_put(
        key,
        "retrieval",
        json.dumps([value.model_dump(mode="json") for value in selected], sort_keys=True),
    )
    return selected
