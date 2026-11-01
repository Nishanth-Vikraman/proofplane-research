from __future__ import annotations

import random
from dataclasses import dataclass

from proofplane_v0.models import CausalRetrievalDescriptor, CorpusBugRecord
from .embeddings import cosine, deterministic_embedding
from .metadata import metadata_overlap
from .mmr import select_mmr


@dataclass
class Retrieved:
    record: CorpusBugRecord
    scores: dict[str, float]


def retrieve(
    query: CausalRetrievalDescriptor,
    records: list[CorpusBugRecord],
    mode: str,
    k: int,
    weights: dict[str, float],
    metadata_weights: dict[str, float],
    seed: int = 17,
    mmr_lambda: float = 0.7,
    candidate_pool_size: int = 10,
) -> list[Retrieved]:
    if mode == "oracle_hidden_family":
        raise PermissionError("oracle mode requires the dedicated explicit oracle path")
    if mode == "random":
        sampled = random.Random(seed).sample(records, min(k, len(records)))
        return [Retrieved(x, {"random": 1.0}) for x in sampled]
    qd = deterministic_embedding(query.retrieval_text)
    qs = deterministic_embedding(" ".join(query.event_sequence))
    scored: list[Retrieved] = []
    for record in records:
        descriptor = record.visible_descriptor
        parts = {
            "descriptor": cosine(qd, deterministic_embedding(descriptor.retrieval_text)),
            "code": cosine(qd, deterministic_embedding(descriptor.visible_change_summary)),
            "sequence": cosine(qs, deterministic_embedding(" ".join(descriptor.event_sequence))),
            "metadata": metadata_overlap(query, descriptor, metadata_weights),
        }
        key = {
            "descriptor_embedding": "descriptor",
            "code_change_embedding": "code",
            "sequence_only": "sequence",
            "metadata_only": "metadata",
        }.get(mode)
        parts["total"] = (
            parts[key]
            if key
            else sum(weights.get(name, 0) * score for name, score in parts.items())
        )
        scored.append(Retrieved(record, parts))
    scored.sort(key=lambda x: x.scores["total"], reverse=True)
    if mode != "hybrid_mmr":
        return scored[:k]
    pool = scored[: max(k, min(candidate_pool_size, len(scored)))]
    lookup = {x.record.corpus_id: x for x in pool}
    primitive = {
        x.record.corpus_id: set(x.record.hidden_payload.mechanism_primitives) for x in pool
    }

    def similarity(a: str, b: str) -> float:
        union = primitive[a] | primitive[b]
        return len(primitive[a] & primitive[b]) / len(union) if union else 0

    ids = select_mmr(
        list(lookup),
        {key: value.scores["total"] for key, value in lookup.items()},
        similarity,
        k,
        mmr_lambda,
    )
    return [lookup[x] for x in ids]
