from __future__ import annotations

from proofplane_v0.models import CausalRetrievalDescriptor


def metadata_overlap(
    query: CausalRetrievalDescriptor, item: CausalRetrievalDescriptor, weights: dict[str, float]
) -> float:
    numerator = denominator = 0.0
    for field, weight in weights.items():
        left, right = set(getattr(query, field)), set(getattr(item, field))
        denominator += weight
        numerator += weight * (len(left & right) / len(left | right) if left | right else 0.0)
    return numerator / denominator if denominator else 0.0
