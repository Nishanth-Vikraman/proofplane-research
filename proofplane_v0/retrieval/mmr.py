from __future__ import annotations

from collections.abc import Callable


def select_mmr(
    items: list[str],
    relevance: dict[str, float],
    similarity: Callable[[str, str], float],
    k: int,
    lambda_: float = 0.7,
) -> list[str]:
    selected: list[str] = []
    while items and len(selected) < k:
        winner = max(
            items,
            key=lambda item: (
                lambda_ * relevance[item]
                - (1 - lambda_) * max((similarity(item, prior) for prior in selected), default=0)
            ),
        )
        selected.append(winner)
        items.remove(winner)
    return selected
