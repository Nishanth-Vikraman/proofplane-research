from __future__ import annotations
from proofplane_v0.models import HiddenBugPayload


def evaluation_bucket(payload: HiddenBugPayload, training: list[HiddenBugPayload]) -> str:
    if payload.confidence < 0.5:
        return "unclassified"
    if any(x.bug_family.lower() == payload.bug_family.lower() for x in training):
        return "seen_family"
    known = {p for item in training for p in item.mechanism_primitives}
    present = set(payload.mechanism_primitives) & known
    if set(payload.mechanism_primitives) <= known and len(present) >= 2:
        return "compositional"
    return "novel_ood"
