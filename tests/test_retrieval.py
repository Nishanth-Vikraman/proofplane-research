import numpy as np
import pytest
from proofplane_v0.models import CausalRetrievalDescriptor
from proofplane_v0.retrieval.embeddings import cosine, deserialize, serialize
from proofplane_v0.retrieval.metadata import metadata_overlap
from proofplane_v0.retrieval.mmr import select_mmr


def d(roles):
    return CausalRetrievalDescriptor(
        component_roles=roles, visible_change_summary="x", retrieval_text="x", confidence=1
    )


def test_embedding_round_trip_and_cosine():
    vector = np.array([1, 2, 3], dtype=np.float32)
    assert np.array_equal(vector, deserialize(serialize(vector), 3))
    assert cosine(vector, vector) == pytest.approx(1)


def test_metadata_overlap():
    assert metadata_overlap(d(["cache"]), d(["cache"]), {"component_roles": 1}) == 1


def test_mmr_diversifies():
    def similarity(a, b):
        return 1.0 if {a, b} == {"a", "b"} else 0.0

    result = select_mmr(["a", "b", "c"], {"a": 1, "b": 0.99, "c": 0.8}, similarity, 2, 0.5)
    assert result == ["a", "c"]
