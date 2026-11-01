from __future__ import annotations

import hashlib
import numpy as np


def serialize(vector: np.ndarray) -> bytes:
    return np.asarray(vector, dtype="<f4").tobytes()


def deserialize(blob: bytes, dimensions: int) -> np.ndarray:
    vector = np.frombuffer(blob, dtype="<f4")
    if vector.size != dimensions:
        raise ValueError("embedding dimension mismatch")
    return vector.copy()


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / denominator) if denominator else 0.0


def deterministic_embedding(text: str, dimensions: int = 64) -> np.ndarray:
    vector = np.zeros(dimensions, dtype=np.float32)
    for token in text.lower().split():
        digest = hashlib.sha256(token.encode()).digest()
        vector[int.from_bytes(digest[:2], "little") % dimensions] += 1
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector
