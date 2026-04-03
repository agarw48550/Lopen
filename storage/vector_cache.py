"""Lightweight in-memory vector cache with optional numpy cosine similarity."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

_NUMPY_AVAILABLE = False
try:
    import numpy as np
    _NUMPY_AVAILABLE = True
except ImportError:
    pass


@dataclass
class VectorEntry:
    key: str
    embedding: list[float]
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorCache:
    """In-memory vector store with cosine similarity search."""

    def __init__(self, max_size: int = 1000) -> None:
        self._store: dict[str, VectorEntry] = {}
        self.max_size = max_size
        logger.info("VectorCache initialised (max_size=%d, numpy=%s)", max_size, _NUMPY_AVAILABLE)

    # ------------------------------------------------------------------
    # Storage
    # ------------------------------------------------------------------

    def store(
        self,
        key: str,
        embedding: list[float],
        text: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Store an embedding with its associated text."""
        if len(self._store) >= self.max_size:
            oldest = next(iter(self._store))
            del self._store[oldest]
            logger.debug("VectorCache evicted oldest entry: %s", oldest)
        self._store[key] = VectorEntry(
            key=key,
            embedding=embedding,
            text=text,
            metadata=metadata or {},
        )

    def get(self, key: str) -> Optional[VectorEntry]:
        return self._store.get(key)

    def delete(self, key: str) -> bool:
        if key in self._store:
            del self._store[key]
            return True
        return False

    def clear(self) -> None:
        self._store.clear()

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 5,
    ) -> list[tuple[float, VectorEntry]]:
        """Return top_k entries sorted by cosine similarity (descending)."""
        if not self._store:
            return []

        results: list[tuple[float, VectorEntry]] = []
        for entry in self._store.values():
            sim = self._cosine_similarity(query_embedding, entry.embedding)
            results.append((sim, entry))

        results.sort(key=lambda x: x[0], reverse=True)
        return results[:top_k]

    @property
    def size(self) -> int:
        return len(self._store)

    # ------------------------------------------------------------------
    # Similarity
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        if len(a) != len(b):
            return 0.0
        if _NUMPY_AVAILABLE:
            na, nb = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
            denom = (np.linalg.norm(na) * np.linalg.norm(nb))
            if denom == 0:
                return 0.0
            return float(np.dot(na, nb) / denom)
        # Pure-Python fallback
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
