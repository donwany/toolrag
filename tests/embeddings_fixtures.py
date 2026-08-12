"""Lightweight embeddings for integration tests (no sentence-transformers required)."""

import hashlib

from langchain_core.embeddings import Embeddings
from toolrag.pgvector_store import DEFAULT_EMBEDDING_DIMENSION


class DeterministicEmbeddings(Embeddings):
    """Hash-based fixed-dimension vectors for pgvector integration tests."""

    def __init__(self, dimension: int = DEFAULT_EMBEDDING_DIMENSION):
        self.dimension = dimension

    def _vectorize(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        values: list[float] = []
        while len(values) < self.dimension:
            for byte in digest:
                values.append((byte / 127.5) - 1.0)
                if len(values) >= self.dimension:
                    break
            digest = hashlib.sha256(digest).digest()
        norm = sum(v * v for v in values) ** 0.5 or 1.0
        return [v / norm for v in values]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vectorize(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vectorize(text)
