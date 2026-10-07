"""Question-to-evidence retrieval."""
from __future__ import annotations

from typing import Sequence

from embeddings import EmbeddingProvider
from schemas import RetrievedChunk
from vector_store import VectorStore


class Retriever:
    """Embeds a question and returns relevant chunks above a similarity floor."""

    def __init__(self, embedder: EmbeddingProvider, store: VectorStore, top_k: int, min_relevance: float) -> None:
        self._embedder = embedder
        self._store = store
        self._top_k = top_k
        self._min_relevance = min_relevance

    def retrieve(self, question: str, doc_ids: Sequence[str] | None = None) -> list[RetrievedChunk]:
        """Return up to ``top_k`` chunks (best first). ``doc_ids=None`` searches everything."""
        if doc_ids is not None and len(doc_ids) == 0:
            return []
        vector = self._embedder.embed_query(question)
        hits = self._store.query(vector, self._top_k, doc_ids)
        return [h for h in hits if h.score >= self._min_relevance]
