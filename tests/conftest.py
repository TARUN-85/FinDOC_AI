"""Shared fakes so tests never touch the network or need an API key."""
from __future__ import annotations

import hashlib
import math
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "Main"))

from config import Settings  # noqa: E402
from embeddings import EmbeddingProvider  # noqa: E402
from llm_provider import LLMProvider  # noqa: E402
from rag_pipeline import RAGPipeline  # noqa: E402
from schemas import Chunk, RetrievedChunk  # noqa: E402
from vector_store import DocumentRegistry, VectorStore  # noqa: E402

DIMS = 256


class FakeEmbedder(EmbeddingProvider):
    """Deterministic bag-of-words embeddings."""

    provider, model, dimensions = "fake", "fake-1", DIMS

    def _vec(self, text: str) -> list[float]:
        v = [0.0] * DIMS
        for w in re.findall(r"[a-z0-9]+", text.lower()):
            v[int(hashlib.md5(w.encode()).hexdigest(), 16) % DIMS] += 1.0
        norm = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / norm for x in v]

    def embed_documents(self, texts, titles=None):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class MemoryStore(VectorStore):
    def __init__(self):
        self.items: dict[str, tuple[Chunk, list[float]]] = {}

    def add_chunks(self, chunks, embeddings):
        for c, e in zip(chunks, embeddings):
            self.items[c.chunk_id] = (c, list(e))

    def query(self, embedding, top_k, doc_ids=None):
        hits = [
            RetrievedChunk(c, sum(a * b for a, b in zip(embedding, e)))
            for c, e in self.items.values()
            if doc_ids is None or c.doc_id in doc_ids
        ]
        return sorted(hits, key=lambda h: -h.score)[:top_k]

    def delete_document(self, doc_id):
        self.items = {k: v for k, v in self.items.items() if v[0].doc_id != doc_id}

    def count(self):
        return len(self.items)


class FakeLLM(LLMProvider):
    provider, model = "fake", "fake-llm"

    def __init__(self, reply="Revenue was $4.2 million [1]."):
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def generate(self, system_instruction, prompt):
        self.calls.append((system_instruction, prompt))
        if isinstance(self.reply, Exception):
            raise self.reply
        return self.reply


@pytest.fixture
def settings(tmp_path):
    return Settings(
        input_dir=tmp_path / "in", database_dir=tmp_path / "db", log_dir=tmp_path / "logs",
        chunk_size=200, chunk_overlap=20, top_k=3, min_relevance=0.2, gemini_api_key="test-key",
    )


@pytest.fixture
def llm():
    return FakeLLM()


@pytest.fixture
def pipeline(settings, llm):
    return RAGPipeline(settings, FakeEmbedder(), MemoryStore(), llm, DocumentRegistry(settings.database_dir / "documents.json"))
