"""Vector index (ChromaDB) and the persistent document registry.

Chroma is a local embedded database suited to a single user or small team.
To move to a managed database (pgvector, Qdrant, Pinecone, ...), implement
:class:`VectorStore` and register it in :func:`create_vector_store`.
"""
from __future__ import annotations

import json
import os
import re
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Sequence

from config import Settings
from logger import get_logger
from schemas import Chunk, ConfigError, DocumentRecord, RetrievedChunk, VectorStoreError

_log = get_logger("store")


class VectorStore(ABC):
    """Interface used by the retriever and pipeline."""

    @abstractmethod
    def add_chunks(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None: ...

    @abstractmethod
    def query(
        self, embedding: Sequence[float], top_k: int, doc_ids: Sequence[str] | None = None
    ) -> list[RetrievedChunk]:
        """Return the closest chunks (score = cosine similarity), best first."""

    @abstractmethod
    def delete_document(self, doc_id: str) -> None: ...

    @abstractmethod
    def count(self) -> int: ...


class ChromaVectorStore(VectorStore):
    """Persistent Chroma collection, one per embedding signature (so dimensions never clash)."""

    PREFIX = "findoc_"

    def __init__(self, path: Path, signature: str) -> None:
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            path.mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(
                path=str(path), settings=ChromaSettings(anonymized_telemetry=False)
            )
            name = self.PREFIX + re.sub(r"[^A-Za-z0-9]+", "_", signature).strip("_")
            self._collection = self._client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})
        except Exception as exc:
            raise VectorStoreError(f"Could not open the vector database at {path}: {type(exc).__name__}.") from exc

    def add_chunks(self, chunks: Sequence[Chunk], embeddings: Sequence[Sequence[float]]) -> None:
        if len(chunks) != len(embeddings):
            raise VectorStoreError("Chunk and embedding counts do not match.")
        try:
            for i in range(0, len(chunks), 200):
                part = chunks[i : i + 200]
                self._collection.upsert(
                    ids=[c.chunk_id for c in part],
                    embeddings=[list(e) for e in embeddings[i : i + 200]],
                    documents=[c.text for c in part],
                    metadatas=[
                        {"doc_id": c.doc_id, "source": c.source, "page": c.page or 0, "chunk_index": c.chunk_index}
                        for c in part
                    ],
                )
        except Exception as exc:
            raise VectorStoreError(f"Could not write to the vector database: {type(exc).__name__}.") from exc

    def query(
        self, embedding: Sequence[float], top_k: int, doc_ids: Sequence[str] | None = None
    ) -> list[RetrievedChunk]:
        try:
            total = self._collection.count()
            if total == 0:
                return []
            where: dict[str, Any] | None = {"doc_id": {"$in": list(doc_ids)}} if doc_ids else None
            res = self._collection.query(
                query_embeddings=[list(embedding)],
                n_results=min(top_k, total),
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise VectorStoreError(f"Could not search the vector database: {type(exc).__name__}.") from exc
        results: list[RetrievedChunk] = []
        for cid, doc, meta, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]):
            chunk = Chunk(cid, meta["doc_id"], meta["source"], meta["page"] or None, int(meta["chunk_index"]), doc)
            results.append(RetrievedChunk(chunk, max(0.0, min(1.0, 1.0 - float(dist)))))
        return results

    def delete_document(self, doc_id: str) -> None:
        """Delete from every findoc collection, including ones from older embedding models."""
        try:
            for item in self._client.list_collections():
                name = getattr(item, "name", item)
                if str(name).startswith(self.PREFIX):
                    self._client.get_collection(name).delete(where={"doc_id": doc_id})
        except Exception as exc:
            raise VectorStoreError(f"Could not delete vectors: {type(exc).__name__}.") from exc

    def count(self) -> int:
        return int(self._collection.count())


def create_vector_store(settings: Settings, signature: str) -> VectorStore:
    """Build the configured vector store or raise a clear ConfigError."""
    if settings.vector_store == "chroma":
        return ChromaVectorStore(settings.database_dir / "chroma", signature)
    raise ConfigError(
        f"Unsupported VECTOR_STORE '{settings.vector_store}'. Supported: chroma. See README, 'Replacing the vector database'."
    )


class DocumentRegistry:
    """JSON file (``DataBase/documents.json``) holding metadata for indexed documents."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._lock = threading.Lock()
        self._records: dict[str, DocumentRecord] = self._load()

    def _load(self) -> dict[str, DocumentRecord]:
        if not self._path.exists():
            return {}
        try:
            raw = json.loads(self._path.read_text(encoding="utf-8"))
            return {k: DocumentRecord.from_dict(v) for k, v in raw.items()}
        except (ValueError, TypeError, OSError) as exc:
            backup = self._path.with_suffix(".corrupt")
            _log.error("Registry unreadable (%s); moved to %s", type(exc).__name__, backup.name)
            os.replace(self._path, backup)
            return {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps({k: v.to_dict() for k, v in self._records.items()}, indent=2), encoding="utf-8")
        os.replace(tmp, self._path)  # atomic on Windows and POSIX

    def all(self) -> list[DocumentRecord]:
        with self._lock:
            return sorted(self._records.values(), key=lambda r: (r.indexed_at, r.filename))

    def get(self, doc_id: str) -> DocumentRecord | None:
        with self._lock:
            return self._records.get(doc_id)

    def put(self, record: DocumentRecord) -> None:
        with self._lock:
            self._records[record.doc_id] = record
            self._save()

    def remove(self, doc_id: str) -> None:
        with self._lock:
            self._records.pop(doc_id, None)
            self._save()
