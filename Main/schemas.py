"""Shared data structures and exception types for FinDOC AI."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# --------------------------------------------------------------------------- #
# Exceptions (all user-facing messages should be actionable)
# --------------------------------------------------------------------------- #
class FinDocError(Exception):
    """Base class for errors whose message is safe to show to the user."""


class ConfigError(FinDocError):
    """Invalid or inconsistent configuration."""


class MissingCredentialsError(ConfigError):
    """A required API key or credential is not configured."""


class DocumentLoadError(FinDocError):
    """A document could not be validated, read, or contained no text."""


class UnsupportedFileTypeError(DocumentLoadError):
    """The uploaded file type is not supported."""


class DuplicateDocumentError(DocumentLoadError):
    """The same file content has already been indexed."""


class ProviderError(FinDocError):
    """An LLM or embedding provider request failed."""


class RateLimitError(ProviderError):
    """The provider rate limit or quota was reached."""


class VectorStoreError(FinDocError):
    """The vector database could not complete an operation."""


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PageText:
    """Text extracted from one page (``page`` is None when pages are unknown)."""

    page: int | None
    text: str


@dataclass(frozen=True)
class Chunk:
    """A searchable piece of a document with citation metadata."""

    chunk_id: str
    doc_id: str
    source: str
    page: int | None
    chunk_index: int
    text: str

    @property
    def location(self) -> str:
        """Human-readable location, e.g. ``page 3, chunk 12``."""
        parts = [f"page {self.page}"] if self.page else []
        parts.append(f"chunk {self.chunk_index}")
        return ", ".join(parts)


@dataclass(frozen=True)
class RetrievedChunk:
    """A chunk returned by retrieval with its cosine similarity (0..1)."""

    chunk: Chunk
    score: float


@dataclass(frozen=True)
class Citation:
    """A citation that points at a real retrieved passage."""

    index: int
    doc_id: str
    source: str
    page: int | None
    chunk_index: int
    location: str


@dataclass
class DocumentRecord:
    """Metadata persisted for every indexed document."""

    doc_id: str
    filename: str
    stored_name: str
    file_type: str
    size_bytes: int
    page_count: int | None
    chunk_count: int
    indexed_at: str
    embedding_signature: str

    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocumentRecord":
        return cls(**{k: data.get(k) for k in cls.__dataclass_fields__})  # type: ignore[attr-defined]


@dataclass
class Answer:
    """Result of a question: text, verified citations, and evidence."""

    question: str
    text: str = ""
    citations: list[Citation] = field(default_factory=list)
    evidence: list[RetrievedChunk] = field(default_factory=list)
    sufficient: bool = True
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
