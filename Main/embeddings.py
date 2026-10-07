"""Embedding providers. Add a provider by subclassing and registering it."""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any, Callable, Sequence

from config import Settings
from gemini_client import call_with_retry, make_client
from schemas import ConfigError, MissingCredentialsError, ProviderError

_BATCH_SIZE = 50
_FULL_DIMS = 3072


class EmbeddingProvider(ABC):
    """Interface used by the RAG pipeline."""

    provider: str
    model: str
    dimensions: int

    @property
    def signature(self) -> str:
        """Identifies the embedding space; vectors from different signatures are incompatible."""
        return f"{self.provider}:{self.model}:{self.dimensions}"

    @abstractmethod
    def embed_documents(self, texts: Sequence[str], titles: Sequence[str] | None = None) -> list[list[float]]:
        """Embed passages for indexing."""

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """Embed a user question for retrieval."""


def _normalize(vec: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


class GeminiEmbeddingProvider(EmbeddingProvider):
    """Gemini embeddings via the ``google-genai`` SDK.

    ``gemini-embedding-001`` uses ``task_type``; ``gemini-embedding-2`` has no
    ``task_type`` and instead expects a task prefix inside the text, and it
    aggregates multiple plain inputs, so each text is wrapped in its own Content.
    """

    provider = "gemini"

    def __init__(self, api_key: str, model: str, dimensions: int, client: Any | None = None) -> None:
        self.model = model
        self.dimensions = dimensions
        self._api_key = api_key
        self._client = client

    @property
    def _is_v2(self) -> bool:
        return self.model.startswith("gemini-embedding-2")

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = make_client(self._api_key)
        return self._client

    def _embed(self, contents: list[Any], task_type: str | None) -> list[list[float]]:
        from google.genai import types

        config = types.EmbedContentConfig(output_dimensionality=self.dimensions, task_type=task_type)
        client = self._get_client()
        response = call_with_retry(
            lambda: client.models.embed_content(model=self.model, contents=contents, config=config)
        )
        vectors = [list(e.values) for e in (response.embeddings or [])]
        if len(vectors) != len(contents):
            raise ProviderError("Gemini returned an unexpected number of embeddings.")
        if not self._is_v2 and self.dimensions != _FULL_DIMS:
            vectors = [_normalize(v) for v in vectors]  # required for truncated -001 vectors
        return vectors

    def _wrap(self, text: str) -> Any:
        from google.genai import types

        return types.Content(parts=[types.Part.from_text(text=text)])

    def embed_documents(self, texts: Sequence[str], titles: Sequence[str] | None = None) -> list[list[float]]:
        out: list[list[float]] = []
        for start in range(0, len(texts), _BATCH_SIZE):
            batch = list(texts[start : start + _BATCH_SIZE])
            if self._is_v2:
                names = titles[start : start + _BATCH_SIZE] if titles else ["none"] * len(batch)
                contents = [self._wrap(f"title: {n} | text: {t}") for n, t in zip(names, batch)]
                out.extend(self._embed(contents, None))
            else:
                out.extend(self._embed(batch, "RETRIEVAL_DOCUMENT"))
        return out

    def embed_query(self, text: str) -> list[float]:
        if self._is_v2:
            return self._embed([self._wrap(f"task: question answering | query: {text}")], None)[0]
        return self._embed([text], "QUESTION_ANSWERING")[0]


def _gemini(settings: Settings) -> EmbeddingProvider:
    if not settings.gemini_api_key:
        raise MissingCredentialsError(
            "GEMINI_API_KEY is required for Gemini embeddings. Add it to Config/.env or your environment."
        )
    return GeminiEmbeddingProvider(settings.gemini_api_key, settings.embedding_model, settings.embedding_dimensions)


_FACTORIES: dict[str, Callable[[Settings], EmbeddingProvider]] = {"gemini": _gemini}


def create_embedding_provider(settings: Settings) -> EmbeddingProvider:
    """Build the configured embedding provider or raise a clear ConfigError."""
    factory = _FACTORIES.get(settings.embedding_provider)
    if factory is None:
        raise ConfigError(
            f"Unsupported EMBEDDING_PROVIDER '{settings.embedding_provider}'. "
            f"Supported: {', '.join(sorted(_FACTORIES))}. See README, 'Adding a provider'."
        )
    return factory(settings)
