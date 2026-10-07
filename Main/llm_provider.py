"""LLM providers. Add a provider by subclassing and registering it."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable

from config import Settings
from gemini_client import call_with_retry, make_client
from schemas import ConfigError, MissingCredentialsError, ProviderError


class LLMProvider(ABC):
    """Interface used by the RAG pipeline."""

    provider: str
    model: str

    @abstractmethod
    def generate(self, system_instruction: str, prompt: str) -> str:
        """Return the model's text answer."""


class GeminiLLMProvider(LLMProvider):
    """Hosted Gemini model accessed through the Gemini API."""

    provider = "gemini"

    def __init__(self, api_key: str, model: str, temperature: float | None = None, client: Any | None = None) -> None:
        self.model = model
        self._api_key = api_key
        self._temperature = temperature
        self._client = client

    def generate(self, system_instruction: str, prompt: str) -> str:
        from google.genai import types

        if self._client is None:
            self._client = make_client(self._api_key)
        client = self._client
        config = types.GenerateContentConfig(system_instruction=system_instruction, temperature=self._temperature)
        response = call_with_retry(
            lambda: client.models.generate_content(model=self.model, contents=prompt, config=config)
        )
        text = (response.text or "").strip() if response is not None else ""
        if not text:
            raise ProviderError("Gemini returned no text (the response may have been blocked). Try rephrasing.")
        return text


def _gemini(settings: Settings) -> LLMProvider:
    if not settings.gemini_api_key:
        raise MissingCredentialsError(
            "GEMINI_API_KEY is required for the Gemini LLM. Add it to Config/.env or your environment."
        )
    return GeminiLLMProvider(settings.gemini_api_key, settings.llm_model, settings.llm_temperature)


_FACTORIES: dict[str, Callable[[Settings], LLMProvider]] = {"gemini": _gemini}


def create_llm_provider(settings: Settings) -> LLMProvider:
    """Build the configured LLM provider or raise a clear ConfigError."""
    factory = _FACTORIES.get(settings.llm_provider)
    if factory is None:
        raise ConfigError(
            f"Unsupported LLM_PROVIDER '{settings.llm_provider}'. "
            f"Supported: {', '.join(sorted(_FACTORIES))}. See README, 'Adding a provider'."
        )
    return factory(settings)
