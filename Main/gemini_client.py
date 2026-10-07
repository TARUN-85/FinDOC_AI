"""Shared Gemini client helpers: client creation, retries, and error translation."""
from __future__ import annotations

import time
from typing import Any, Callable, TypeVar

from logger import get_logger
from schemas import FinDocError, MissingCredentialsError, ProviderError, RateLimitError

T = TypeVar("T")
_log = get_logger("gemini")
_RETRYABLE = {429, 500, 502, 503, 504}


def make_client(api_key: str) -> Any:
    """Create a ``google-genai`` client (hosted Gemini API, not open source)."""
    if not api_key:
        raise MissingCredentialsError(
            "GEMINI_API_KEY is not set. Add it to Config/.env or your environment (see README)."
        )
    from google import genai

    return genai.Client(api_key=api_key)


def _status_code(exc: Exception) -> int | None:
    code = getattr(exc, "code", None)
    return code if isinstance(code, int) else None


def translate_error(exc: Exception) -> FinDocError:
    """Map SDK/network exceptions to short, secret-free, actionable messages."""
    if isinstance(exc, FinDocError):
        return exc
    code = _status_code(exc)
    if code == 429:
        return RateLimitError(
            "Gemini rate limit or quota reached. Wait a minute and try again, or check your quota."
        )
    if code in (401, 403) or (code == 400 and "api key" in str(exc).lower()):
        return MissingCredentialsError(
            "Gemini rejected the API key. Check that GEMINI_API_KEY is correct and has access."
        )
    if code == 404:
        return ProviderError(
            "Gemini could not find the configured model. It may be retired or unavailable to your account; "
            "update LLM_MODEL / EMBEDDING_MODEL (see README)."
        )
    if code is not None and code >= 500:
        return ProviderError("The Gemini service is temporarily unavailable. Please try again shortly.")
    if code is not None:
        return ProviderError(f"Gemini rejected the request (HTTP {code}).")
    return ProviderError(f"Could not reach Gemini ({type(exc).__name__}). Check your network connection.")


def call_with_retry(
    fn: Callable[[], T],
    attempts: int = 3,
    base_delay: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Run ``fn`` with exponential backoff on rate-limit/transient errors."""
    for attempt in range(1, attempts + 1):
        try:
            return fn()
        except FinDocError:
            raise
        except Exception as exc:
            err = translate_error(exc)
            if attempt == attempts or _status_code(exc) not in _RETRYABLE:
                _log.warning("Gemini call failed: %s", type(err).__name__)
                raise err from exc
            sleep(base_delay * 2 ** (attempt - 1))
    raise ProviderError("Gemini request failed.")  # pragma: no cover
