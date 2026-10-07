"""Configuration loading for FinDOC AI.

Precedence (highest first):
    1. Process environment variables
    2. ``Config/.env`` (then ``<project root>/.env``)
    3. ``Config/settings.yaml``
    4. Built-in defaults

Blank values (``EMBEDDING_MODEL=``) count as "unset" and fall through.
Secrets (the API key) are only read from environment sources, never YAML.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Mapping

import yaml
from dotenv import dotenv_values

from schemas import ConfigError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "Config"
VALID_LOG_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR")


@dataclass(frozen=True)
class Settings:
    """Immutable application settings."""

    # Generation and embedding are configured independently.
    llm_provider: str = "gemini"
    llm_model: str = "gemini-3.8-flash"
    llm_temperature: float | None = None  # None = provider default (recommended for Gemini 3)
    embedding_provider: str = "gemini"
    embedding_model: str = "gemini-embedding-001"
    embedding_dimensions: int = 768
    vector_store: str = "chroma"
    chunk_size: int = 1000
    chunk_overlap: int = 150
    top_k: int = 5
    min_relevance: float = 0.30
    max_upload_mb: int = 25
    log_level: str = "INFO"
    input_dir: Path = PROJECT_ROOT / "InputDoc"
    database_dir: Path = PROJECT_ROOT / "DataBase"
    log_dir: Path = PROJECT_ROOT / "Logs"
    gemini_api_key: str = field(default="", repr=False)  # never printed or logged

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


_PATH_FIELDS = {"input_dir", "database_dir", "log_dir"}


def _resolve_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (yaml.YAMLError, OSError) as exc:
        raise ConfigError(f"Could not read {path.name}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path.name} must contain a mapping of setting names to values.")
    return {str(k).lower(): v for k, v in data.items()}


def _coerce(name: str, value: Any, target: Any) -> Any:
    """Convert a raw value to the type expected for setting ``name``."""
    try:
        if name in _PATH_FIELDS:
            return _resolve_path(value)
        if name == "llm_temperature":
            return float(value)
        if "int" in str(target):
            return int(str(value).strip())
        if "float" in str(target):
            return float(str(value).strip())
        text = str(value).strip()
        return text.lower() if name.endswith("_provider") or name == "vector_store" else text
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Setting {name.upper()} has an invalid value: {value!r}.") from exc


def validate_settings(s: Settings) -> None:
    """Raise :class:`ConfigError` for impossible or risky combinations."""
    if not 100 <= s.chunk_size <= 6000:
        raise ConfigError("CHUNK_SIZE must be between 100 and 6000 characters.")
    if not 0 <= s.chunk_overlap < s.chunk_size:
        raise ConfigError("CHUNK_OVERLAP must be zero or more and smaller than CHUNK_SIZE.")
    if not 1 <= s.top_k <= 50:
        raise ConfigError("TOP_K must be between 1 and 50.")
    if not 0.0 <= s.min_relevance <= 1.0:
        raise ConfigError("MIN_RELEVANCE must be between 0 and 1.")
    if not 128 <= s.embedding_dimensions <= 3072:
        raise ConfigError("EMBEDDING_DIMENSIONS must be between 128 and 3072.")
    if s.max_upload_mb < 1:
        raise ConfigError("MAX_UPLOAD_MB must be at least 1.")
    if s.log_level.upper() not in VALID_LOG_LEVELS:
        raise ConfigError(f"LOG_LEVEL must be one of {', '.join(VALID_LOG_LEVELS)}.")
    if s.llm_temperature is not None and not 0.0 <= s.llm_temperature <= 2.0:
        raise ConfigError("LLM_TEMPERATURE must be between 0 and 2 (or left blank).")
    for name in ("llm_provider", "llm_model", "embedding_provider", "embedding_model", "vector_store"):
        if not getattr(s, name):
            raise ConfigError(f"{name.upper()} must not be empty.")


def load_settings(
    env: Mapping[str, str] | None = None,
    settings_file: Path | None = None,
    use_dotenv: bool = True,
) -> Settings:
    """Load settings following the documented precedence order."""
    environ = os.environ if env is None else env
    yaml_values = _read_yaml(settings_file or CONFIG_DIR / "settings.yaml")
    if "gemini_api_key" in yaml_values:
        raise ConfigError("Do not put secrets in settings.yaml. Set GEMINI_API_KEY in Config/.env or the environment.")

    dotenv_vals: dict[str, str | None] = {}
    if use_dotenv:
        for candidate in (PROJECT_ROOT / ".env", CONFIG_DIR / ".env"):  # later file wins
            if candidate.exists():
                dotenv_vals.update(dotenv_values(candidate))

    defaults = Settings()
    values: dict[str, Any] = {}
    for f in fields(Settings):
        key = f.name.upper()
        sources: list[Any] = [environ.get(key), dotenv_vals.get(key)]
        if f.name != "gemini_api_key":
            sources.append(yaml_values.get(f.name))
        raw = next((v for v in sources if not _blank(v)), None)
        if raw is None:
            values[f.name] = getattr(defaults, f.name)
        elif f.name == "gemini_api_key":
            values[f.name] = str(raw).strip()
        else:
            values[f.name] = _coerce(f.name, raw, f.type)
    settings = Settings(**values)
    validate_settings(settings)
    return settings


def ensure_directories(settings: Settings) -> None:
    """Create data and log directories if they do not exist."""
    for path in (settings.input_dir, settings.database_dir, settings.log_dir):
        path.mkdir(parents=True, exist_ok=True)
