"""Logging setup with secret redaction. Never log document text or prompts."""
from __future__ import annotations

import logging
import re
from logging.handlers import RotatingFileHandler
from pathlib import Path

_REDACTIONS = [
    (re.compile(r"AIza[0-9A-Za-z_\-]{20,}"), "[REDACTED_KEY]"),
    (re.compile(r"(?i)((?:api[_-]?key|token|secret)\s*[=:]\s*)\S+"), r"\1[REDACTED]"),
]
_LOGGER_NAME = "findoc"


def redact(text: str) -> str:
    """Mask anything that looks like an API key or credential."""
    for pattern, repl in _REDACTIONS:
        text = pattern.sub(repl, text)
    return text


class _RedactingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


def setup_logging(log_dir: Path, level: str = "INFO") -> logging.Logger:
    """Configure the ``findoc`` logger once (idempotent)."""
    logger = logging.getLogger(_LOGGER_NAME)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    if logger.handlers:
        return logger
    log_dir.mkdir(parents=True, exist_ok=True)
    formatter = _RedactingFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    file_handler = RotatingFileHandler(log_dir / "findoc.log", maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    stream_handler = logging.StreamHandler()
    for handler in (file_handler, stream_handler):
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{_LOGGER_NAME}.{name}")
