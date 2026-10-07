"""Safe saving and text extraction for PDF, DOCX, and TXT files."""
from __future__ import annotations

import os
import re
from pathlib import Path

from schemas import DocumentLoadError, PageText, UnsupportedFileTypeError

SUPPORTED_EXTENSIONS = (".pdf", ".docx", ".txt")
_MAGIC = {".pdf": b"%PDF", ".docx": b"PK"}  # DOCX is a ZIP container


def safe_filename(name: str) -> str:
    """Strip directories and unsafe characters; keep a lowercase extension."""
    base = Path(name.replace("\\", "/")).name
    stem, ext = os.path.splitext(base)
    stem = re.sub(r"[^A-Za-z0-9._ -]+", "_", stem).strip(" ._-")
    stem = re.sub(r"\s+", "_", stem) or "document"
    return f"{stem[:100]}{ext.lower()}"


def validate_upload(filename: str, data: bytes, max_bytes: int) -> str:
    """Validate type, size, and content signature. Returns the extension."""
    ext = Path(filename.replace("\\", "/")).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileTypeError(
            f"'{Path(filename).name}' is not supported. Upload PDF, DOCX, or TXT files."
        )
    if not data:
        raise DocumentLoadError(f"'{Path(filename).name}' is empty.")
    if len(data) > max_bytes:
        raise DocumentLoadError(
            f"'{Path(filename).name}' is larger than the {max_bytes // (1024 * 1024)} MB upload limit."
        )
    magic = _MAGIC.get(ext)
    if magic and not data.startswith(magic):
        raise DocumentLoadError(f"'{Path(filename).name}' does not look like a valid {ext[1:].upper()} file.")
    if ext == ".txt" and b"\x00" in data[:4096]:
        raise DocumentLoadError(f"'{Path(filename).name}' looks like a binary file, not plain text.")
    return ext


def save_upload(filename: str, data: bytes, directory: Path, max_bytes: int) -> Path:
    """Validate and save an upload without ever overwriting an existing file."""
    validate_upload(filename, data, max_bytes)
    directory.mkdir(parents=True, exist_ok=True)
    safe = safe_filename(filename)
    stem, ext = os.path.splitext(safe)
    counter = 0
    while True:
        candidate = directory / (safe if counter == 0 else f"{stem}_{counter}{ext}")
        try:
            with open(candidate, "xb") as handle:  # "x" fails if the file exists
                handle.write(data)
            return candidate
        except FileExistsError:
            counter += 1


def _load_pdf(path: Path) -> list[PageText]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    if reader.is_encrypted and not reader.decrypt(""):
        raise DocumentLoadError(f"'{path.name}' is password-protected. Remove the password and upload again.")
    return [PageText(i, (page.extract_text() or "")) for i, page in enumerate(reader.pages, start=1)]


def _load_docx(path: Path) -> list[PageText]:
    from docx import Document
    from docx.table import Table

    parts: list[str] = []
    for block in Document(str(path)).iter_inner_content():
        if isinstance(block, Table):
            for row in block.rows:
                cells = [c.text.strip() for c in row.cells]
                if any(cells):
                    parts.append(" | ".join(cells))
        elif block.text.strip():
            parts.append(block.text)
    return [PageText(None, "\n".join(parts))]  # DOCX has no fixed page boundaries


def _load_txt(path: Path) -> list[PageText]:
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    return [PageText(None, text)]


def load_document(path: Path) -> list[PageText]:
    """Extract non-empty page texts, or raise a user-friendly error."""
    loaders = {".pdf": _load_pdf, ".docx": _load_docx, ".txt": _load_txt}
    loader = loaders.get(path.suffix.lower())
    if loader is None:
        raise UnsupportedFileTypeError(f"'{path.name}' is not supported. Upload PDF, DOCX, or TXT files.")
    try:
        pages = loader(path)
    except DocumentLoadError:
        raise
    except Exception as exc:  # corrupted files raise many different library errors
        raise DocumentLoadError(f"Could not read '{path.name}'. The file may be corrupted.") from exc
    pages = [p for p in pages if p.text.strip()]
    if not pages:
        raise DocumentLoadError(
            f"No text could be extracted from '{path.name}'. Scanned PDFs need OCR first; "
            "this app does not perform OCR."
        )
    return pages
