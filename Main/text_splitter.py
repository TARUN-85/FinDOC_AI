"""Overlapping text chunking that prefers natural boundaries."""
from __future__ import annotations

import re
from typing import Sequence

from schemas import Chunk, PageText

_SEPARATORS = ("\n\n", "\n", ". ", "; ", ", ", " ")


def normalize_text(text: str) -> str:
    """Normalize line endings and collapse excess whitespace."""
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = re.sub(r"[ \t\f\v]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _find_break(text: str, start: int, end: int, chunk_size: int) -> int:
    """Pick the last natural separator in the second half of the window."""
    lower = start + chunk_size // 2
    for sep in _SEPARATORS:
        idx = text.rfind(sep, lower, end)
        if idx != -1:
            return idx + len(sep)
    return end


def split_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list[str]:
    """Split ``text`` into chunks of at most ``chunk_size`` characters.

    Consecutive chunks share roughly ``overlap`` characters.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")
    text = normalize_text(text)
    if not text:
        return []
    n = len(text)
    if n <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < n:
        end = min(start + chunk_size, n)
        if end < n:
            end = _find_break(text, start, end, chunk_size)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= n:
            break
        nxt = end - overlap
        if nxt > start:
            space = text.find(" ", nxt, end)  # start the next chunk on a word boundary
            start = space + 1 if space != -1 else nxt
        else:
            start = end
    return chunks


def split_pages(
    pages: Sequence[PageText], doc_id: str, source: str, chunk_size: int, overlap: int
) -> list[Chunk]:
    """Chunk each page separately so every chunk keeps a page number."""
    chunks: list[Chunk] = []
    for page in pages:
        for piece in split_text(page.text, chunk_size, overlap):
            index = len(chunks)
            chunks.append(Chunk(f"{doc_id}:{index}", doc_id, source, page.page, index, piece))
    return chunks
