import pytest

from schemas import PageText
from text_splitter import normalize_text, split_pages, split_text


def test_short_text_single_chunk():
    assert split_text("Hello world", 100, 10) == ["Hello world"]


def test_empty_text():
    assert split_text("   \n ", 100, 10) == []


def test_chunks_respect_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(400))
    chunks = split_text(text, 200, 40)
    assert len(chunks) > 3
    assert all(len(c) <= 200 for c in chunks)
    for a, b in zip(chunks, chunks[1:]):
        assert set(a.split()[-3:]) & set(b.split()[:10])  # overlapping words


def test_all_content_is_covered():
    words = [f"w{i}" for i in range(300)]
    joined = " ".join(c for c in split_text(" ".join(words), 150, 30))
    assert all(w in joined.split() for w in words)


def test_prefers_paragraph_boundary():
    text = ("a" * 60 + "\n\n" + "b" * 60 + "\n\n" + "c" * 60)
    chunks = split_text(text, 100, 10)
    assert chunks[0] == "a" * 60


def test_no_infinite_loop_on_unbroken_text():
    assert len(split_text("x" * 1000, 100, 20)) >= 10


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, -1)])
def test_invalid_parameters(size, overlap):
    with pytest.raises(ValueError):
        split_text("text", size, overlap)


def test_normalize_text():
    assert normalize_text("a \r\n\n\n\n b\x00") == "a\n\nb"


def test_split_pages_keeps_metadata():
    pages = [PageText(1, "First page. " * 30), PageText(2, "Second page. " * 30)]
    chunks = split_pages(pages, "doc1", "report.pdf", 100, 10)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert {c.page for c in chunks} == {1, 2}
    assert all(c.source == "report.pdf" and c.doc_id == "doc1" for c in chunks)
    assert chunks[0].location.startswith("page 1")
