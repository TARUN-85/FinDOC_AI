"""FinDOC AI - Streamlit entry point. Run with: streamlit run Main/app.py"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # allow flat imports from any launch directory

import streamlit as st  # noqa: E402

from config import ensure_directories, load_settings  # noqa: E402
from document_loader import SUPPORTED_EXTENSIONS  # noqa: E402
from logger import setup_logging  # noqa: E402
from rag_pipeline import DISCLAIMER, RAGPipeline, create_pipeline  # noqa: E402
from schemas import Answer, FinDocError  # noqa: E402

st.set_page_config(page_title="FinDOC AI", page_icon="📄", layout="wide")


@st.cache_resource(show_spinner=False)
def load_pipeline() -> RAGPipeline:
    """Create the pipeline once per server process (failures are not cached)."""
    settings = load_settings()
    ensure_directories(settings)
    setup_logging(settings.log_dir, settings.log_level)
    return create_pipeline(settings)


def md(text: str) -> str:
    """Escape Markdown/LaTeX specials so '$' in finance text is not rendered as math."""
    return re.sub(r"([\\`*_{}\[\]()#+!|<>$~])", r"\\\1", text)


def flash(kind: str, message: str) -> None:
    st.session_state.setdefault("flash", []).append((kind, message))


def show_flash() -> None:
    for kind, message in st.session_state.pop("flash", []):
        getattr(st, kind)(message)


def render_answer(answer: Answer) -> None:
    if answer.error:
        st.error(answer.error)
        return
    if answer.sufficient:
        st.markdown(answer.text.replace("$", "\\$"))
    else:
        st.warning(answer.text)
    for warning in answer.warnings:
        st.caption(f"⚠️ {warning}")
    if answer.citations:
        st.markdown("**Sources**")
        for c in answer.citations:
            st.markdown(f"- \\[{c.index}\\] **{md(c.source)}**, {c.location}")
    if answer.evidence:
        with st.expander(f"Retrieved evidence ({len(answer.evidence)} passages)"):
            for i, hit in enumerate(answer.evidence, start=1):
                st.markdown(f"**\\[{i}\\] {md(hit.chunk.source)}**, {hit.chunk.location} · relevance {hit.score:.2f}")
                st.text(hit.chunk.text)  # st.text never interprets Markdown/HTML from documents
    st.caption(DISCLAIMER)


def sidebar(pipeline: RAGPipeline) -> list[str]:
    """Render the sidebar; returns the selected document ids (empty = all)."""
    settings = pipeline.settings
    st.session_state.setdefault("uploader_key", 0)
    with st.sidebar:
        st.header("📄 FinDOC AI")
        show_flash()

        st.subheader("Add documents")
        files = st.file_uploader(
            "PDF, DOCX or TXT",
            type=[e.lstrip(".") for e in SUPPORTED_EXTENSIONS],
            accept_multiple_files=True,
            key=f"uploader_{st.session_state.uploader_key}",
        )
        if st.button("Index documents", type="primary", disabled=not files, use_container_width=True):
            with st.spinner("Extracting, embedding and indexing…"):
                for f in files:
                    try:
                        rec = pipeline.index_file(f.name, f.getvalue())
                        flash("success", f"Indexed {rec.filename} ({rec.chunk_count} chunks).")
                    except FinDocError as exc:
                        flash("error", str(exc))
                    except Exception as exc:
                        flash("error", f"Could not index {f.name} ({type(exc).__name__}).")
            st.session_state.uploader_key += 1
            st.rerun()

        docs = pipeline.list_documents()
        labels = {d.doc_id: f"{d.filename} ({d.chunk_count} chunks)" for d in docs}
        st.subheader("Indexed documents")
        if not docs:
            st.info("No documents yet. Upload a file above to get started.")
        st.session_state["selected_docs"] = [d for d in st.session_state.get("selected_docs", []) if d in labels]
        selected = st.multiselect(
            "Search in", list(labels), format_func=labels.get, key="selected_docs",
            placeholder="All indexed documents", disabled=not docs,
        )
        if docs:
            with st.expander("Manage documents"):
                for d in docs:
                    st.markdown(f"**{md(d.filename)}**")
                    pages = f"{d.page_count} pages · " if d.page_count else ""
                    st.caption(f"{d.file_type.upper()} · {pages}{d.chunk_count} chunks · {d.size_bytes / 1024:.0f} KB")
                    cols = st.columns(2)
                    if not pipeline.is_current(d):
                        st.caption("⚠️ Indexed with a different embedding model.")
                        if cols[0].button("Re-index", key=f"re_{d.doc_id}"):
                            try:
                                with st.spinner("Re-indexing…"):
                                    pipeline.reindex_document(d.doc_id)
                                flash("success", f"Re-indexed {d.filename}.")
                            except FinDocError as exc:
                                flash("error", str(exc))
                            st.rerun()
                    if cols[1].button("Remove", key=f"rm_{d.doc_id}"):
                        try:
                            pipeline.remove_document(d.doc_id)
                            flash("success", f"Removed {d.filename} and its vectors.")
                        except FinDocError as exc:
                            flash("error", str(exc))
                        st.rerun()

        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        with st.expander("Configuration status"):
            st.write(f"**LLM:** {settings.llm_provider} / `{settings.llm_model}`")
            st.write(f"**Embeddings:** {settings.embedding_provider} / `{settings.embedding_model}` ({settings.embedding_dimensions}d)")
            st.write(f"**Vector store:** {settings.vector_store}")
            st.write(f"**Chunking:** {settings.chunk_size} chars, {settings.chunk_overlap} overlap, top-{settings.top_k}")
            st.write("**API key:** configured ✅")  # the value is never displayed
    return selected


def main() -> None:
    try:
        pipeline = load_pipeline()
    except FinDocError as exc:
        st.title("FinDOC AI")
        st.error(f"Setup problem: {exc}")
        st.info("Copy `Config/.env.example` to `Config/.env`, set `GEMINI_API_KEY`, then restart or refresh. See the README.")
        st.stop()

    selected = sidebar(pipeline)
    st.title("FinDOC AI")
    st.caption("Ask questions about your finance documents. Answers come only from retrieved passages, with citations.")
    st.session_state.setdefault("messages", [])

    has_docs = bool(pipeline.list_documents())
    if not has_docs:
        st.info("👈 Upload a PDF, DOCX or TXT finance document in the sidebar, then ask a question here.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "user":
                st.markdown(md(msg["content"]))
            else:
                render_answer(msg["answer"])

    prompt = st.chat_input("Ask a question about your documents", disabled=not has_docs)
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(md(prompt))
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and drafting an answer…"):
                answer = pipeline.answer(prompt, selected or None)
            render_answer(answer)
        st.session_state.messages.append({"role": "assistant", "answer": answer})


main()
