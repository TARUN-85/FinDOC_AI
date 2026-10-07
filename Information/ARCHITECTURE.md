# Architecture

```
Streamlit UI (app.py)
      │
RAGPipeline (rag_pipeline.py) ── orchestrates everything
 ├─ document_loader.py   validate/save/extract PDF, DOCX, TXT
 ├─ text_splitter.py     overlapping chunks + page metadata
 ├─ EmbeddingProvider    embeddings.py  (Gemini; pluggable)
 ├─ VectorStore          vector_store.py (Chroma; pluggable) + DocumentRegistry (JSON)
 ├─ Retriever            retriever.py   (embed question, search, relevance floor)
 └─ LLMProvider          llm_provider.py (Gemini; pluggable)
Shared: config.py (settings), logger.py (redacting logs), gemini_client.py (retry/errors), schemas.py
```

## Key decisions
- **Provider interfaces** keep generation and embeddings independent; the pipeline never imports the Gemini SDK.
- **One Chroma collection per embedding signature** (`provider:model:dimensions`) so vectors of different sizes never mix. Deletes sweep all collections.
- **doc_id = SHA-256 prefix of file bytes** gives free duplicate detection.
- **Grounding:** the model sees only the question and retrieved passages; a sentinel (`INSUFFICIENT_EVIDENCE`) signals missing evidence; citations are built only from passage numbers that actually exist, so fabricated citations are dropped.
- **Untrusted documents:** passages are delimited, delimiter tags are stripped from them, and the system instruction tells the model to treat them as data.
- **No chat history** is sent to the model.
- **Failure handling:** indexing rolls back (file, vectors) on any error; provider errors become short user-facing messages; the UI never shows a traceback.

## Replacing components
See README: "Adding a provider" and "Replacing the vector database".
