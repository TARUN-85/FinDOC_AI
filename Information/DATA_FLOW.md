# Data flow

## Indexing
1. User uploads files in the sidebar → `validate_upload` (extension, size, signature).
2. SHA-256 → `doc_id`; duplicates rejected.
3. `save_upload` writes a sanitized, non-colliding name into `InputDoc/`.
4. `load_document` extracts text per page (DOCX: whole document, tables included). Empty/scanned/encrypted files raise friendly errors.
5. `split_pages` → chunks with `source`, `page`, `chunk_index`.
6. `EmbeddingProvider.embed_documents` (batches of 50, retry with backoff on 429/5xx).
7. `VectorStore.add_chunks` → `DataBase/chroma/`; `DocumentRegistry.put` → `DataBase/documents.json`.
Any failure in steps 4–7 deletes the saved file and partial vectors.

## Question answering
1. Question + selected documents (empty selection = all current documents).
2. Stale documents (different embedding model) are skipped with a warning.
3. `embed_query` → Chroma cosine search filtered by `doc_id` → drop hits below `MIN_RELEVANCE`.
4. No hits → "not enough information" without calling the LLM.
5. Prompt = system rules + `<question>` + numbered `<evidence>` passages → `LLMProvider.generate`.
6. Sentinel → insufficient-evidence message. Otherwise `[n]` markers become verified citations; no valid markers → a warning.
7. UI shows answer, sources, expandable evidence, and the disclaimer.

## What leaves your machine
Chunk text (indexing), question text and retrieved passages (answering) go to the Gemini API. Nothing else is transmitted. Logs contain no document text or secrets.
