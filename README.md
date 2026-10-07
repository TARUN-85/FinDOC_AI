# FinDOC AI

A Streamlit app for asking questions about finance documents. Upload PDF, DOCX or TXT files, index them with
retrieval-augmented generation (RAG), and get answers drawn **only** from the retrieved passages, with citations.

Gemini (a hosted model reached through the Gemini API, not open source) is the default provider for both generation and embeddings.

## Limitations
- Informational only. Not investment, tax, legal or accounting advice. Always verify figures against the source.
- No OCR: scanned/image-only PDFs have no extractable text and are rejected with a message.
- DOCX files have no page numbers (citations use chunk numbers). Chunks never span pages, so facts split across a page break may be missed.
- Each question is answered independently; chat history is **not** sent to the model, so make follow-ups self-contained.
- Local Chroma storage targets one user or a small team on one machine.
- Tables in PDFs may extract imperfectly; check the evidence panel for numeric answers.

## Requirements
Python **3.12** and a Gemini API key from <https://aistudio.google.com/apikey>.

## Setup

**macOS / Linux**
```bash
cd findoc-ai
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp Config/.env.example Config/.env      # then edit Config/.env and set GEMINI_API_KEY
streamlit run Main/app.py
```

**Windows PowerShell**
```powershell
cd findoc-ai
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1            # if blocked: Set-ExecutionPolicy -Scope Process RemoteSigned
pip install -r requirements.txt
Copy-Item Config\.env.example Config\.env   # then edit Config\.env and set GEMINI_API_KEY
streamlit run Main/app.py
```

**Setting the key safely:** put it in `Config/.env` (git-ignored) or export it for one session
(`export GEMINI_API_KEY=...` / `$env:GEMINI_API_KEY="..."`). Never put it in `settings.yaml` (the app refuses) or commit it.
The key is never shown in the UI or written to logs (logs also redact key-like strings).

Run tests (no key or network needed): `python -m pytest`

## Configuration precedence
1. Process environment variables
2. `Config/.env` (then `.env` in the project root)
3. `Config/settings.yaml`
4. Built-in defaults

Blank values are treated as unset. Relative paths resolve against the project root, not the working directory.

| Variable | Default | Notes |
|---|---|---|
| `GEMINI_API_KEY` | – | Required for Gemini; environment only |
| `LLM_PROVIDER` / `LLM_MODEL` | `gemini` / `gemini-3.8-flash` | See model note below |
| `LLM_TEMPERATURE` | unset | Blank = provider default (recommended for Gemini 3) |
| `EMBEDDING_PROVIDER` / `EMBEDDING_MODEL` | `gemini` / `gemini-embedding-001` | Separate from the LLM |
| `EMBEDDING_DIMENSIONS` | `768` | 128–3072 |
| `VECTOR_STORE` | `chroma` | |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | Characters |
| `TOP_K` / `MIN_RELEVANCE` | `5` / `0.30` | Passages retrieved; cosine similarity floor |
| `MAX_UPLOAD_MB`, `LOG_LEVEL` | `25`, `INFO` | |

### Model choices (verified against Google's docs on 2026-10-07)
- **LLM default is `gemini-3.8-flash`, not `gemini-2.5-flash`.** Google's deprecation page currently says access to 2.5 models is limited to users who already used them and recommends newer models for new projects, and earlier reports indicated a retirement date in October 2026. If you already have access you can set `LLM_MODEL=gemini-2.5-flash`. Model availability changes often; if you see a "model not found" message, check <https://ai.google.dev/gemini-api/docs/deprecations>.
- **Embeddings default to `gemini-embedding-001`** (text, stable; listed shutdown not before May 2028). `gemini-embedding-2` is the newer multimodal model and is also supported (set `EMBEDDING_MODEL=gemini-embedding-2`); it uses task prefixes instead of `task_type`, which the code handles.
- **Switching embedding models makes old vectors incompatible.** Documents show a warning and are skipped in search until you click **Re-index** (the original file is kept in `InputDoc/`).
- Not verified: I could not call the live API in the build environment, so network behavior (quota, exact error text) is untested; SDK types and configuration objects were checked against `google-genai`.

### Adding a provider
Subclass `LLMProvider` (`Main/llm_provider.py`) and/or `EmbeddingProvider` (`Main/embeddings.py`), add a factory to that file's `_FACTORIES` dict, add any key to `Settings` (read from environment only), then set `LLM_PROVIDER` / `EMBEDDING_PROVIDER`. Unknown providers or missing credentials fail at startup with a clear message.

### Replacing the vector database
Implement `VectorStore` (`add_chunks`, `query`, `delete_document`, `count`) in `Main/vector_store.py` for pgvector, Qdrant, Pinecone, etc., and register it in `create_vector_store`. Return cosine similarity in 0–1 as the score.

## How indexing and storage work
- Uploads are validated (extension, size, file signature), sanitized, and saved to `InputDoc/` without overwriting (`name_1.pdf`, …). Identical content is detected and not re-indexed.
- Text is extracted, split per page into overlapping chunks, embedded, and stored in `DataBase/chroma/`; metadata is in `DataBase/documents.json`.
- Removing a document deletes its vectors, stored file and metadata. Logs go to `Logs/findoc.log` (no document text, prompts or secrets).

## Project layout
As requested, plus: `Main/gemini_client.py` (shared client, retry and error translation), `tests/conftest.py` (test fakes), `.gitkeep` files.

## Troubleshooting
| Symptom | Fix |
|---|---|
| "GEMINI_API_KEY is required" | Create `Config/.env` from the example and set the key; restart |
| "Gemini rejected the API key" | Regenerate the key; check it has access |
| "rate limit or quota reached" | Wait, or review your quota/plan; the app retries automatically a few times |
| "could not find the configured model" | Change `LLM_MODEL` / `EMBEDDING_MODEL` (see model note) |
| "No text could be extracted" | Scanned PDF – run OCR first |
| Documents flagged "different embedding model" | Click **Re-index** |
| Answers say documents lack information | Select the right documents, rephrase, or lower `MIN_RELEVANCE` / raise `TOP_K` |
| `ModuleNotFoundError` | Activate the venv and re-run `pip install -r requirements.txt` |

## Privacy
Documents, vectors and metadata stay on your machine. To index and answer, the **text of your chunks and your questions are sent to the
Gemini API**, subject to Google's terms and data-use policy for your account type (paid and free tiers differ). Do not upload documents you
are not permitted to share with that service.
