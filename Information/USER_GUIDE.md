# User guide

1. **Start:** `streamlit run Main/app.py` (see README for setup and the API key).
2. **Add documents:** in the sidebar choose PDF/DOCX/TXT files → **Index documents**. Progress and results appear in the sidebar.
3. **Choose scope:** "Search in" lists indexed documents. Leave it empty to search all of them.
4. **Ask:** type a specific question, e.g. "What was total revenue in FY2024?". Include the company/period if you have several documents.
5. **Read the result:** the answer cites passages as [1], [2]; **Sources** lists file and page/chunk; **Retrieved evidence** shows the exact text used. Check figures there.
6. **"Documents do not contain enough information"** means the evidence didn't support an answer. Rephrase, pick different documents, or add the missing one.
7. **Manage:** *Manage documents* → **Remove** deletes the file, vectors and metadata; **Re-index** appears after an embedding-model change. **Clear chat** resets the conversation.

Follow-up questions are not remembered by the model; restate what you mean. Answers are informational, not professional advice.
