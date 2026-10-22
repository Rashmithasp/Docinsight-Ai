# DocInsight AI

**From Text to Insight — Ask Your Documents**

DocInsight AI is a lightweight Streamlit RAG app. Upload multiple text-based PDFs, ask a question, and review the actual page passages used to answer it. Gemini is optional; without a key, the app provides a clearly labelled extractive passage instead.

## Quick start (Windows PowerShell)

From this project directory:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

If Python 3.12 is not installed, replace `-3.12` with an installed Python 3.10+ version. On first processing, the app downloads `all-MiniLM-L6-v2` into your user cache at `.cache/docinsight-ai`; allow a moment and an internet connection for this initial model download. The embedding model then runs locally.

### Gemini answers (optional)

Create a Gemini API key in Google AI Studio and enter it in the app's sidebar. The password field is session-only; the key is not written to project files. Alternatively, set `GEMINI_API_KEY` in your shell before starting Streamlit. Never commit a real key.

With no key, the app returns the best matching retrieved passage and its source as an **Extractive fallback**. It does not present that passage as a generated answer.

## Using the app

1. Upload one or more text-based PDFs.
2. Click **Process documents** and wait for extraction, embedding, and FAISS indexing to finish.
3. Ask a document-specific question in the chat box.
4. Inspect the answer's expandable supporting passage(s), filename, and page number.
5. Ask a question not covered by the PDFs to verify abstention. Clear the chat to restart the conversation.

Scanned/image-only PDFs need OCR first. Keep the PDFs open during the session; document indexes and chat history are held in Streamlit session state and are not persisted.

## RAG architecture

```text
PDF uploads
   │
   ▼
PyMuPDF extracts each page separately
   │  source filename + 1-based page retained
   ▼
220-word chunks, 40-word overlap (within each page)
   │
   ▼
all-MiniLM-L6-v2 sentence embeddings (normalized)
   │
   ▼
FAISS IndexFlatIP cosine-style semantic ranking
   │  up to 4 passages; low-relevance matches are filtered
   ▼
Gemini 2.5 Flash grounded prompt ── or ── labelled extractive fallback
   │
   ▼
Answer + expandable passages with actual PDF filename/page citations
```

Gemini is instructed to use only retrieved evidence and to return **“I don't know based on the uploaded documents.”** if the evidence is insufficient. A semantic relevance threshold also prevents very weak retrieval results from being sent as evidence. RAG is not a guarantee of correctness: check the cited passages, particularly for consequential decisions.

## Five-question test sheet

See [test_questions.md](./test_questions.md) for five executed retrieval checks against a temporary two-page fixture PDF. Re-run the questions against the actual documents used in your demo; answers and relevance scores depend on the uploaded PDFs.

## 60-second demo script

> “DocInsight AI turns PDFs into a searchable knowledge base. I’ll upload two documents and process them. The app extracts text page by page, keeps each page citation attached to overlapping chunks, embeds them with MiniLM, and indexes those vectors in FAISS. I’ll ask a question now: with Gemini configured, the generated answer uses only the top matching passages; without a key, the app labels its best passage as an extractive fallback. You can expand each citation to verify the exact text and page. Finally, I’ll ask something these documents don’t contain; the app abstains instead of making up an answer.”

## Streamlit Community Cloud

1. Push the **contents** of `docinsight_starter` to a GitHub repository root (do not include `.env`, API keys, or `.venv`).
2. In [Streamlit Community Cloud](https://share.streamlit.io/), create an app and select this repository with `app.py` as the main file.
3. Add `GEMINI_API_KEY` under the app's **Settings → Secrets** if generated answers are desired, using TOML:

   ```toml
   GEMINI_API_KEY = "your-key"
   ```

4. Deploy. On its first run, the app downloads the sentence-transformer model; deployment therefore needs outbound access and enough build/runtime memory. For a demo without Gemini, leave the secret unset.

## Troubleshooting

- **No readable text:** the PDF may contain only scanned images; OCR it first.
- **Model download or memory issue:** first-time embedding-model setup needs internet and memory. Try a smaller number of pages or PDFs.
- **Gemini error:** verify the API key and available model/API access. The app surfaces the error; retry without the key to use the extractive mode.
- **FAISS install issue:** use a supported 64-bit Python 3.10+ environment and reinstall from `requirements.txt`.
