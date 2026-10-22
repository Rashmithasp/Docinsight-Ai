import os
from pathlib import Path
from typing import Any

import streamlit as st
from google.genai.errors import APIError
from huggingface_hub import snapshot_download
from huggingface_hub.errors import HfHubHTTPError
from sentence_transformers import SentenceTransformer
from streamlit.errors import StreamlitSecretNotFoundError

from rag_engine import (
    DEFAULT_MODEL_NAME,
    NO_ANSWER,
    MIN_RELEVANCE_SCORE,
    build_faiss_index,
    retrieve,
    read_pdf,
    summarize_passages,
    is_comparison_question,
)


st.set_page_config(
    page_title="DocInsight AI",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root { --navy: #10243a; --teal: #12a89d; --paper: #f4f8fb; }
    .stApp { background: var(--paper); color: var(--navy); }
    [data-testid="stHeader"] { background: transparent; }
    [data-testid="stSidebar"] { background: #10243a; }
    [data-testid="stSidebar"] * { color: #f4f8fb; }
    .hero {
        background: linear-gradient(115deg, #10243a, #17465a);
        padding: 1.5rem 1.8rem; border-radius: 16px; color: white;
        margin: 0.2rem 0 1rem 0;
    }
    .hero h1 { color: white; margin: 0; font-size: 2.15rem; }
    .hero p { color: #cce7e8; margin: .35rem 0 0; font-size: 1.05rem; }
    div.stButton > button[kind="primary"] {
        background: #12a89d; border-color: #12a89d; color: white;
        font-weight: 650; border-radius: 9px;
    }
    [data-testid="stMetric"] {
        background: white; padding: .8rem 1rem; border-radius: 12px;
        border: 1px solid #e2ebf0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading the semantic embedding model…")
def load_embedding_model(model_name: str) -> SentenceTransformer:
    model_dir = Path.home() / ".cache" / "docinsight-ai" / model_name.rsplit("/", 1)[-1]
    local_model_path = snapshot_download(model_name, local_dir=str(model_dir))
    return SentenceTransformer(local_model_path)


def init_state() -> None:
    defaults: dict[str, Any] = {
        "chunks": [],
        "faiss_index": None,
        "embedder": None,
        "processed_names": [],
        "chat_history": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def configured_api_key() -> str:
    environment_key = os.getenv("GEMINI_API_KEY", "")
    try:
        return st.secrets.get("GEMINI_API_KEY", environment_key)
    except StreamlitSecretNotFoundError:
        return environment_key


def extractive_answer(
    question: str,
    sources: list[dict[str, Any]],
    embedder: SentenceTransformer | None = None,
    answer_type: str = "Extractive answer (extractive, not AI-generated)",
) -> tuple[str, str]:
    comparison = is_comparison_question(question)
    answer = summarize_passages(
        question,
        sources,
        max_sentences=5 if comparison else 3,
        embedder=embedder,
    )
    if answer == NO_ANSWER:
        return NO_ANSWER, "Grounded response"
    return answer, answer_type


def answer_question(
    question: str,
    sources: list[dict[str, Any]],
    api_key: str,
    embedder: SentenceTransformer | None = None,
) -> tuple[str, str]:
    if not sources:
        return NO_ANSWER, "Grounded response"

    api_key = api_key.strip()
    if not api_key:
        return extractive_answer(question, sources, embedder)

    from google import genai

    evidence = "\n\n".join(
        f"[Source {i}: {item['source']}, page {item['page']}]\n{item['text']}"
        for i, item in enumerate(sources, start=1)
    )
    prompt = f"""You answer questions about uploaded documents.
Use only the evidence below. Do not use prior knowledge or make inferences unsupported
by the evidence. If it does not contain the answer, reply exactly:
"{NO_ANSWER}"
For each factual claim, cite the source using its exact filename and page number.
Do not invent or alter source names or page numbers. Keep the answer concise and under
100 words. For factual questions, use 2–3 sentences; for comparison or analytical
questions, use 4–6 relevant sentences when the evidence supports them.

Question: {question}

Evidence:
{evidence}
"""
    try:
        with genai.Client(api_key=api_key) as client:
            available_models = [
                model.name.removeprefix("models/")
                for model in client.models.list()
                if model.name
                and "generateContent" in (model.supported_actions or [])
            ]
            preferred_model = "gemini-3.8-flash"
            model_name = (
                preferred_model
                if preferred_model in available_models
                else next(
                    (name for name in available_models if "flash" in name.lower()),
                    None,
                )
            )
            if model_name is None:
                return extractive_answer(
                    question,
                    sources,
                    embedder,
                    "Extractive fallback (Gemini model unavailable; not AI-generated)",
                )
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
    except APIError:
        return extractive_answer(
            question,
            sources,
            embedder,
            "Extractive fallback (Gemini API error; not AI-generated)",
        )

    text = response.text
    if not text or not text.strip():
        return extractive_answer(
            question,
            sources,
            embedder,
            "Extractive fallback (Gemini returned no answer; not AI-generated)",
        )
    return text.strip(), "Gemini grounded answer"


def render_sources(sources: list[dict[str, Any]]) -> None:
    if not sources:
        return
    st.markdown("**Retrieved supporting passages**")
    for item in sources:
        label = (
            f"{item['source']} — page {item['page']} "
            f"· relevance {item['score']:.2f}"
        )
        with st.expander(label):
            st.write(item["text"])


init_state()
st.markdown(
    '<div class="hero"><h1>DocInsight AI</h1>'
    '<p>From Text to Insight · Ask your documents, and inspect the evidence.</p></div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.subheader("Workspace")
    api_key = st.text_input(
        "Gemini API key",
        type="password",
        value=configured_api_key(),
        help="Used only for this session. You can leave it blank for extractive answers.",
    )
    st.caption("Your key is not saved to the project or displayed.")
    st.divider()
    st.markdown("**How it works**")
    st.caption("PDF pages → overlapping chunks → semantic search → cited answer")
    st.caption(f"Embedding model: `{DEFAULT_MODEL_NAME}`")

with st.container(border=True):
    st.subheader("1 · Add your documents")
    uploads = st.file_uploader(
        "Choose one or more text-based PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Scanned/image-only PDFs need OCR before their text can be searched.",
    )
    left, right = st.columns([1, 2])
    with left:
        process_clicked = st.button(
            "Process documents",
            type="primary",
            use_container_width=True,
            disabled=not uploads,
        )
    with right:
        st.caption("Files are processed in memory for this session. Reprocess after changing the selection.")

if process_clicked and uploads:
    st.session_state.chunks = []
    st.session_state.faiss_index = None
    st.session_state.embedder = None
    st.session_state.processed_names = []
    st.session_state.chat_history = []
    extracted: list[dict[str, Any]] = []
    processed_names: list[str] = []
    failed: list[str] = []
    progress = st.progress(0, text="Reading PDF pages…")
    for position, uploaded in enumerate(uploads, start=1):
        try:
            file_chunks = read_pdf(uploaded.getvalue(), uploaded.name)
            extracted.extend(file_chunks)
            processed_names.append(uploaded.name)
        except (OSError, RuntimeError, ValueError) as exc:
            failed.append(f"{uploaded.name}: {exc}")
        progress.progress(position / len(uploads), text=f"Reading {uploaded.name}…")

    if failed:
        for error in failed:
            st.error(error)
    if not extracted:
        progress.empty()
        st.error("No readable text was extracted. Check the PDFs or OCR scanned pages first.")
    else:
        try:
            with st.spinner("Loading the embedding model and building the FAISS index…"):
                embedder = load_embedding_model(DEFAULT_MODEL_NAME)
                index = build_faiss_index(extracted, embedder)
            st.session_state.chunks = extracted
            st.session_state.faiss_index = index
            st.session_state.embedder = embedder
            st.session_state.processed_names = processed_names
            st.session_state.chat_history = []
            progress.empty()
            st.success(
                f"Ready: {len(st.session_state.processed_names)} PDF(s), "
                f"{len(extracted)} searchable chunks."
            )
        except (HfHubHTTPError, ImportError, OSError, RuntimeError, ValueError) as exc:
            progress.empty()
            st.error(f"Document indexing failed: {exc}")

chunks = st.session_state.chunks
metrics = st.columns(3)
metrics[0].metric("Documents", len(st.session_state.processed_names))
metrics[1].metric("Text chunks", len(chunks))
metrics[2].metric("Max passages per question", 6 if chunks else 0)

st.subheader("2 · Ask your documents")
if not chunks:
    st.info("Process at least one PDF to start asking questions.")
else:
    for entry in st.session_state.chat_history:
        with st.chat_message(entry["role"]):
            if entry["role"] == "assistant":
                st.caption(entry.get("answer_type", ""))
            st.markdown(entry["content"])
            if entry["role"] == "assistant":
                render_sources(entry.get("sources", []))

    question = st.chat_input("Ask a question grounded in your PDFs…")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching your documents…"):
                    sources = retrieve(
                        question,
                        chunks,
                        st.session_state.faiss_index,
                        st.session_state.embedder,
                        top_k=6 if is_comparison_question(question) else 4,
                        min_score=MIN_RELEVANCE_SCORE,
                    )
                answer, answer_type = answer_question(
                    question, sources, api_key, st.session_state.embedder
                )
                st.caption(answer_type)
                st.markdown(answer)
                render_sources(sources)
                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "answer_type": answer_type,
                        "sources": sources,
                    }
                )
            except (APIError, ImportError, OSError, RuntimeError, ValueError) as exc:
                st.error(f"Could not answer this question: {exc}")

    if st.button("Clear chat"):
        st.session_state.chat_history = []
        st.rerun()

st.caption(
    "Answers are limited to retrieved passages. Verify citations against the source PDF, "
    "especially for high-stakes decisions."
)
