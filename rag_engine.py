"""PDF extraction, overlapping chunking, and FAISS semantic retrieval."""

import re
from typing import Any, Protocol

import faiss
import numpy as np
import pymupdf as fitz

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
NO_ANSWER = "I don't know based on the uploaded documents."
# IndexFlatIP over normalized embeddings returns cosine similarity. Sample-document
# evidence scored >= 0.253, while the unrelated Tesla question scored <= 0.058.
MIN_RELEVANCE_SCORE = 0.20
CHUNK_WORDS = 220
CHUNK_OVERLAP = 40
_STOP_WORDS = {
    "a", "about", "an", "and", "are", "as", "at", "be", "between", "by",
    "compare", "contrast", "describe", "did", "do", "does", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "the", "their", "this",
    "to", "was", "what", "when", "where", "which", "who", "why", "with",
}


class Embedder(Protocol):
    def encode(self, sentences: list[str], **kwargs: Any) -> Any: ...


def read_pdf(data: bytes, filename: str) -> list[dict[str, Any]]:
    """Extract page-aware overlapping chunks from a text-based PDF."""
    if not data:
        raise ValueError("The uploaded file is empty.")

    chunks: list[dict[str, Any]] = []
    try:
        document = fitz.open(stream=data, filetype="pdf")
    except (fitz.FileDataError, ValueError) as exc:
        raise ValueError("This file is not a readable PDF.") from exc

    with document:
        for page_number, page in enumerate(document, start=1):
            words = page.get_text().split()
            if not words:
                continue
            step = CHUNK_WORDS - CHUNK_OVERLAP
            for start in range(0, len(words), step):
                text = " ".join(words[start : start + CHUNK_WORDS]).strip()
                if len(text) >= 45:
                    chunks.append(
                        {
                            "text": text,
                            "source": filename,
                            "page": page_number,
                        }
                    )
                if start + CHUNK_WORDS >= len(words):
                    break

    if not chunks:
        raise ValueError("No selectable text was found; this may be a scanned PDF.")
    return chunks


def _encode(embedder: Embedder, texts: list[str]) -> np.ndarray:
    vectors = embedder.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[0] != len(texts) or vectors.shape[1] == 0:
        raise ValueError("The embedding model returned an invalid vector shape.")
    if not np.isfinite(vectors).all():
        raise ValueError("The embedding model returned non-finite values.")
    return np.ascontiguousarray(vectors)


def build_faiss_index(chunks: list[dict[str, Any]], embedder: Embedder) -> faiss.Index:
    """Embed all chunks and return an inner-product index over normalized vectors."""
    if not chunks:
        raise ValueError("Cannot index an empty chunk list.")
    vectors = _encode(embedder, [chunk["text"] for chunk in chunks])
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def retrieve(
    question: str,
    chunks: list[dict[str, Any]],
    index: faiss.Index,
    embedder: Embedder,
    top_k: int = 4,
    min_score: float = MIN_RELEVANCE_SCORE,
) -> list[dict[str, Any]]:
    """Return the highest-scoring passages, excluding weak semantic matches."""
    if not question.strip():
        raise ValueError("Enter a question before searching.")
    if not chunks or index.ntotal == 0:
        return []
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    query_vector = _encode(embedder, [question])
    if query_vector.shape[1] != index.d:
        raise ValueError("Question and document embeddings have different dimensions.")
    scores, positions = index.search(query_vector, min(top_k, len(chunks)))
    results: list[dict[str, Any]] = []
    for score, position in zip(scores[0], positions[0]):
        if position < 0 or float(score) < min_score:
            continue
        results.append({**chunks[int(position)], "score": float(score)})
    return results


def is_comparison_question(question: str) -> bool:
    return bool(
        re.search(
            r"\b(compare|contrast|versus|vs\.?|differences? between|analy[sz]\w*|impact|pros and cons|strengths? and weaknesses?)\b",
            question,
            re.IGNORECASE,
        )
    )


def summarize_passages(
    question: str,
    sources: list[dict[str, Any]],
    max_sentences: int = 3,
    embedder: Embedder | None = None,
) -> str:
    """Rank source sentences semantically and retain citations on each sentence."""
    if not sources:
        return NO_ANSWER

    candidates: list[tuple[int, int, str, str, int]] = []
    seen: set[str] = set()
    for source_order, source in enumerate(sources):
        sentences = re.split(r"(?<=[.!?])\s+", source["text"].strip())
        for sentence_order, sentence in enumerate(sentences):
            sentence = sentence.strip()
            normalized = " ".join(re.findall(r"\w+", sentence.lower()))
            if len(sentence.split()) < 4 or normalized in seen:
                continue
            seen.add(normalized)
            candidates.append(
                (
                    source_order,
                    sentence_order,
                    sentence,
                    source["source"],
                    source["page"],
                )
            )

    if not candidates:
        return NO_ANSWER

    query = " ".join(re.findall(r"\b[\w'-]+\b", question.lower()))
    definition = re.match(
        r"(?:what|who|where|when)\s+(?:is|are|was|were)\s+(?:(?:a|an|the)\s+)?(.+?)[?.!]*$",
        question.strip(),
        re.IGNORECASE,
    )
    if definition:
        focus_terms = [
            term.lower()
            for term in re.findall(r"\b[\w'-]+\b", definition.group(1))
            if len(term) > 1 and term.lower() not in _STOP_WORDS
        ]
        if focus_terms:
            sentence_terms = [
                set(re.findall(r"\b[\w'-]+\b", item[2].lower()))
                for item in candidates
            ]
            frequencies = {
                term: sum(term in terms for terms in sentence_terms)
                for term in focus_terms
            }
            present_terms = [term for term in focus_terms if frequencies[term] > 0]
            if present_terms:
                anchor = min(present_terms, key=lambda term: frequencies[term])
                candidates = [
                    item for item, terms in zip(candidates, sentence_terms)
                    if anchor in terms
                ]

    if embedder is None:
        query_terms = set(re.findall(r"\b[\w'-]+\b", query))
        similarities = np.asarray(
            [
                len(query_terms & set(re.findall(r"\b[\w'-]+\b", item[2].lower())))
                / max(len(query_terms), 1)
                for item in candidates
            ],
            dtype=np.float32,
        )
    else:
        vectors = _encode(embedder, [query, *(item[2] for item in candidates)])
        similarities = vectors[1:] @ vectors[0]
    ranked = [
        (float(score), *candidate)
        for score, candidate in zip(similarities, candidates)
        if float(score) >= max(0.30, float(similarities.max()) * 0.65)
    ]
    if not ranked:
        return NO_ANSWER

    comparison = re.search(
        r"(?:compare|contrast)\s+(.+?)\s+(?:and|with|to|against|vs\.?|versus)\s+(.+?)(?:[?.!]|$)",
        question,
        re.IGNORECASE,
    )
    if comparison is None:
        comparison = re.search(
            r"differences?\s+between\s+(.+?)\s+and\s+(.+?)(?:[?.!]|$)",
            question,
            re.IGNORECASE,
        )
    selected: list[tuple[float, int, int, str, str, int]] = []
    if comparison:
        sentence_terms = [
            set(re.findall(r"\b[\w'-]+\b", item[3].lower()))
            for item in ranked
        ]
        for concept in comparison.groups():
            terms = [
                term.lower()
                for term in re.findall(r"\b[\w'-]+\b", concept)
                if len(term) > 1 and term.lower() not in _STOP_WORDS
            ]
            if not terms:
                continue
            anchor = min(
                terms,
                key=lambda term: sum(term in words for words in sentence_terms),
            )
            match = next((item for item in ranked if anchor in set(
                re.findall(r"\b[\w'-]+\b", item[3].lower())
            )), None)
            if match is None:
                return NO_ANSWER
            if match not in selected:
                selected.append(match)

    for item in ranked:
        if len(selected) >= max_sentences:
            break
        if item not in selected:
            selected.append(item)

    selected.sort(key=lambda item: (item[1], item[2]))
    return " ".join(
        f"{sentence} [{filename}, page {page}]"
        for _, _, _, sentence, filename, page in selected[:max_sentences]
    )
