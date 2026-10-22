"""
Day 6: knowledge base + retriever (the actual fix for Day 5's failure).

Recall Day 5's hard evidence: pasting a whole ~60,000-token document into
the prompt made the model silently lose most of it and confidently
hallucinate a wrong answer. RAG's fix is NOT "give the model more
context" -- it's "give the model LESS, but the RIGHT part."

RAG = Retrieval-Augmented Generation, in three stages:
  1. CHUNK   -- split every document into small pieces.
  2. RETRIEVE -- given a question, find the few chunks most likely to
                contain the answer (two ways shown below: keyword
                matching, then the better semantic/embedding approach).
  3. GENERATE -- hand ONLY those few chunks to the model, not the whole
                document, and let it answer from that small, relevant
                excerpt.

This file does stages 1 and 2. rag_app.py does stage 3 (and calls back
into this file for 1 and 2).
"""
import os
import re

import ollama

DOCS_DIR = os.path.join(os.path.dirname(__file__), "sample_docs")
EMBED_MODEL = "nomic-embed-text"

# A "chunk" is a small window of words. ~150 words is small enough to
# comfortably fit many chunks inside the model's context window, while
# still being a coherent piece of text (not a single sentence fragment).
CHUNK_SIZE_WORDS = 150
CHUNK_OVERLAP_WORDS = 30  # overlap avoids cutting an answer in half
                          # exactly at a chunk boundary.


# ------------------------------------------------------------------
# STAGE 1: CHUNKING
# ------------------------------------------------------------------
def chunk_text(text: str, filename: str) -> list[dict]:
    """Split one document's text into overlapping word-count chunks.
    Returns a list of {"filename", "text"} dicts -- plain Python, no AI."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + CHUNK_SIZE_WORDS
        chunk_words = words[start:end]
        chunks.append({"filename": filename, "text": " ".join(chunk_words)})
        if end >= len(words):
            break
        start = end - CHUNK_OVERLAP_WORDS
    return chunks


def load_knowledge_base() -> list[dict]:
    """Read every file in sample_docs/ and chunk it. This is the
    'knowledge base with multiple documents' from lecture 72."""
    all_chunks = []
    for filename in sorted(os.listdir(DOCS_DIR)):
        path = os.path.join(DOCS_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        all_chunks.extend(chunk_text(text, filename))
    return all_chunks


# ------------------------------------------------------------------
# STAGE 2a: KEYWORD RETRIEVER (lecture 75) -- the simple, first version.
# No AI at all here -- just counting shared words. Included so you can
# compare it against the semantic version below and see WHY embeddings
# are an improvement.
# ------------------------------------------------------------------
def keyword_retrieve(query: str, chunks: list[dict], top_k: int = 3) -> list[dict]:
    query_words = set(re.findall(r"\w+", query.lower()))

    def score(chunk):
        chunk_words = set(re.findall(r"\w+", chunk["text"].lower()))
        return len(query_words & chunk_words)  # how many words overlap

    scored = sorted(chunks, key=score, reverse=True)
    return scored[:top_k]


# ------------------------------------------------------------------
# STAGE 2b: SEMANTIC RETRIEVER (lectures 77-86) -- embeddings + cosine
# similarity. This is the real fix for the keyword retriever's biggest
# weakness: it only matches literal shared words, not MEANING.
# ------------------------------------------------------------------
def embed(text: str) -> list[float]:
    """Turn text into a list of numbers (a 'vector') that captures its
    MEANING. This is done by a separate, small pretrained model
    (nomic-embed-text) -- not the chat model (qwen3). Same idea as
    qwen3 itself: a model trained once, used many times, except this
    one's only job is "represent meaning as numbers," not "chat."""
    return ollama.embed(model=EMBED_MODEL, input=text)["embeddings"][0]


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """A pure-math measure of how similar two vectors' DIRECTIONS are
    (not their size). 1.0 = identical meaning, 0 = unrelated,
    -1.0 = opposite. No AI in this function -- just arithmetic."""
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(y * y for y in b) ** 0.5
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def build_semantic_index(chunks: list[dict]) -> list[dict]:
    """Pre-compute an embedding for every chunk, ONCE. This is the
    'index' -- expensive to build, cheap to search afterwards."""
    for chunk in chunks:
        chunk["embedding"] = embed(chunk["text"])
    return chunks


def semantic_retrieve(query: str, indexed_chunks: list[dict], top_k: int = 3) -> list[dict]:
    """The real retriever: embed the question, compare it against every
    chunk's pre-computed embedding, return the most similar ones."""
    query_vec = embed(query)
    scored = sorted(
        indexed_chunks,
        key=lambda c: cosine_similarity(query_vec, c["embedding"]),
        reverse=True,
    )
    return scored[:top_k]
