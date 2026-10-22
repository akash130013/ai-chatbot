"""
Day 6 exercise: a complete RAG application (browser version, Streamlit).

This is the fix for Day 5's hard failure: pasting a whole ~60,000-token
document made the model silently lose most of it and hallucinate a
confident wrong answer. The fix here is NOT "give it a bigger context
window" -- it's "stop sending the whole document in the first place."

The full RAG flow, every time you ask a question:
  1. RETRIEVE -- embed your question, compare it against every
     pre-computed chunk embedding (knowledge_base.semantic_retrieve),
     and pull out only the top few most relevant chunks.
  2. AUGMENT  -- build a prompt that includes ONLY those few chunks,
     not the whole document(s).
  3. GENERATE -- send that small, focused prompt to the chat model,
     which now answers from maybe 300-600 words of real, relevant
     text instead of 60,000+ words of mostly irrelevant text.

This is why "R-A-G" is the name: RETRIEVAL-AUGMENTED GENERATION.
"""
import hashlib
import os
import pickle

import ollama
import streamlit as st

from knowledge_base import DOCS_DIR, load_knowledge_base, build_semantic_index, semantic_retrieve

MODEL = os.getenv("MODEL", "qwen3:1.7b")
INDEX_CACHE = os.path.join(os.path.dirname(__file__), "indexed_chunks.pkl")

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using ONLY "
    "the excerpts provided below. If the excerpts don't contain the "
    "answer, say so honestly instead of guessing."
)


def _docs_fingerprint() -> str:
    """A short hash of every file's name + content + modified time in
    sample_docs/. If a SINGLE byte changes in ANY document -- a new
    file added, an edit, a deletion -- this value changes too. This is
    how we detect "the documents changed, the cached index is stale."""
    hasher = hashlib.sha256()
    for filename in sorted(os.listdir(DOCS_DIR)):
        path = os.path.join(DOCS_DIR, filename)
        hasher.update(filename.encode())
        hasher.update(str(os.path.getmtime(path)).encode())
    return hasher.hexdigest()


@st.cache_resource(show_spinner="Building knowledge base index...")
def get_index():
    """Building embeddings for every chunk is the slow, one-time cost of
    RAG. We cache it (to disk, and in Streamlit's memory) so it only
    happens once PER VERSION of the documents -- but if the documents
    change, the cache is thrown away and rebuilt automatically.

    Bug this fixes: without the fingerprint check, editing/adding a
    file in sample_docs/ would silently keep answering from the OLD,
    stale index forever, since nothing ever told the cache to expire."""
    current_fingerprint = _docs_fingerprint()

    if os.path.exists(INDEX_CACHE):
        with open(INDEX_CACHE, "rb") as f:
            cached = pickle.load(f)
        if isinstance(cached, dict) and cached.get("fingerprint") == current_fingerprint:
            return cached["chunks"]
        # Fingerprint mismatch (or an old-format cache with no fingerprint
        # at all) -- a document changed. Fall through and rebuild.

    chunks = load_knowledge_base()
    indexed = build_semantic_index(chunks)
    with open(INDEX_CACHE, "wb") as f:
        pickle.dump({"fingerprint": current_fingerprint, "chunks": indexed}, f)
    return indexed


def _mentioned_filename(question: str, index: list[dict]) -> str | None:
    """If the question literally names one of our real filenames (e.g.
    "check in facilities.txt"), return it. This is plain string
    matching -- no AI, no embeddings -- deliberately simple and
    guaranteed-reliable, unlike letting similarity search "guess" which
    file you meant (which we just proved fails for small files)."""
    known_filenames = {c["filename"] for c in index}
    question_lower = question.lower()
    for filename in known_filenames:
        if filename.lower() in question_lower:
            return filename
    return None


def answer_with_rag(question: str, top_k: int = 3) -> tuple[str, list[dict]]:
    index = get_index()

    # NEW: if the user named a specific file, don't gamble on similarity
    # search finding it among hundreds of chunks from OTHER files (we
    # proved this fails when one file dominates the corpus). Restrict
    # the search space to just that file's chunks first.
    named_file = _mentioned_filename(question, index)
    search_space = (
        [c for c in index if c["filename"] == named_file] if named_file else index
    )

    # STEP 1: RETRIEVE -- find the few most relevant chunks, out of
    # potentially hundreds, using meaning (embeddings), not keywords.
    top_chunks = semantic_retrieve(question, search_space, top_k=top_k)

    # STEP 2: AUGMENT -- build a prompt from ONLY those chunks.
    excerpts = "\n\n---\n\n".join(
        f"[From {c['filename']}]\n{c['text']}" for c in top_chunks
    )
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Excerpts:\n\n{excerpts}\n\nQuestion: {question}"},
    ]

    # STEP 3: GENERATE -- the model only ever sees these few small
    # excerpts, never the full 390-chunk knowledge base.
    response = ollama.chat(model=MODEL, messages=messages)
    return response["message"]["content"], top_chunks


# -------------------------------------------------------------------
st.set_page_config(page_title="RAG Demo", page_icon="🔎")
st.title("🔎 RAG Demo")
st.caption(f"Running locally with Ollama ({MODEL} + nomic-embed-text)")

st.sidebar.markdown(
    "**Knowledge base:** `company_policy.txt` + `big_report.txt` "
    "(a ~47,000-word document with one secret code buried on page 97 -- "
    "see Day 5 notes for why pasting this whole file broke the model)."
)
st.sidebar.markdown(
    "**Try asking:**\n"
    "- Who manages human resources at this company?\n"
    "- What is the secret code mentioned in the report?\n"
    "- How many days can I work remotely?"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant excerpts and thinking..."):
            try:
                reply, used_chunks = answer_with_rag(prompt)
                st.markdown(reply)
                with st.expander("See which excerpts were retrieved"):
                    for c in used_chunks:
                        st.markdown(f"**{c['filename']}**")
                        st.text(c["text"][:300])
                st.session_state.messages.append({"role": "assistant", "content": reply})
            except Exception as exc:
                st.error(f"{exc}\n\nIs Ollama running? Try: brew services start ollama")
