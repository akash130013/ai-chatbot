# Day 6: Build Your First RAG System

This section directly fixes the failure proven in [Day 5](day-05-document-ai.md): pasting a whole ~60,668-token document into the prompt made the model silently hallucinate a wrong answer. **RAG's fix is not a bigger context window — it's sending less, but the right part.**

## Q1. What does RAG actually stand for, mechanically?
**R**etrieval-**A**ugmented **G**eneration, three stages, matching `day6/knowledge_base.py` + `day6/rag_app.py`:
1. **Retrieve** — given a question, find the few chunks of text most likely to contain the answer, out of potentially hundreds.
2. **Augment** — build a prompt containing *only* those few chunks.
3. **Generate** — send that small, focused prompt to the chat model, same as every prior day.

Nothing here retrains the model (same as every prior day's "instructing, not training" lesson) — it just changes *what text* gets handed to it.

## Q2. What is "chunking" and why do it?
Splitting every document into small overlapping pieces (`chunk_text()`, 150 words with 30-word overlap) *before* anything else. A whole 47,000-word document can't be searched or sent to the model as one unit — chunking turns it into hundreds of small, independently-matchable pieces. The overlap exists so an answer sitting right at a chunk boundary doesn't get cut in half.

## Q3. Keyword retrieval vs. semantic retrieval — what's the real difference?
Two retrievers are in `knowledge_base.py`, matching the course's own order (simple first, then the real fix):

- **`keyword_retrieve`** — counts literally shared words between the question and each chunk. No AI at all, just `set` overlap.
- **`semantic_retrieve`** — turns the question and every chunk into number-vectors (**embeddings**) using a separate small pretrained model (`nomic-embed-text`, not the chat model), then finds the closest match by **cosine similarity** (pure math, see Q4).

**Tested proof of the difference:** asked *"Who manages human resources at this company?"* — no word in that sentence (`manages`, `human`, `resources`) literally appears in the policy text (`HR Head: Priya Nair`). Semantic retrieval still found the right chunk, because `nomic-embed-text` represents **meaning**, not literal characters. Keyword matching would need the literal word "HR" to appear in the question to succeed.

## Q4. What is an embedding, and what is cosine similarity?
An **embedding** is a list of numbers (768 of them, for `nomic-embed-text`) that represents a piece of text's *meaning* as a point in space. Texts with similar meaning end up as nearby points, even with completely different words.

**Cosine similarity** (`cosine_similarity()`) is the pure-math formula that measures how close two of those points are — 1.0 = same meaning, 0 = unrelated. No AI happens in this function; it's arithmetic (dot product ÷ magnitudes), same as any vector-math you'd write in plain code.

## Q5. The decisive test: re-running Day 5's exact failure through RAG
Built a knowledge base of 390 chunks across two files: `company_policy.txt` and `big_report.txt` (the same realistic 46,668-word / ~60,668-token document from Day 5, with the secret code `ZX9471Q` buried on page 97).

| Approach | Result |
|---|---|
| Day 5: paste the whole 46,668-word doc directly | ❌ Hallucinated "98, 99, and 100" |
| Day 6: RAG — retrieve top 3 of 390 chunks, send only those (~450 words) | ✅ **"The secret code mentioned in the report is ZX9471Q."** |

Same model (`qwen3:1.7b`), same document, same question. The only difference: **what text actually reached the model.** This is the whole point of RAG, proven with real numbers, not just theory.

## Q6. Isn't building all those embeddings slow?
It's a **one-time cost**, not a per-question cost. Embedding all 390 chunks took **14.6 seconds**, and `rag_app.py` caches the result to disk (`indexed_chunks.pkl`, via `@st.cache_resource` + pickle) so it only happens once, not on every question. Each individual question afterward only needs **one new embedding** (the question itself) compared against the pre-built index — fast.

## Q7. How many chunks get sent to the model per question?
`top_k=3` in `answer_with_rag()` — only the 3 most relevant chunks out of all 390 are included in the prompt, roughly 450 words instead of 46,668. This is a tunable number: too few risks missing the answer; too many re-introduces the Day 5 overflow problem. 3–5 is a typical starting point in real systems.

## Interview one-liners
- "RAG doesn't retrain the model or expand its context window — it narrows what's sent to it, by retrieving only the chunks relevant to the current question."
- "Embeddings let retrieval match on meaning, not literal keywords — confirmed by a paraphrased query with zero shared words still finding the right chunk."
- "Index building (embedding every chunk) is a one-time, cacheable cost; only the query itself needs embedding per request."
