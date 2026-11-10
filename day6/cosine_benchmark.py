"""
Standalone script: pure-Python cosine similarity vs NumPy, compared for
correctness AND speed, using your real 395-chunk RAG knowledge base.

Run it yourself:
    cd day6
    source ../.venv/bin/activate
    python cosine_benchmark.py

Nothing in knowledge_base.py or rag_app.py is changed by this file.
"""
import pickle
import time

import numpy as np

from knowledge_base import cosine_similarity as cosine_similarity_pure, embed


# The NumPy version -- same formula as knowledge_base.cosine_similarity,
# just written with a library instead of raw Python loops.
def cosine_similarity_np(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def main() -> None:
    # Reuse the real, already-built index from rag_app.py if it exists,
    # instead of re-embedding 395 chunks again (which costs ~15s on its
    # own and has nothing to do with what we're actually testing here).
    try:
        with open("indexed_chunks.pkl", "rb") as f:
            cached = pickle.load(f)
        chunks = cached["chunks"] if isinstance(cached, dict) else cached
    except FileNotFoundError:
        print("No cached index found. Run rag_app.py once first, or")
        print("this script will just test correctness on a small example.")
        chunks = None

    # --- Step 1: prove both functions agree -----------------------------
    print("=== Correctness check ===")
    v1 = embed("What is the secret code?")
    v2 = embed("The secret code for this test is ZX9471Q.")
    pure_result = cosine_similarity_pure(v1, v2)
    numpy_result = cosine_similarity_np(v1, v2)
    print(f"Pure Python: {pure_result}")
    print(f"NumPy:       {numpy_result}")
    print(f"Match (within floating-point rounding): "
          f"{abs(pure_result - numpy_result) < 1e-10}")
    print()

    if not chunks:
        return

    # --- Step 2: benchmark speed across the WHOLE knowledge base --------
    # This is the realistic test: comparing one question's embedding
    # against every chunk's embedding, exactly what semantic_retrieve()
    # does on every question you ask the app.
    query_vec = embed("What is the secret code mentioned in the report?")
    all_vecs = [c["embedding"] for c in chunks]
    print(f"=== Speed over {len(all_vecs)} chunks ===")

    t0 = time.perf_counter()
    for v in all_vecs:
        cosine_similarity_pure(query_vec, v)
    pure_time = time.perf_counter() - t0
    print(f"Pure Python loop: {pure_time*1000:.2f} ms")

    t0 = time.perf_counter()
    for v in all_vecs:
        cosine_similarity_np(query_vec, v)
    numpy_time = time.perf_counter() - t0
    print(f"NumPy (one at a time): {numpy_time*1000:.2f} ms")

    # The REAL NumPy advantage: comparing against ALL chunks in one go,
    # instead of looping and calling the function once per chunk.
    t0 = time.perf_counter()
    matrix = np.array(all_vecs)                      # shape: (395, 768)
    q = np.array(query_vec)                          # shape: (768,)
    sims = matrix @ q / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(q))
    vectorized_time = time.perf_counter() - t0
    print(f"NumPy (vectorized, whole matrix at once): {vectorized_time*1000:.2f} ms")

    print()
    print(f"Vectorized NumPy was {pure_time/vectorized_time:.1f}x faster "
          f"than the pure Python loop.")


if __name__ == "__main__":
    main()
