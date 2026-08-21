"""FAISS index over claim embeddings.

IndexFlatIP (exact inner-product search) is the default per design spec section 8;
switch to IVFFlat only if claim count exceeds ~50k. All vectors are L2-
normalized upstream, so inner product == cosine similarity.
"""

from __future__ import annotations

from typing import Any

import faiss
import numpy as np

import adonis.pair.embed as embed_mod


def build_index(vectors: np.ndarray) -> Any:
    """Build a FAISS IndexFlatIP over (n, d) float32 rows. Untyped: faiss has
    no stubs."""
    if vectors.ndim != 2:
        raise ValueError(f"expected 2D vectors, got {vectors.ndim}D")
    if len(vectors) == 0:
        raise ValueError("cannot index an empty vector set")
    index = faiss.IndexFlatIP(int(vectors.shape[1]))
    index.add(np.ascontiguousarray(vectors, dtype=np.float32))
    return index


def search(index: Any, query: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (distances, indices) for the top-k neighbors of each query row.

    k is clamped to the index size. `query` must be (m, d) with the same d as
    the index. Similarity is cosine (0..1) for normalized vectors.
    """
    if query.ndim != 2:
        raise ValueError(f"expected 2D query, got {query.ndim}D")
    k = max(1, min(k, int(index.ntotal)))
    distances, indices = index.search(np.ascontiguousarray(query, dtype=np.float32), k)
    return distances, indices


def nearest(
    index: Any, query: np.ndarray, k: int
) -> list[tuple[int, float]]:
    """Top-k neighbors of a single query as (index_position, cosine) pairs."""
    if query.ndim == 1:
        query = query.reshape(1, -1)
    distances, indices = search(index, query, k)
    row = list(zip(indices[0].tolist(), distances[0].tolist()))
    return [(int(i), float(d)) for i, d in row if i != -1]


def embed_and_index(
    texts: list[str], model: embed_mod.Embedder | None = None
) -> tuple[Any, np.ndarray]:
    """Embed a batch and return (index, vectors). One call for the pipeline."""
    vectors = embed_mod.embed_texts(texts, model=model)
    return build_index(vectors), vectors
