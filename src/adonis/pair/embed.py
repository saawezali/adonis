"""Claim embedding via sentence-transformers (PLAN.md M3).

The model loads lazily (first use downloads it unless cached). Callers may
inject any object with `encode(texts, normalize_embeddings=...)` for offline
tests. Embeddings are always L2-normalized so FAISS IndexFlatIP yields cosine
similarity.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from adonis.config import get_settings


class Embedder(Protocol):
    """Minimal contract: encode a list of texts into a 2-D float array."""

    def encode(
        self, texts: list[str], *, normalize_embeddings: bool = False
    ) -> np.ndarray:
        ...


_model: Embedder | None = None


def load_embedder(force: bool = False) -> Embedder:
    """Load the configured sentence-transformers model once and cache it."""
    global _model
    if _model is None or force:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(get_settings().embedding_model)
    assert _model is not None
    return _model


def embed_texts(texts: list[str], model: Embedder | None = None) -> np.ndarray:
    """Embed texts, normalized to unit length (rows of shape (n, d))."""
    model = model if model is not None else load_embedder()
    vectors = model.encode(list(texts), normalize_embeddings=True)
    arr = np.asarray(vectors, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"embedder returned {arr.ndim}D array; expected 2D")
    return arr
