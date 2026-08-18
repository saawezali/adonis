"""Embedder wrapper: normalization, shapes, injectability."""

from __future__ import annotations

import numpy as np

from adonis.pair.embed import embed_texts
from tests.fakes import FakeEmbedder


def test_embed_texts_normalizes_to_unit_length():
    vecs = embed_texts(["hello world", "another claim"], model=FakeEmbedder(dim=8))
    assert vecs.shape == (2, 8)
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_embed_texts_passes_through_override_vectors():
    model = FakeEmbedder(
        dim=3, overrides={"Atlas ships in March.": [3.0, 0.0, 0.0]}
    )
    vecs = embed_texts(["Atlas ships in March."], model=model)
    assert np.allclose(vecs[0], [1.0, 0.0, 0.0])


def test_embed_texts_rejects_scalar_output():
    class BadEmbedder:
        def encode(self, texts, *, normalize_embeddings=False):
            return np.float32(1.0)

    try:
        embed_texts(["x"], model=BadEmbedder())  # type: ignore[arg-type]
    except ValueError:
        assert True
    else:
        raise AssertionError("expected ValueError for 0D output")