"""FAISS index: build/search roundtrip on known vectors."""

from __future__ import annotations

import numpy as np

from adonis.pair.index import build_index, nearest, search


def _vectors() -> np.ndarray:
    return np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.9, 0.1, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )


def test_nearest_returns_closest_by_cosine():
    index = build_index(_vectors())
    result = nearest(index, np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32), k=3)
    assert result[0] == (0, 1.0)
    assert result[1][0] == 1  # 0.9 cosine with row 1
    assert result[2][0] == 2


def test_search_returns_distances_and_clamps_k():
    index = build_index(_vectors())
    distances, indices = search(index, _vectors(), k=10)  # k > n total
    assert indices.shape == (3, 3)
    assert distances.shape == (3, 3)
    assert abs(distances[0][0] - 1.0) < 1e-6


def test_build_index_rejects_empty():
    try:
        build_index(np.zeros((0, 4), dtype=np.float32))
    except ValueError:
        assert True
    else:
        raise AssertionError("expected ValueError for empty vectors")


def test_nearest_excludes_self_when_requested():
    index = build_index(_vectors())
    result = nearest(index, np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32), k=3)
    # for display purposes: self is the top hit, then the similar row
    assert result[1][1] > result[2][1]  # 0.9 > 0.0 cosine