"""Confidence calibration: logistic fit, fallback, degenerate guards."""

from __future__ import annotations

import numpy as np

from adonis.score.confidence import CalibratedModel, build_calibrated_model


def _make_model() -> CalibratedModel:
    return CalibratedModel(
        fitted=True,
        weights=np.asarray([1.0, 2.0]),
        mean=np.asarray([0.5, 0.5]),
        std=np.asarray([0.25, 0.25]),
    )


def test_fallback_blend_without_fit():
    model = CalibratedModel(fitted=False)
    assert model.transform(1.0, 1.0) == 1.0  # clipped
    assert model.transform(0.0, 0.0) == 0.0
    assert model.transform(1.0, 0.5) == 0.8  # 0.6 * 1.0 + 0.4 * 0.5
    mid = model.transform(0.5, 0.5)
    assert 0.4 <= mid <= 0.6


def test_fit_transform_is_monotone_in_features():
    model = _make_model()
    low = model.transform(0.0, 0.0)
    high = model.transform(1.0, 1.0)
    assert 0.0 <= low < high <= 1.0


def test_build_without_labels_falls_back(tmp_env):
    from adonis.db import apply_migrations, get_conn

    apply_migrations()
    conn = get_conn()
    try:
        model = build_calibrated_model(conn)
        assert model.fitted is False
    finally:
        conn.close()


def test_build_with_few_samples_falls_back(tmp_env):
    from adonis.db import apply_migrations, get_conn

    apply_migrations()
    conn = get_conn()
    try:
        model = build_calibrated_model(conn)
        assert model.fitted is False
    finally:
        conn.close()