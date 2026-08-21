"""Final flag confidence calibration (PLAN.md M4).

A logistic regression is fit on eval-labeled pairs: features are the
judge's raw confidence and the candidate combined score; the target is
whether the pair is truly in a contradiction category. With fewer than
CONFIDENCE_MIN_SAMPLES labeled pairs a deterministic heuristic blend is
used instead (and always on degenerate targets).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass

import numpy as np

_FLAG_LABELS = {"genuine_contradiction", "superseded_by_time", "different_scope", "ambiguous"}
_TN_LABELS = {"true_negative_unrelated", "true_negative_near_dup"}

CONFIDENCE_MIN_SAMPLES = 10
_LR_ITERS = 500
_LR_STEP = 0.1
_LR_L2 = 1e-3


@dataclass
class CalibratedModel:
    """Transforms (judge_confidence, combined_score) into final confidence."""

    fitted: bool
    weights: np.ndarray | None = None
    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    def transform(self, judge_confidence: float, combined_score: float) -> float:
        x = np.asarray([judge_confidence, combined_score], dtype=np.float64)
        if not self.fitted or self.weights is None or self.mean is None or self.std is None:
            # Heuristic blend: judge says it's a flag and the pair ranked well.
            return float(
                np.clip(0.6 * judge_confidence + 0.4 * max(0.0, combined_score), 0.0, 1.0)
            )
        z = (x - self.mean) / (self.std + 1e-9)
        proba = 1.0 / (1.0 + np.exp(-float(self.weights @ z)))
        return float(np.clip(proba, 0.0, 1.0))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _fit_lr(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit standardized logistic regression by gradient descent."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    z = (X - mean) / (std + 1e-9)
    w = np.zeros(z.shape[1], dtype=np.float64)
    for _ in range(_LR_ITERS):
        grad = z.T @ (_sigmoid(z @ w) - y) / len(y) + _LR_L2 * w
        w -= _LR_STEP * grad
    return w, mean, std


def build_calibrated_model(conn: sqlite3.Connection) -> CalibratedModel:
    """Fit on labeled pairs that also have judge outputs; fall back otherwise."""
    rows = conn.execute(
        """
        SELECT el.label AS truth, jo.judge_confidence, cp.combined_score
        FROM eval_labels el
        JOIN claims c1 ON c1.id = el.claim_a_id
        JOIN claims c2 ON c2.id = el.claim_b_id
        JOIN candidate_pairs cp
             ON ((cp.claim_a_id = el.claim_a_id AND cp.claim_b_id = el.claim_b_id)
              OR (cp.claim_a_id = el.claim_b_id AND cp.claim_b_id = el.claim_a_id))
        JOIN judge_outputs jo ON jo.candidate_pair_id = cp.id
        """
    ).fetchall()
    usable = []
    for row in rows:
        truth: str = row["truth"]
        if truth in _FLAG_LABELS:
            usable.append((row["judge_confidence"], row["combined_score"], 1.0))
        elif truth in _TN_LABELS:
            usable.append((row["judge_confidence"], row["combined_score"], 0.0))
    if len(usable) < CONFIDENCE_MIN_SAMPLES:
        return CalibratedModel(fitted=False)
    X = np.asarray([[c, s] for c, s, _ in usable], dtype=np.float64)
    y = np.asarray([t for _, _, t in usable], dtype=np.float64)
    if int(y.sum()) == 0 or int(y.sum()) == len(y):
        return CalibratedModel(fitted=False)
    weights, mean, std = _fit_lr(X, y)
    return CalibratedModel(fitted=True, weights=weights, mean=mean, std=std)