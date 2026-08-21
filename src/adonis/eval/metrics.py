"""Eval metrics over labeled pairs.

Pure functions: per-category precision/recall/F1 from (truth, prediction)
tuples, micro/macro averages, and citation-faithfulness. Predictions are
the judge labels; None means the pair was never judged (a missed
contradiction counts as a false negative, a true-negative label counts
as nothing).
"""

from __future__ import annotations

from dataclasses import dataclass, field

CONTRADICTION_CATEGORIES = [
    "genuine_contradiction",
    "superseded_by_time",
    "different_scope",
    "ambiguous",
]
TRUE_NEGATIVE_CATEGORIES = ["true_negative_near_dup", "true_negative_unrelated"]
ALL_CATEGORIES = CONTRADICTION_CATEGORIES + TRUE_NEGATIVE_CATEGORIES

FLAG_LABELS = set(CONTRADICTION_CATEGORIES)


@dataclass(frozen=True)
class CategoryMetrics:
    category: str
    tp: int
    fp: int
    fn: int
    precision: float | None  # None when no predictions were made for the category
    recall: float | None
    f1: float | None


@dataclass
class SummaryMetrics:
    categories: list[CategoryMetrics] = field(default_factory=list)
    micro: dict[str, float] = field(default_factory=dict)
    macro_f1: float | None = None
    detection_recall: float | None = None  # flagged / labeled contradictions
    detection_precision: float | None = None  # flagged-contradiction / judged flags
    citation_faithfulness: float | None = None


def per_category_metrics(
    predictions: list[tuple[str | None, str | None]],
    categories: list[str] | None = None,
) -> list[CategoryMetrics]:
    """Compute P/R/F1 per category over (truth, predicted) pairs.

    `predicted` None = pair was never judged. Contradiction categories
    count judge-label hits (tp), wrong flags (fp), and missed
    contradictions (fn). True-negative categories use flag-avoidance
    semantics: they can never be "predicted", so a labeled TN that was not
    flagged is a tp, one that WAS flagged is a fn, and fp is always 0.
    """
    categories = categories or ALL_CATEGORIES
    counts = {c: [0, 0, 0] for c in categories}  # tp, fp, fn
    for truth, predicted in predictions:
        for category in categories:
            if category in TRUE_NEGATIVE_CATEGORIES:
                if truth == category:
                    if predicted in FLAG_LABELS:
                        counts[category][2] += 1  # flagged a harmless pair
                    else:
                        counts[category][0] += 1
            elif predicted == category:
                if truth == category:
                    counts[category][0] += 1
                else:
                    counts[category][1] += 1
            elif truth == category:
                counts[category][2] += 1
    results = []
    for category in categories:
        tp, fp, fn = counts[category]
        precision = tp / (tp + fp) if tp + fp else None
        recall = tp / (tp + fn) if tp + fn else None
        f1 = 2 * precision * recall / (precision + recall) if precision is not None and recall and precision + recall > 0 else None
        results.append(CategoryMetrics(category, tp, fp, fn, precision, recall, f1))
    return results


def micro_metrics(results: list[CategoryMetrics]) -> dict[str, float]:
    """Aggregate counts across categories, then recompute P/R/F1."""
    tp = sum(r.tp for r in results)
    fp = sum(r.fp for r in results)
    fn = sum(r.fn for r in results)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def macro_f1(results: list[CategoryMetrics]) -> float | None:
    """Mean F1 over categories that have labeled samples (None excluded)."""
    scored = [r.f1 for r in results if r.f1 is not None]
    if not scored:
        return None
    return sum(scored) / len(scored)


def detection_rates(
    predictions: list[tuple[str | None, str | None]],
) -> tuple[float | None, float | None]:
    """(detection_recall, detection_precision) over the pair set.

    detection_recall: fraction of labeled contradictions that were flagged
    (any contradiction label). detection_precision: of the flagged pairs,
    the fraction whose truth is a contradiction.
    """
    labeled_contradictions = sum(1 for truth, _ in predictions if truth in FLAG_LABELS)
    flagged = sum(1 for _, predicted in predictions if predicted in FLAG_LABELS)
    flagged_and_contradiction = sum(
        1 for truth, predicted in predictions
        if predicted in FLAG_LABELS and truth in FLAG_LABELS
    )
    recall = flagged_and_contradiction / labeled_contradictions if labeled_contradictions else None
    precision = flagged_and_contradiction / flagged if flagged else None
    return recall, precision


def citation_faithfulness(overall_passes: list[int | bool]) -> float | None:
    """Fraction of verified judge outputs with overall_pass = 1."""
    if not overall_passes:
        return None
    return sum(1 for p in overall_passes if p) / len(overall_passes)