"""Zero-shot NER over claim spans with GLiNER (PLAN.md M2).

GLiNER is loaded lazily (first call downloads the model unless it is
cached); callers may inject any object exposing
`predict_entities(text, labels, threshold=...)` for offline tests.
"""

from __future__ import annotations

from dataclasses import dataclass

from adonis.config import get_settings


@dataclass(frozen=True)
class Mention:
    """A named-entity mention with offsets into the document raw_text."""

    text: str
    label: str
    start: int
    end: int
    score: float = 1.0


_model = None


def load_model(force: bool = False) -> object:
    """Load the configured GLiNER model once and cache it."""
    global _model
    if _model is None or force:
        from gliner import GLiNER

        _model = GLiNER.from_pretrained(get_settings().gliner_model)
    return _model


def _dedupe_overlaps(mentions: list[Mention]) -> list[Mention]:
    """Keep the highest-scoring mention among overlapping spans."""
    kept: list[Mention] = []
    for m in sorted(mentions, key=lambda m: (m.start, -m.score)):
        if any(m.start < k.end and k.start < m.end for k in kept):
            continue
        kept.append(m)
    return kept


def extract_mentions(
    span_text: str,
    offset: int = 0,
    *,
    model: object | None = None,
    labels: list[str] | None = None,
    threshold: float | None = None,
) -> list[Mention]:
    """Run NER on `span_text`; results shifted by `offset` into the document.

    `model` is injectable for tests; defaults to the cached GLiNER instance.
    """
    model = model if model is not None else load_model()
    labels = labels if labels is not None else get_settings().extract_labels
    threshold = threshold if threshold is not None else get_settings().gliner_threshold
    predictions = model.predict_entities(span_text, labels, threshold=threshold)  # type: ignore[attr-defined]
    mentions = [
        Mention(
            text=str(p["text"]),
            label=str(p["label"]),
            start=offset + int(p["start"]),
            end=offset + int(p["end"]),
            score=float(p.get("score", 1.0)),
        )
        for p in predictions
    ]
    return _dedupe_overlaps(mentions)
