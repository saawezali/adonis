"""Citation span verification, lexical part.

Checks that the judge's cited span text actually matches the claim it is
supposed to support: a normalized verbatim comparison plus a rapidfuzz
ratio for whitespace/punctuation drift. Used by the verification pass;
the entailment check (verify/entailment.py) covers semantic support.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rapidfuzz import fuzz

from adonis.config import get_settings

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")


@dataclass(frozen=True)
class SpanMatchResult:
    verbatim: bool
    fuzzy_ratio: float  # 0..100, rapidfuzz ratio of normalized texts


def normalize_span(text: str) -> str:
    """Lowercase, collapse whitespace, drop punctuation."""
    return _PUNCT_RE.sub("", _WS_RE.sub(" ", text)).strip().lower()


def span_match(claim_text: str, span_text: str) -> SpanMatchResult:
    """Compare a cited span against the claim it must support.

    Empty span or claim text is a guaranteed mismatch (verbatim=False,
    fuzzy=0).
    """
    normalized_claim = normalize_span(claim_text)
    normalized_span = normalize_span(span_text)
    if not normalized_claim or not normalized_span:
        return SpanMatchResult(verbatim=False, fuzzy_ratio=0.0)
    ratio = float(fuzz.ratio(normalized_claim, normalized_span))
    verbatim = normalized_claim == normalized_span
    return SpanMatchResult(verbatim=verbatim, fuzzy_ratio=ratio)


def span_pass(result: SpanMatchResult, *, min_ratio: float | None = None) -> bool:
    """A span passes lexical verification when it matches the claim verbatim
    or within the configured fuzzy threshold. Empty spans never pass."""
    threshold = (
        min_ratio
        if min_ratio is not None
        else get_settings().span_fuzzy_threshold / 100.0
    )
    return result.verbatim or result.fuzzy_ratio >= threshold * 100.0