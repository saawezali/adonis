"""M4: citation verification (lexical + entailment)."""

from adonis.verify.entailment import EntailmentResult, verify_entailment
from adonis.verify.span_match import (
    SpanMatchResult,
    normalize_span,
    span_match,
    span_pass,
)

__all__ = [
    "EntailmentResult",
    "SpanMatchResult",
    "normalize_span",
    "span_match",
    "span_pass",
    "verify_entailment",
]