"""Deterministic offline judge exercising the §1.4 decision rules.

Used by scripts/run_pipeline.py when run without --llm, and by
scripts/eval_trick_set.py as the reference implementation. It parses the
judge prompt back into claim views and applies the decision order:
same text -> not_conflicting; differing as_of dates ->
superseded_by_time; scope mismatch -> different_scope; otherwise
differing texts -> genuine_contradiction. Cites each full claim text.
"""

from __future__ import annotations

import json
import re

from adonis.llm.client import LLMClient

_FIELD_RE = re.compile(
    r"<claim_[ab]>\n"
    r"text: (?P<text>.*?)\n"
    r"temporal: (?P<temporal>\S[^\n]*)\n"
    r"scope: (?P<scope>\S[^\n]*)\n"
    r"</claim_[ab]>",
    re.DOTALL,
)


class DemoJudge(LLMClient):
    """An LLMClient that decides deterministically from the prompt fields."""

    model = "demo-judge"

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.0,
    ) -> dict[str, object]:
        del system
        matches = _FIELD_RE.findall(user)
        if len(matches) != 2:
            return {"label": "ambiguous", "confidence": 0.5, "reasoning": "parse error"}
        (text_a, temporal_a, scope_a), (text_b, temporal_b, scope_b) = matches
        as_of_a = _as_of(temporal_a)
        as_of_b = _as_of(temporal_b)
        text_a = text_a.strip()
        text_b = text_b.strip()
        if text_a == text_b:
            label, conf = "not_conflicting", 0.95
        elif as_of_a is not None and as_of_b is not None and as_of_a != as_of_b:
            label, conf = "superseded_by_time", 0.9
        elif scope_a != scope_b:
            label, conf = "different_scope", 0.9
        else:
            label, conf = "genuine_contradiction", 0.8
        return {
            "label": label,
            "confidence": conf,
            "reasoning": f"demo judge: {label}",
            "cited_span_a_start": 0,
            "cited_span_a_end": len(text_a),
            "cited_span_b_start": 0,
            "cited_span_b_end": len(text_b),
        }


def _as_of(raw: str) -> str | None:
    """Extract the as_of date from serialized temporal (or unspecified)."""
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(value, dict):
        return None
    as_of = value.get("as_of")
    return str(as_of) if isinstance(as_of, str) else None