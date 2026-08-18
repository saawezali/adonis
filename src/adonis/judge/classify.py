"""LLM-as-judge (prompt judge_v1, PLAN.md M3).

Takes one candidate pair, applies the §1.4 decision rules, and returns a
label + raw confidence + reasoning + cited spans. Spans are returned
relative to each claim's own text; the pipeline converts them to absolute
document offsets and validates them (they must resolve to non-blank text).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from adonis.llm.client import LLMClient

_PROMPT_VERSION = "judge_v1"
_PROMPTS_DIR = Path(__file__).parent.parent / "llm" / "prompts"

VALID_LABELS = {
    "genuine_contradiction",
    "superseded_by_time",
    "different_scope",
    "ambiguous",
    "not_conflicting",
}


@dataclass(frozen=True)
class JudgeResult:
    label: str
    confidence: float
    reasoning: str
    span_a_start: int  # relative to claim A text
    span_a_end: int
    span_b_start: int
    span_b_end: int


@dataclass(frozen=True)
class ClaimView:
    """What the judge sees about one claim."""

    id: str
    text: str
    temporal: dict[str, str] | None
    scope: dict[str, str] | None


def prompt_text() -> str:
    return (_PROMPTS_DIR / f"{_PROMPT_VERSION}.txt").read_text(encoding="utf-8")


def prompt_version() -> str:
    return _PROMPT_VERSION


def build_judge_prompt(claim_a: ClaimView, claim_b: ClaimView) -> tuple[str, str]:
    system = prompt_text()
    user = (
        "<claim_a>\n"
        f"text: {claim_a.text}\n"
        f"temporal: {json.dumps(claim_a.temporal) if claim_a.temporal else 'unspecified'}\n"
        f"scope: {json.dumps(claim_a.scope) if claim_a.scope else 'unspecified'}\n"
        "</claim_a>\n"
        "<claim_b>\n"
        f"text: {claim_b.text}\n"
        f"temporal: {json.dumps(claim_b.temporal) if claim_b.temporal else 'unspecified'}\n"
        f"scope: {json.dumps(claim_b.scope) if claim_b.scope else 'unspecified'}\n"
        "</claim_b>"
    )
    return system, user


def _int_or_none(value: object) -> int | None:
    return int(value) if isinstance(value, (int, float)) else None


def parse_judge_response(
    response: dict[str, object], text_a: str, text_b: str
) -> JudgeResult | None:
    """Validate the judge's JSON against the two claim texts.

    Returns None when the response is structurally invalid (bad label,
    unparsable spans, or spans outside the claim texts). Callers count
    failures and keep the pair unjudged.
    """
    label = response.get("label")
    if not isinstance(label, str) or label not in VALID_LABELS:
        return None
    start_a = _int_or_none(response.get("cited_span_a_start"))
    end_a = _int_or_none(response.get("cited_span_a_end"))
    start_b = _int_or_none(response.get("cited_span_b_start"))
    end_b = _int_or_none(response.get("cited_span_b_end"))
    if start_a is None or end_a is None or start_b is None or end_b is None:
        return None
    if _span_invalid(start_a, end_a, text_a) or _span_invalid(start_b, end_b, text_b):
        return None
    confidence = response.get("confidence")
    conf = float(confidence) if isinstance(confidence, (int, float)) else 0.0
    reasoning = response.get("reasoning")
    reason = reasoning if isinstance(reasoning, str) else ""
    return JudgeResult(
        label=label,
        confidence=min(1.0, max(0.0, conf)),
        reasoning=reason,
        span_a_start=start_a,
        span_a_end=end_a,
        span_b_start=start_b,
        span_b_end=end_b,
    )


def _span_invalid(start: int, end: int, text: str) -> bool:
    return start < 0 or end <= start or end > len(text) or not text[start:end].strip()


def judge_pair(
    client: LLMClient, claim_a: ClaimView, claim_b: ClaimView
) -> tuple[JudgeResult | None, str | None]:
    """Judge one pair. Returns (result, error); exactly one is non-None."""
    system, user = build_judge_prompt(claim_a, claim_b)
    try:
        response = client.complete_json(system, user)
    except Exception as exc:  # noqa: BLE001
        return None, f"judge call failed: {exc!r}"
    result = parse_judge_response(response, claim_a.text, claim_b.text)
    if result is None:
        return None, f"unparseable judge response: {str(response)[:200]!r}"
    return result, None
