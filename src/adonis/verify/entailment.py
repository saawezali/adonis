"""Citation span entailment verification via LLM (PLAN.md M4).

For each judge-cited span, re-ask the model whether the span text actually
supports the claim it was cited for. Returns an entailment score in 0..1
plus a boolean pass; pass requires score >= ADONIS_ENTAIL_MIN_CONFIDENCE.
The span/cite strings are included verbatim so the model can detect spans
that merely sound related but do not entail the claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from adonis.config import get_settings
from adonis.llm.client import LLMClient

_PROMPT_VERSION = "entail_v1"
_PROMPTS_DIR = Path(__file__).parent.parent / "llm" / "prompts"


@dataclass(frozen=True)
class EntailmentResult:
    score: float  # 0..1
    pass_: bool
    reason: str


def prompt_version() -> str:
    return _PROMPT_VERSION


def prompt_text() -> str:
    p = _PROMPTS_DIR / f"{_PROMPT_VERSION}.txt"
    if not p.exists():
        raise FileNotFoundError(f"prompt not found: {p}")
    return p.read_text(encoding="utf-8")


def build_entailment_prompt(span_text: str, claim_text: str) -> tuple[str, str]:
    system = prompt_text()
    user = (
        f"<cited_span>\n{span_text}\n</cited_span>\n"
        f"<claim>\n{claim_text}\n</claim>"
    )
    return system, user


def parse_entailment_response(
    response: dict[str, object], *, min_confidence: float | None = None
) -> EntailmentResult | None:
    """Validate the verifier's JSON. None when structurally invalid."""
    score_raw = response.get("score")
    entailed = response.get("entailed")
    if not isinstance(score_raw, (int, float)) or not isinstance(entailed, bool):
        return None
    reason = response.get("reason")
    score = min(1.0, max(0.0, float(score_raw)))
    threshold = (
        min_confidence
        if min_confidence is not None
        else get_settings().entail_min_confidence
    )
    effective = entailed and score >= threshold
    return EntailmentResult(
        score=score,
        pass_=effective,
        reason=reason if isinstance(reason, str) else "",
    )


def verify_entailment(
    client: LLMClient, span_text: str, claim_text: str
) -> tuple[EntailmentResult | None, str | None]:
    """Verify one cited span against its claim. Returns (result, error)."""
    system, user = build_entailment_prompt(span_text, claim_text)
    try:
        response = client.complete_json(system, user)
    except Exception as exc:  # noqa: BLE001
        return None, f"entailment call failed: {exc!r}"
    result = parse_entailment_response(response)
    if result is None:
        return None, f"unparseable entailment response: {str(response)[:200]!r}"
    return result, None


def entailment_pass(result: EntailmentResult) -> bool:
    return result.pass_