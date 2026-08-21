"""LLM claim extraction (prompt claims_v1).

Per design spec M2: atomic claims in declarative form, with citation spans,
temporal/scope properties and a triviality score. Pipeline here:
chunk document -> per-chunk LLM call -> validate spans -> apply the
triviality filter. Trivial or invalid claims are dropped and counted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from adonis.config import get_settings
from adonis.extract.chunk import Chunk, chunk_document
from adonis.llm.client import LLMClient

_PROMPT_VERSION = "claims_v1"
_PROMPTS_DIR = Path(__file__).parent.parent / "llm" / "prompts"

_MAX_SPAN_CHARS = 600
_TRIVIAL_RE = re.compile(
    r"\b(page|document|note|file|section|heading|header)\b.*\b(title|heading|name)\b",
    re.IGNORECASE,
)


@dataclass
class ClaimRecord:
    """One extracted claim, with citation offsets into the document raw_text."""

    claim_text: str
    span_start: int
    span_end: int
    topics: list[str] = field(default_factory=list)
    temporal: dict[str, str] | None = None
    scope: dict[str, str] | None = None
    triviality_score: float = 0.5


@dataclass
class ExtractionStats:
    """Counters for one extraction run over one or more documents."""

    chunks: int = 0
    llm_calls: int = 0
    claims_from_llm: int = 0
    trivial_dropped: int = 0
    span_dropped: int = 0
    shape_dropped: int = 0
    errors: list[str] = field(default_factory=list)


def prompt_text() -> str:
    """Return the current claim-extraction system prompt."""
    p = _PROMPTS_DIR / f"{_PROMPT_VERSION}.txt"
    if not p.exists():
        raise FileNotFoundError(f"prompt not found: {p} (run from repo root or check install)")
    return p.read_text(encoding="utf-8")


def prompt_version() -> str:
    return _PROMPT_VERSION


def build_claim_prompt(chunk_text: str) -> tuple[str, str]:
    """Return (system, user) prompt pair for one chunk."""
    user = (
        "Analyze the following document text. Extract every checkable factual "
        "claim as described.\n\n<text>\n"
        f"{chunk_text}\n"
        "</text>"
    )
    return prompt_text(), user


def _normalize_for_match(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _span_is_valid(chunk: Chunk, raw: dict[str, object]) -> tuple[int, int] | None:
    """Validate LLM-provided span offsets against the chunk text.

    Returns (chunk-relative start, end) on success, None otherwise. A span is
    valid when both offsets are ints, the range is non-empty, within bounds,
    not absurdly long, and the covered slice is non-blank.
    """
    start = raw.get("span_start")
    end = raw.get("span_end")
    if not isinstance(start, int) or not isinstance(end, int):
        return None
    if end <= start:
        return None
    if end - start > _MAX_SPAN_CHARS:
        return None
    if start < 0 or end > len(chunk.text):
        return None
    slice_text = chunk.text[start:end]
    if not slice_text.strip():
        return None
    return start, end


def _is_declarative(claim_text: str) -> bool:
    text = claim_text.strip()
    if not text:
        return False
    if text.endswith(("?", "!")):
        return False
    # Strip leading quotes/brackets/digits before checking capitalization
    stripped = text.lstrip("\"'“”‘’([{")
    # Allow non-letter starts (digits, etc.)
    if stripped and stripped[0].isalpha():
        return not stripped[0].islower()
    return True


def _is_trivial(
    raw: dict[str, object], cutoff: float, claim_text: str, span_text: str
) -> bool:
    # triviality_score = 1 - informativeness: high means noise (per prompt v1).
    score = raw.get("triviality_score")
    if isinstance(score, (int, float)) and score >= cutoff:
        return True
    # Only drop extremely short claims; 10 chars is safe lower bound
    if len(claim_text.strip()) < 10:
        return True
    return bool(_TRIVIAL_RE.search(span_text))


def _claims_from_llm_response(
    response: dict[str, object], chunk: Chunk, cutoff: float, stats: ExtractionStats
) -> list[ClaimRecord]:
    claims_raw = response.get("claims")
    if not isinstance(claims_raw, list):
        stats.errors.append(f"chunk {chunk.start}: 'claims' missing or not a list")
        return []
    claims: list[ClaimRecord] = []
    for item in claims_raw:
        if not isinstance(item, dict):
            stats.shape_dropped += 1
            continue
        claim_text = item.get("claim_text")
        if not isinstance(claim_text, str):
            stats.shape_dropped += 1
            continue
        if not _is_declarative(claim_text):
            stats.shape_dropped += 1
            continue
        span = _span_is_valid(chunk, item)
        if span is None:
            stats.span_dropped += 1
            continue
        span_start, span_end = span
        if _is_trivial(item, cutoff, claim_text, chunk.text[span_start:span_end]):
            stats.trivial_dropped += 1
            continue
        topics = item.get("topics")
        topics_list = (
            [t for t in topics if isinstance(t, str) and t.strip()]
            if isinstance(topics, list)
            else []
        )
        temporal = item.get("temporal")
        scope = item.get("scope")
        claims.append(
            ClaimRecord(
                claim_text=claim_text.strip(),
                span_start=chunk.start + span_start,
                span_end=chunk.start + span_end,
                topics=topics_list[:3],
                temporal=(
                    {str(k): str(v) for k, v in temporal.items()}
                    if isinstance(temporal, dict) and temporal
                    else None
                ),
                scope=(
                    {str(k): str(v) for k, v in scope.items()}
                    if isinstance(scope, dict) and scope
                    else None
                ),
                triviality_score=(
                    float(item["triviality_score"])
                    if isinstance(item.get("triviality_score"), (int, float))
                    else 0.0
                ),
            )
        )
    return claims


def extract_claims_from_chunk(
    client: LLMClient, chunk: Chunk, *, cutoff: float | None = None, max_tokens: int = 2048
) -> tuple[list[ClaimRecord], ExtractionStats]:
    """Run one extraction call on a chunk. Stats are fresh for this chunk."""
    cutoff = cutoff if cutoff is not None else get_settings().triviality_cutoff
    stats = ExtractionStats(chunks=1, llm_calls=1)
    system, user = build_claim_prompt(chunk.text)
    try:
        response = client.complete_json(system, user, max_tokens=max_tokens)
    except Exception as exc:
        stats.errors.append(f"chunk {chunk.start}: LLM call failed: {exc!r}")
        return [], stats
    claims = _claims_from_llm_response(response, chunk, cutoff, stats)
    stats.claims_from_llm = len(claims)
    return claims, stats


def extract_document_claims(
    client: LLMClient,
    raw_text: str,
    *,
    max_chars: int | None = None,
    cutoff: float | None = None,
) -> tuple[list[ClaimRecord], ExtractionStats]:
    """Extract claims from a whole document, merging chunk-level results.

    Citation spans are validated against the chunk text and shifted to
    document-absolute offsets; trivial and non-declarative claims are dropped.
    """
    chunks = chunk_document(raw_text, max_chars=max_chars)
    stats = ExtractionStats(chunks=len(chunks), llm_calls=0)
    claims: list[ClaimRecord] = []
    seen: set[str] = set()
    for chunk in chunks:
        chunk_claims, chunk_stats = extract_claims_from_chunk(
            client, chunk, cutoff=cutoff
        )
        stats.llm_calls += chunk_stats.llm_calls
        stats.trivial_dropped += chunk_stats.trivial_dropped
        stats.span_dropped += chunk_stats.span_dropped
        stats.shape_dropped += chunk_stats.shape_dropped
        stats.errors.extend(chunk_stats.errors)
        for cl in chunk_claims:
            # Intra-doc dedup on normalized claim_text
            key = cl.claim_text.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            claims.append(cl)
    stats.claims_from_llm = len(claims)
    return claims, stats


def span_text(document_raw: str, claim: ClaimRecord) -> str:
    """The exact document substring cited by a claim (for entity extraction)."""
    return document_raw[claim.span_start : claim.span_end]
