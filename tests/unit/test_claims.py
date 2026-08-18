"""Claim extraction: spans, triviality filter, declarative validation."""

from __future__ import annotations

from adonis.extract.claims import (
    build_claim_prompt,
    extract_document_claims,
    prompt_version,
)
from tests.fakes import FakeLLMClient

DOC = (
    "Atlas ships in March. It runs on Postgres and costs 500 euros a month.\n"
    "Is Atlas ready for launch? We should improve the onboarding. This page "
    "has a title. The budget is 500 EUR for travel.\n"
)


def _response_with(**claim_fields: object) -> dict[str, object]:
    claim: dict[str, object] = {
        "claim_text": "Atlas ships in March.",
        "span_start": 0,
        "span_end": 21,
        "triviality_score": 0.05,
        "topics": ["atlas", "release"],
        "temporal": {"as_of": "2026-03-01"},
        "scope": None,
    }
    claim.update(claim_fields)
    return {"claims": [claim]}


def test_prompt_version_and_build():
    assert prompt_version() == "claims_v1"
    system, user = build_claim_prompt("hello")
    assert "claim" in system
    assert "hello" in user


def test_extract_single_claim_with_absolute_offsets():
    client = FakeLLMClient(by_text={DOC: _response_with()})
    claims, stats = extract_document_claims(client, DOC)
    assert len(claims) == 1
    claim = claims[0]
    assert claim.claim_text == "Atlas ships in March."
    assert claim.span_start == 0
    assert claim.span_end == 21
    assert DOC[claim.span_start : claim.span_end] == "Atlas ships in March."
    assert claim.temporal == {"as_of": "2026-03-01"}
    assert claim.topics == ["atlas", "release"]
    assert stats.chunks == 1
    assert stats.llm_calls == 1


def test_span_shifted_when_claim_not_at_chunk_start():
    text = "Preface. Atlas ships in March.\n"
    start = text.find("Atlas")
    client = FakeLLMClient(
        by_text={
            text: {
                "claims": [
                    {
                        "claim_text": "Atlas ships in March.",
                        "span_start": 9,
                        "span_end": 30,
                        "triviality_score": 0.1,
                        "topics": ["atlas"],
                        "temporal": None,
                        "scope": None,
                    }
                ]
            }
        }
    )
    claims, _ = extract_document_claims(client, text)
    assert len(claims) == 1
    assert claims[0].span_start == start
    assert text[claims[0].span_start : claims[0].span_end] == "Atlas ships in March."


def test_triviality_filter_drops_low_value_claims():
    client = FakeLLMClient(
        by_text={
            DOC: _response_with(triviality_score=0.95),
        }
    )
    claims, stats = extract_document_claims(client, DOC)
    assert claims == []
    assert stats.trivial_dropped == 1


def test_triviality_filter_cutoff_override():
    client = FakeLLMClient(by_text={DOC: _response_with(triviality_score=0.95)})
    claims, _ = extract_document_claims(client, DOC, cutoff=0.99)
    assert len(claims) == 1


def test_invalid_span_dropped():
    client = FakeLLMClient(
        by_text={DOC: _response_with(span_start=1000, span_end=1010)}
    )
    claims, stats = extract_document_claims(client, DOC)
    assert claims == []
    assert stats.span_dropped == 1


def test_question_claim_dropped():
    client = FakeLLMClient(
        by_text={DOC: _response_with(claim_text="Is Atlas ready for launch?")}
    )
    claims, stats = extract_document_claims(client, DOC)
    assert claims == []
    assert stats.shape_dropped == 1


def test_empty_claims_response():
    client = FakeLLMClient(by_text={DOC: {"claims": []}})
    claims, stats = extract_document_claims(client, DOC)
    assert claims == []
    assert stats.claims_from_llm == 0


def test_llm_failure_reported_not_raised():
    class BoomClient(FakeLLMClient):
        def complete_json(self, system: str, user: str, **kwargs: object) -> dict[str, object]:
            raise RuntimeError("api down")

    claims, stats = extract_document_claims(BoomClient(), DOC)
    assert claims == []
    assert len(stats.errors) == 1


def test_multi_chunk_offsets():
    text = ("Sentence one. " * 50) + "Atlas ships in March. " + ("Tail here. " * 60)
    target = "Atlas ships in March."
    start = text.find(target)
    by_text: dict[str, dict[str, object]] = {}
    for chunk, _start, _end in _chunks_of(text):
        if target in chunk:
            rel = chunk.find(target)
            by_text[chunk] = {
                "claims": [
                    {
                        "claim_text": target,
                        "span_start": rel,
                        "span_end": rel + len(target),
                        "triviality_score": 0.1,
                        "topics": ["atlas"],
                        "temporal": None,
                        "scope": None,
                    }
                ]
            }
    client = FakeLLMClient(by_text=by_text)
    claims, stats = extract_document_claims(client, text, max_chars=200)
    assert len(claims) == 1
    assert claims[0].span_start == start
    assert text[claims[0].span_start : claims[0].span_end] == target
    assert stats.chunks > 1
    assert stats.llm_calls == stats.chunks


def _chunks_of(text: str, max_chars: int = 200):
    from adonis.extract.chunk import chunk_document

    return [(c.text, c.start, c.end) for c in chunk_document(text, max_chars)]
