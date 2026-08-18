"""Judge: prompt construction, response validation, demo roundtrip."""

from __future__ import annotations

from adonis.judge.classify import (
    ClaimView,
    build_judge_prompt,
    judge_pair,
    parse_judge_response,
    prompt_version,
)
from tests.fakes import FakeLLMClient

CLAIM_A = ClaimView(id="a", text="Atlas ships in March.", temporal={"as_of": "2026-03-01"}, scope=None)
CLAIM_B = ClaimView(id="b", text="Atlas ships in July.", temporal={"as_of": "2026-07-01"}, scope=None)


def _ok_response(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "label": "genuine_contradiction",
        "confidence": 0.85,
        "reasoning": "same scope, different dates",
        "cited_span_a_start": 0,
        "cited_span_a_end": 20,
        "cited_span_b_start": 0,
        "cited_span_b_end": 19,
    }
    base.update(overrides)
    return base


def test_prompt_version():
    assert prompt_version() == "judge_v1"


def test_build_judge_prompt_contains_both_claims_and_metadata():
    system, user = build_judge_prompt(CLAIM_A, CLAIM_B)
    assert "judge" in system
    assert "Atlas ships in March." in user
    assert "Atlas ships in July." in user
    assert "2026-03-01" in user  # temporal flows through
    assert "<claim_a>" in user and "<claim_b>" in user


def test_parse_judge_response_valid():
    result = parse_judge_response(_ok_response(), CLAIM_A.text, CLAIM_B.text)
    assert result is not None
    assert result.label == "genuine_contradiction"
    assert result.confidence == 0.85
    assert result.span_a_end == 20


def test_parse_rejects_unknown_label():
    assert (
        parse_judge_response(_ok_response(label="explodes"), CLAIM_A.text, CLAIM_B.text)
        is None
    )


def test_parse_rejects_missing_confidence_and_reasoning_defaults():
    result = parse_judge_response(
        _ok_response(confidence=None, reasoning=None), CLAIM_A.text, CLAIM_B.text
    )
    assert result is not None
    assert result.confidence == 0.0
    assert result.reasoning == ""


def test_parse_rejects_out_of_bounds_spans():
    assert (
        parse_judge_response(
            _ok_response(cited_span_a_end=999), CLAIM_A.text, CLAIM_B.text
        )
        is None
    )
    assert (
        parse_judge_response(
            _ok_response(cited_span_a_start=5, cited_span_a_end=5),
            CLAIM_A.text,
            CLAIM_B.text,
        )
        is None
    )


def test_parse_rejects_blank_span():
    assert (
        parse_judge_response(
            _ok_response(cited_span_a_start=5, cited_span_a_end=6),
            CLAIM_A.text,
            CLAIM_B.text,
        )
        is None
    )


def test_judge_pair_with_fake_client():
    client = FakeLLMClient(by_text={CLAIM_A.text: _ok_response()})
    result, error = judge_pair(client, CLAIM_A, CLAIM_B)
    assert error is None
    assert result is not None
    assert result.label == "genuine_contradiction"


def test_judge_pair_reports_llm_failure():
    class Boom(FakeLLMClient):
        def complete_json(self, system: str, user: str, **kwargs: object) -> dict[str, object]:
            raise RuntimeError("api down")

    result, error = judge_pair(Boom(), CLAIM_A, CLAIM_B)
    assert result is None
    assert "api down" in (error or "")


def test_demo_judge_roundtrip_via_pipeline_judge():
    from adonis.judge.demo import DemoJudge

    system, user = build_judge_prompt(CLAIM_A, CLAIM_B)
    response = DemoJudge().complete_json(system, user)
    assert isinstance(response, dict)
    parsed = parse_judge_response(response, CLAIM_A.text, CLAIM_B.text)
    assert parsed is not None
    assert parsed.label == "superseded_by_time"  # as_of 2026-03-01 vs 2026-07-01


def test_demo_judge_superseded_only_when_as_of_present():
    from adonis.judge.demo import DemoJudge

    plain_a = ClaimView(id="a", text="Atlas ships in March.", temporal=None, scope=None)
    plain_b = ClaimView(id="b", text="Atlas ships in July.", temporal=None, scope=None)
    system, user = build_judge_prompt(plain_a, plain_b)
    response = DemoJudge().complete_json(system, user)
    assert response["label"] == "genuine_contradiction"


def test_demo_judge_not_conflicting_on_same_text():
    from adonis.judge.demo import DemoJudge

    same = ClaimView(id="x", text="Atlas ships in March.", temporal=None, scope=None)
    system, user = build_judge_prompt(same, same)
    response = DemoJudge().complete_json(system, user)
    assert response["label"] == "not_conflicting"


def test_demo_judge_different_scope():
    from adonis.judge.demo import DemoJudge

    eu = ClaimView(id="a", text="Uptime is 99 percent.", temporal=None, scope={"region": "EU"})
    us = ClaimView(id="b", text="Uptime is 95 percent.", temporal=None, scope={"region": "US"})
    system, user = build_judge_prompt(eu, us)
    response = DemoJudge().complete_json(system, user)
    assert response["label"] == "different_scope"
    # spans cited are valid within the claim texts
    assert parse_judge_response(response, eu.text, us.text) is not None


def test_judge_pair_passes_json_through():
    client = FakeLLMClient(
        by_text={
            CLAIM_A.text: {
                "label": "superseded_by_time",
                "confidence": 0.9,
                "reasoning": "later date updates value",
                "cited_span_a_start": 0,
                "cited_span_a_end": 20,
                "cited_span_b_start": 0,
                "cited_span_b_end": 19,
            }
        }
    )
    result, error = judge_pair(client, CLAIM_A, CLAIM_B)
    assert error is None
    assert result is not None
    assert result.label == "superseded_by_time"


def test_confidence_clamped():
    result = parse_judge_response(
        _ok_response(confidence=2.0), CLAIM_A.text, CLAIM_B.text
    )
    assert result is not None
    assert result.confidence == 1.0