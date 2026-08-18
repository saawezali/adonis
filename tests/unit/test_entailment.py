"""Entailment verification: prompt, response validation, demo roundtrip."""

from __future__ import annotations

from adonis.verify.demo import DemoVerifier
from adonis.verify.entailment import (
    build_entailment_prompt,
    parse_entailment_response,
    prompt_version,
    verify_entailment,
)
from tests.fakes import FakeLLMClient

SPAN = "Atlas ships in March."
CLAIM = "Atlas ships in March."


def _ok_response(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "entailed": True,
        "score": 0.95,
        "reason": "the span states the claim directly",
    }
    base.update(overrides)
    return base


def test_prompt_version():
    assert prompt_version() == "entail_v1"


def test_build_prompt_contains_span_and_claim():
    system, user = build_entailment_prompt(SPAN, CLAIM)
    assert "cited span" in system
    assert SPAN in user and CLAIM in user


def test_parse_valid():
    result = parse_entailment_response(_ok_response())
    assert result is not None
    assert result.pass_
    assert result.score == 0.95


def test_parse_rejects_wrong_shapes():
    assert parse_entailment_response({"entailed": True}) is None
    assert parse_entailment_response({"score": 0.5, "entailed": "yes"}) is None


def test_parse_low_score_fails_despite_entailed():
    result = parse_entailment_response(_ok_response(score=0.5), min_confidence=0.8)
    assert result is not None
    assert result.score == 0.5
    assert not result.pass_


def test_parse_clamps_score():
    result = parse_entailment_response(_ok_response(score=3.0))
    assert result is not None
    assert result.score == 1.0


def test_verify_entailment_with_fake_client():
    client = FakeLLMClient(by_text={SPAN: _ok_response()})
    result, error = verify_entailment(client, SPAN, CLAIM)
    assert error is None
    assert result is not None
    assert result.pass_


def test_verify_reports_llm_failure():
    class Boom(FakeLLMClient):
        def complete_json(self, system: str, user: str, **kwargs: object) -> dict[str, object]:
            raise RuntimeError("network down")

    result, error = verify_entailment(Boom(), SPAN, CLAIM)
    assert result is None
    assert "network down" in (error or "")


def test_demo_verifier_passes_verbatim():
    system, user = build_entailment_prompt(SPAN, CLAIM)
    response = DemoVerifier().complete_json(system, user)
    assert response["entailed"] is True
    assert response["score"] == 1.0
    assert parse_entailment_response(response) is not None


def test_demo_verifier_fails_unrelated_span():
    system, user = build_entailment_prompt("The budget is 800 euros.", "Atlas ships in March.")
    response = DemoVerifier().complete_json(system, user)
    assert response["entailed"] is False
    assert response["score"] < 0.5


def test_demo_verifier_parse_error_is_not_entailed():
    response = DemoVerifier().complete_json("system", "garbage prompt")
    assert response["entailed"] is False