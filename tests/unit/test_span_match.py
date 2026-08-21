"""Lexical citation verification: verbatim + fuzzy span matching."""

from __future__ import annotations

from adonis.verify.span_match import normalize_span, span_match, span_pass


def test_normalize_span_lowercases_and_strips():
    assert normalize_span("  Atlas   Ships, in March! \n") == "atlas ships in march"


def test_verbatim_match():
    result = span_match("Atlas ships in March.", "Atlas ships in March.")
    assert result.verbatim
    assert result.fuzzy_ratio == 100.0
    assert span_pass(result)


def test_whitespace_and_case_drift_still_verbatim():
    result = span_match("Atlas ships in March.", "atlas ships in march.  ")
    assert result.verbatim
    assert span_pass(result)


def test_slight_drift_passes_fuzzy():
    result = span_match("Atlas ships in March.", "Atlas ships in March next year.")
    assert not result.verbatim
    assert result.fuzzy_ratio > 70.0
    assert span_pass(result, min_ratio=0.7)


def test_unrelated_span_fails():
    result = span_match("Atlas ships in March.", "We run Postgres in production.")
    assert not result.verbatim
    assert result.fuzzy_ratio < 50.0
    assert not span_pass(result)


def test_empty_span_never_passes():
    result = span_match("Atlas ships in March.", "   ")
    assert result.verbatim is False
    assert result.fuzzy_ratio == 0.0
    assert not span_pass(result)


def test_different_date_fails_threshold():
    result = span_match("Atlas ships in March.", "Atlas ships in July.")
    assert not result.verbatim
    assert result.fuzzy_ratio < 90.0
    assert not span_pass(result, min_ratio=0.9)