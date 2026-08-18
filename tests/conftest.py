"""Fixture: isolated settings + a scratch database per test."""

from __future__ import annotations

import pytest

from adonis import config as cfg


@pytest.fixture
def tmp_env(tmp_path, monkeypatch):
    """Point all ADONIS_ paths at a scratch dir and reset cached settings."""
    monkeypatch.setenv("ADONIS_DB_PATH", str(tmp_path / "db" / "adonis.sqlite"))
    monkeypatch.setenv("ADONIS_CORPUS_DIR", str(tmp_path / "corpus"))
    monkeypatch.setenv("ADONIS_REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("ADONIS_LLM_API_KEY", "test-key")
    cfg._settings = None
    yield tmp_path
    cfg._settings = None
