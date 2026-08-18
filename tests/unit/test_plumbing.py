"""Plumbing smoke test — no network, no LLM calls.

Covers M1 fundamentals that exist at this commit: config loading, schema
migration application, and idempotent re-application. Ingestion itself is
the rest of M1 and is not implemented yet.
"""

from __future__ import annotations

import pytest


@pytest.fixture()
def isolated_db(tmp_path, monkeypatch):
    """Point Adonis at a temp SQLite file and apply migrations."""
    from adonis import config as cfg

    monkeypatch.setenv("ADONIS_DB_PATH", str(tmp_path / "test.sqlite"))
    monkeypatch.setenv("ADONIS_CORPUS_DIR", str(tmp_path / "corpus"))
    monkeypatch.setenv("ADONIS_REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("ADONIS_LLM_API_KEY", "test-key")
    # Drop cached settings so env changes take effect.
    cfg._settings = None
    yield tmp_path
    cfg._settings = None


def test_migrations_create_tables(isolated_db):
    from adonis.db import apply_migrations, get_conn

    apply_migrations()
    apply_migrations()  # idempotent re-run must not raise

    expected = {
        "documents",
        "entities",
        "claims",
        "entity_mentions",
        "candidate_pairs",
        "judge_outputs",
        "verification_results",
        "flags",
        "eval_labels",
        "llm_calls",
        "schema_migrations",
    }
    conn = get_conn()
    try:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        names = {r["name"] for r in rows}
    finally:
        conn.close()
    assert expected <= names, f"missing tables: {expected - names}"


def test_settings_reads_env(isolated_db, monkeypatch):
    from adonis.config import get_settings

    monkeypatch.setenv("ADONIS_TOP_K", "7")
    monkeypatch.setenv("ADONIS_LLM_PROVIDER", "openai")
    from adonis import config as cfg
    cfg._settings = None

    s = get_settings()
    assert s.top_k == 7
    assert s.llm_provider == "openai"
    assert s.extractor_model.startswith("claude-")  # default still present


def test_llm_client_factory_requires_key(tmp_path, monkeypatch):
    """Without a key the factory must refuse rather than silently fail later."""
    from adonis import config as cfg
    cfg._settings = None
    monkeypatch.setenv("ADONIS_LLM_API_KEY", "")
    monkeypatch.setenv("ADONIS_DB_PATH", str(tmp_path / "x.sqlite"))

    from adonis.llm.client import get_client

    with pytest.raises(RuntimeError, match="ADONIS_LLM_API_KEY"):
        get_client("judge")
