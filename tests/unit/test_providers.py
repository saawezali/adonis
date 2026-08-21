"""get_client tier/provider resolution incl. custom (OpenAI-compatible) endpoints."""

from __future__ import annotations

import pytest


@pytest.fixture()
def env(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ADONIS_DB_PATH", str(tmp_path / "db.sqlite"))
    monkeypatch.setenv("ADONIS_REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("ADONIS_LLM_API_KEY", "sk-test")
    from adonis import config as cfg

    cfg._settings = None
    yield
    cfg._settings = None


def test_default_provider_is_anthropic(env):
    from adonis.config import provider_for_tier, reload_settings
    from adonis.llm.anthropic import AnthropicClient
    from adonis.llm.client import get_client

    settings = reload_settings()
    assert provider_for_tier(settings, "extractor") == "anthropic"
    assert provider_for_tier(settings, "judge") == "anthropic"
    client = get_client("judge")
    assert isinstance(client, AnthropicClient)
    assert client.model == "claude-3-5-sonnet-latest"


def test_tier_override_selects_openai_compatible(env, monkeypatch):
    monkeypatch.setenv("ADONIS_LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ADONIS_EXTRACTOR_PROVIDER", "openai")
    monkeypatch.setenv("ADONIS_LLM_BASE_URL", "https://api.openai.com/v1")
    from adonis import config as cfg
    from adonis.llm.client import get_client
    from adonis.llm.openai import OpenAIClient

    cfg._settings = None
    client = get_client("extractor")
    assert isinstance(client, OpenAIClient)
    assert client.base_url == "https://api.openai.com/v1"


def test_custom_provider_requires_base_url(env, monkeypatch):
    monkeypatch.setenv("ADONIS_LLM_PROVIDER", "custom")
    monkeypatch.delenv("ADONIS_LLM_BASE_URL", raising=False)
    from adonis import config as cfg
    from adonis.llm.client import get_client

    cfg._settings = None
    with pytest.raises(RuntimeError, match="ADONIS_LLM_BASE_URL"):
        get_client("judge")


def test_custom_provider_allows_empty_key(env, monkeypatch):
    monkeypatch.setenv("ADONIS_LLM_PROVIDER", "custom")
    monkeypatch.setenv("ADONIS_LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("ADONIS_LLM_API_KEY", "")
    from adonis import config as cfg
    from adonis.llm.client import get_client
    from adonis.llm.openai import OpenAIClient

    cfg._settings = None
    client = get_client("judge")  # local inference: no key required
    assert isinstance(client, OpenAIClient)
    assert client.base_url == "http://localhost:11434/v1"


def test_missing_key_still_raises_for_cloud(env, monkeypatch):
    monkeypatch.setenv("ADONIS_LLM_PROVIDER", "openai")
    monkeypatch.setenv("ADONIS_LLM_API_KEY", "")
    from adonis import config as cfg
    from adonis.llm.client import get_client

    cfg._settings = None
    with pytest.raises(RuntimeError, match="ADONIS_LLM_API_KEY"):
        get_client("extractor")


def test_unknown_tier_rejected(env):
    from adonis.config import reload_settings
    from adonis.llm.client import get_client

    reload_settings()
    with pytest.raises(ValueError, match="unknown tier"):
        get_client("pipeline")