"""Web console API: settings (masked), save-to-.env, probe, status."""

from __future__ import annotations

from pathlib import Path

import pytest

from adonis import config as cfg


@pytest.fixture()
def env(tmp_path, monkeypatch):
    """Isolated state: .env in tmp_path, scratch DB, cached settings reset."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ADONIS_DB_PATH", str(tmp_path / "db" / "adonis.sqlite"))
    monkeypatch.setenv("ADONIS_REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("ADONIS_LLM_API_KEY", "sk-secret-ABCD")
    cfg._settings = None
    yield
    cfg._settings = None


@pytest.fixture()
def client(env):
    from fastapi.testclient import TestClient

    from adonis.web.app import create_app

    return TestClient(create_app())


def test_settings_endpoint_masks_key(client):
    resp = client.get("/api/settings")
    assert resp.status_code == 200
    body = resp.json()
    assert body["key_set"] is True
    assert body["llm_api_key"] != "sk-secret-ABCD"
    assert body["llm_api_key"].endswith("ABCD")
    assert "custom" in body["providers"]  # custom inference is selectable


def test_settings_page_served(client):
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Adonis" in resp.text and "console" in resp.text.lower()
    assert "api/settings" in resp.text


def test_save_settings_writes_env_and_masks(client):
    resp = client.post(
        "/api/settings",
        json={
            "llm_provider": "custom",
            "llm_base_url": "http://localhost:11434/v1",
            "extractor_model": "llama3.1",
            "judge_model": "llama3.1",
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["llm_provider"] == "custom"
    assert body["env_file"].endswith(".env")
    text = Path(".env").read_text(encoding="utf-8")
    assert "ADONIS_LLM_PROVIDER=custom" in text
    assert "ADONIS_LLM_BASE_URL=http://localhost:11434/v1" in text
    # The saved key still uses the env-provided one; it is not echoed back.
    assert "sk-secret-ABCD" not in text


def test_save_settings_rejects_bad_provider(client):
    resp = client.post("/api/settings", json={"llm_provider": "telepathy"})
    assert resp.status_code == 400
    assert "not in" in resp.json()["detail"]


def test_probe_custom_without_url_returns_400(client):
    resp = client.post("/api/test", json={"llm_provider": "custom"})
    assert resp.status_code == 400
    assert "base URL" in resp.json()["detail"]


def test_probe_with_mocked_client(client, monkeypatch):
    """POST /api/test builds an adapter from the patch; mock the network part."""

    def fake_probe(*args, **kwargs):
        return {
            "ok": True, "latency_ms": 5, "provider": "openai", "model": "gpt-4o",
            "base_url": "https://api.openai.com/v1",
        }

    import importlib

    web_module = importlib.import_module("adonis.web.app")
    monkeypatch.setattr(web_module, "_probe", fake_probe)
    resp = client.post(
        "/api/test",
        json={"llm_provider": "openai", "judge_model": "gpt-4o",
              "llm_api_key": "sk-new"},
    )
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


def test_probe_surfaces_network_error(client, monkeypatch):
    import importlib

    from fastapi import HTTPException

    def raise_wrapper(*args, **kwargs):
        raise HTTPException(status_code=502, detail="boom")

    web_module = importlib.import_module("adonis.web.app")
    monkeypatch.setattr(web_module, "_probe", raise_wrapper)
    resp = client.post("/api/test", json={"llm_provider": "openai"})
    assert resp.status_code == 502


def test_status_returns_counts(client):
    resp = client.get("/api/status")
    assert resp.status_code == 200
    body = resp.json()
    for key in ("claims", "documents", "flags", "eval_labels"):
        assert key in body
    assert body["judge_effective"] == "anthropic"


def test_report_redirect(client):
    resp = client.get("/report", follow_redirects=False)
    assert resp.status_code == 307
    assert resp.headers["location"] == "/reports/index.html"