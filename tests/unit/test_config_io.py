"""Config persistence (.env) + per-tier provider resolution + masking."""

from __future__ import annotations

from pathlib import Path

import pytest

from adonis import config as cfg


@pytest.fixture()
def clean_settings(monkeypatch, tmp_path):
    """Ensure .env lives in tmp_path and settings start fresh."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ADONIS_DB_PATH", str(tmp_path / "db.sqlite"))
    monkeypatch.setenv("ADONIS_REPORTS_DIR", str(tmp_path / "reports"))
    cfg._settings = None
    yield tmp_path
    cfg._settings = None


def test_save_settings_creates_and_updates_env(clean_settings):
    cfg.save_settings({"ADONIS_LLM_PROVIDER": "openai", "ADONIS_LLM_API_KEY": "sk-test-1234"})
    env_path = Path(".env")
    assert env_path.exists()
    text = env_path.read_text(encoding="utf-8")
    assert "ADONIS_LLM_PROVIDER=openai" in text
    assert "ADONIS_LLM_API_KEY=sk-test-1234" in text

    # Updating an existing key replaces in place; comments are preserved.
    env_path.write_text(
        "# header comment\nADONIS_LLM_PROVIDER=openai\nADONIS_LLM_API_KEY=old-key\n",
        encoding="utf-8",
    )
    cfg.save_settings({"ADONIS_LLM_PROVIDER": "custom"})
    out = env_path.read_text(encoding="utf-8").splitlines()
    assert out[0] == "# header comment"
    assert out[1] == "ADONIS_LLM_PROVIDER=custom"


def test_save_settings_ignores_blank_values(clean_settings):
    cfg.save_settings({"ADONIS_JUDGE_MODEL": "gpt-4o"})
    # Blank values must not clobber existing ones.
    cfg.save_settings({"ADONIS_JUDGE_MODEL": ""})
    text = Path(".env").read_text(encoding="utf-8")
    assert "ADONIS_JUDGE_MODEL=gpt-4o" in text
    assert "ADONIS_JUDGE_MODEL=" not in text.replace("ADONIS_JUDGE_MODEL=gpt-4o", "")


def test_reload_picks_up_saved_env(clean_settings):
    Path(".env").write_text(
        "ADONIS_LLM_PROVIDER=custom\nADONIS_LLM_BASE_URL=http://localhost:11434/v1\n",
        encoding="utf-8",
    )
    s = cfg.reload_settings()
    assert s.llm_provider == "custom"
    assert s.llm_base_url == "http://localhost:11434/v1"


def test_provider_for_tier(clean_settings):
    cfg.save_settings({
        "ADONIS_LLM_PROVIDER": "anthropic",
        "ADONIS_EXTRACTOR_PROVIDER": "custom",
    })
    s = cfg.reload_settings()
    assert cfg.provider_for_tier(s, "extractor") == "custom"
    assert cfg.provider_for_tier(s, "judge") == "anthropic"  # falls back to global
    with pytest.raises(ValueError):
        cfg.provider_for_tier(s, "bogus")


def test_mask_secret():
    assert cfg.mask_secret("") == ""
    assert cfg.mask_secret("abc") == "***"
    masked = cfg.mask_secret("sk-proj-ABCD")
    assert masked.endswith("ABCD")
    assert masked != "sk-proj-ABCD"


def test_env_file_is_restricted(clean_settings):
    cfg.save_settings({"ADONIS_LLM_API_KEY": "sk-test-1234"})
    mode = Path(".env").stat().st_mode & 0o777
    assert mode == 0o600