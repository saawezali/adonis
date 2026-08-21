"""Typed application configuration, loaded from environment / .env.

provider-independent. Concrete adapter is selected per
tier: a provider-specific override (ADONIS_EXTRACTOR_PROVIDER /
ADONIS_JUDGE_PROVIDER) wins, otherwise ADONIS_LLM_PROVIDER is used. Tiers are
configured independently (extractor: cheap/fast; judge: larger/smarter).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["anthropic", "openai", "custom"]
ALL_PROVIDERS: tuple[Literal["anthropic", "openai", "custom"], ...] = (
    "anthropic",
    "openai",
    "custom",
)

#: Built-in defaults per provider for the web UI (edit via settings page).
PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "anthropic": {"base_url": "", "extractor_model": "claude-3-5-haiku-latest", "judge_model": "claude-3-5-sonnet-latest"},
    "openai": {"base_url": "https://api.openai.com/v1", "extractor_model": "gpt-4o-mini", "judge_model": "gpt-4o"},
    "custom": {"base_url": "http://localhost:11434/v1", "extractor_model": "llama3.1", "judge_model": "llama3.1"},
}


class Settings(BaseSettings):
    """Configuration. Reads from environment and .env."""

    model_config = SettingsConfigDict(
        env_prefix="ADONIS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    llm_provider: Provider = "anthropic"
    # Tier overrides (None = fall back to llm_provider for that tier).
    extractor_provider: Provider | None = None
    judge_provider: Provider | None = None
    llm_api_key: str = ""
    llm_base_url: str | None = None
    extractor_model: str = "claude-3-5-haiku-latest"
    judge_model: str = "claude-3-5-sonnet-latest"

    # Embeddings / candidate generation
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    top_k: int = 20
    triviality_cutoff: float = 0.5  # raised from 0.3 after audit A4
    span_fuzzy_threshold: int = 90
    # Canonical weight for hybrid scoring. Legacy aliases below are deprecated but kept for compat.
    similarity_weight: float = 0.7  # weight for embedding cosine in combined score
    candidate_entity_weight: float = 0.3  # deprecated alias: mirrors 1 - similarity_weight
    judge_per_claim: int = 3
    candidate_sim_weight: float = 0.7  # deprecated alias for similarity_weight
    candidate_ent_weight: float = 0.3  # deprecated alias for 1 - similarity_weight
    candidate_min_similarity: float = 0.0  # deprecated: not used
    candidate_require_entity_overlap: bool = False  # deprecated: not used
    candidate_intra_doc_dedup: bool = True  # deprecated: dedup now handled via hash
    # Cap on candidate pairs judged per pipeline run (cost guard).
    judge_top_n: int = 50

    # Verification (M4): lexical span matching + LLM entailment.
    entail_min_confidence: float = 0.8

    # Claim extraction (M2)
    chunk_max_chars: int = 1200
    gliner_model: str = "urchade/gliner_medium-v2.1"
    gliner_threshold: float = 0.5
    canonicalize_fuzzy_threshold: float = 0.88
    canonicalize_llm_merge: bool = False
    extract_labels: list[str] = [
        "PERSON",
        "ORGANIZATION",
        "PROJECT",
        "PRODUCT",
        "PLACE",
        "DATE",
        "TIME",
        "NUMBER",
        "MONEY",
        "AMOUNT",
        "TECHNOLOGY",
        "SKILL",
        "TOPIC",
        "URL",
    ]

    # Paths
    db_path: Path = Path("data/db/adonis.sqlite")
    corpus_dir: Path = Path("data/corpus")
    reports_dir: Path = Path("reports")

    # Google Drive live sync (OAuth)
    google_client_id: str | None = None
    google_client_secret: str | None = None
    google_redirect_uri: str | None = None  # defaults to http://127.0.0.1:8000/api/connections/drive/callback
    drive_sync_interval_s: int = 300

    # Uploads
    max_upload_mb: int = 100


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return cached settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reload_settings() -> Settings:
    """Re-read .env / environment and rebuild the cached settings."""
    global _settings
    _settings = Settings()
    return _settings


def provider_for_tier(settings: Settings, tier: str) -> Provider:
    """Effective provider for a tier: override wins, else the global one."""
    if tier == "extractor":
        return settings.extractor_provider or settings.llm_provider
    if tier == "judge":
        return settings.judge_provider or settings.llm_provider
    raise ValueError(f"unknown tier: {tier!r}; expected 'extractor' or 'judge'")


def save_settings(patch: dict[str, str]) -> Path:
    """Merge ADONIS_* values into .env and return its path.

    Existing KEY=value lines are updated in place (comments preserved);
    new keys are appended. Missing values (empty string) are ignored so the
    UI can leave fields blank to keep the current value. Values containing
    newlines or leading '=' are rejected to prevent injection.
    """
    # Validate patch: reject injection characters
    for k, v in patch.items():
        if not k.startswith("ADONIS_"):
            raise ValueError(f"unexpected env key {k!r}")
        if "\n" in v or "\r" in v:
            raise ValueError(f"value for {k} contains newline")
    env_path = Path(".env")
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    written: set[str] = set()

    def _update(line: str) -> str | None:
        if "=" not in line:
            return None
        key = line.split("=", 1)[0].strip()
        if patch.get(key):
            value = patch[key]
            written.add(key)
            return f"{key}={value}"
        return None

    out: list[str] = []
    for line in lines:
        updated = _update(line)
        if updated is not None:
            out.append(updated)
        else:
            out.append(line)
    for key, value in patch.items():
        if value and key not in written and not any(
            l.split("=", 1)[0].strip() == key for l in out
        ):
            out.append(f"{key}={value}")
    env_path.write_text("\n".join(out) + ("\n" if out else ""), encoding="utf-8")
    try:
        env_path.chmod(0o600)
    except OSError:
        pass
    return env_path


def mask_secret(secret: str) -> str:
    """Display-safe mask: keep last 4 chars, blank if empty."""
    if not secret:
        return ""
    if len(secret) <= 4:
        return "*" * len(secret)
    return "…" * 6 + secret[-4:]
