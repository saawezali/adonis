"""Typed application configuration, loaded from environment / .env.

Per PLAN.md section 2: provider-independent. Concrete adapter is selected by
`ADONIS_LLM_PROVIDER`; two tiers (extractor, judge) are configured independently.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Configuration. Reads from environment and .env."""

    model_config = SettingsConfigDict(
        env_prefix="ADONIS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM
    llm_provider: Literal["anthropic", "openai"] = "anthropic"
    llm_api_key: str = ""
    llm_base_url: str | None = None
    extractor_model: str = "claude-3-5-haiku-latest"
    judge_model: str = "claude-3-5-sonnet-latest"

    # Embeddings / candidate generation
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    top_k: int = 20
    triviality_cutoff: float = 0.3
    span_fuzzy_threshold: int = 90
    candidate_entity_weight: float = 0.3
    judge_per_claim: int = 3
    candidate_sim_weight: float = 0.7
    candidate_ent_weight: float = 0.3
    candidate_min_similarity: float = 0.0
    candidate_require_entity_overlap: bool = False
    candidate_intra_doc_dedup: bool = True

    # Candidate pairs (M3): combined = similarity_weight * cosine
    # + (1 - similarity_weight) * entity_overlap_jaccard.
    similarity_weight: float = 0.7
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


_settings: Settings | None = None


def get_settings() -> Settings:
    """Return cached settings instance."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
