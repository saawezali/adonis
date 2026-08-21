# Changelog

All notable changes to Adonis. Format: [Keep a Changelog](https://keepachangelog.com/), versioning `SemVer`.

## [0.1.0] — 2026-08-21

### Added
* Provider-independent `LLMClient` (anthropic/openai/custom tiers), local console at `http://127.0.0.1:8000/` (Dashboard/Documents/Connections/Pipeline/Flags/Eval/Settings).
* Ingest (local .md/.txt, Notion .zip, Drive .docx/.gdoc) with sandboxed corpus, zip-slip guard, content-hash dedup.
* Claim extraction (chunk → LLM `claims_v1` + temporal/scope + triviality 0.5), GLiNER NER, rapidfuzz canonicalization.
* Candidate generation (FAISS `IndexFlatIP` + hybrid scoring `similarity_weight 0.7`, ordered `a<b`, intra-doc skip).
* Judge `judge_v1` + verification (verbatim/fuzzy + LLM entailment 0.8) + calibrated flags + Jinja2 report.
* Eval harness (per-category P/R/F1, micro/macro, detection, faithfulness) + trick set + `measure_recall` recall@K.
* Security: `.env` 600, `\n` guard, LIKE escape, batched deletes (900), streaming upload (1 MB chunks, 50 files), random OAuth `state`, job cap 3.
* Migrations `001_init` → `002_ui_console` → `003_fix_uniqueness` (ordered pairs, flag uniqueness).
* `adonis` / `adonis-doctor` entry points, `Dockerfile` + `docker-compose.yml`, `adonis.spec` (PyInstaller), synthetic sample generator (`scripts/generate_sample_corpus.py`).

### Fixed
* 30 audit findings remediated (candidate `a<b`, flag uniqueness, span bounds, declarative/triviality, micro semantics, gdrive URL).
* Upload streaming, path sandbox, batch deletes, OAuth state, drive token persist, hybrid stats.

## [0.0.1] — initial scaffolding
* Repo + milestones M1–M5 scaffolding.
