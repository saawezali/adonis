# Adonis — Contradiction Finder Across Your Docs

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)](#) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE) [![Version 0.1.0](https://img.shields.io/badge/version-0.1.0-black)](#)

> Finds where two documents make conflicting claims about the same entity, with citation-grounded explanations — not just a flag. Local-first, provider-independent, built to show real AI systems engineering rather than a thin LLM wrapper.

This README is the operational guide — install, configure, run, and evaluate.

---

## 1. What it does

* Ingests a personal corpus (Notion export zips, Google Drive `.docx`/`.gdoc`, local `.md`/`.txt`) into a normalized SQLite store with content-hash dedup.
* Extracts **atomic declarative claims** `(entity, attribute, value, scope, time)` with citation offsets, temporal/scope props, and a triviality score.
* Detects entities with zero-shot **GLiNER**, canonicalizes via fuzzy clustering.
* Generates **candidate pairs** without O(n²) LLM calls: FAISS embedding top-K + entity-overlap hybrid scoring.
* **LLM-as-judge** classifies each pair into a 5-way taxonomy and cites spans.
* Verifies citations **lexically** (verbatim+fuzzy) and **semantically** (LLM entailment); only verified pairs become **flags**.
* Calibrates confidence (logistic regression on labeled pairs, heuristic fallback) and renders a sortable **HTML report** plus a full **local console** for triage.

### Taxonomy (non-binary by design)

| Condition (same entity/attribute) | Label |
|---|---|
| Same scope + same time, different values | `genuine_contradiction` |
| Same scope, later time, deliberate update | `superseded_by_time` |
| Different scope / sub-case | `different_scope` |
| Insufficient context | `ambiguous` |
| Same value, same scope, same time | `not_conflicting` |

Evaluation also tracks `true_negative_near_dup` and `true_negative_unrelated` for per-category P/R/F1.

---

## 2. Architecture

```
Ingest (notion/gdrive/local → normalize) → Chunk (spaCy) → Extract claims+temporal/scope (LLM)
  → NER (GLiNER over citation span) → Canonicalize (rapidfuzz) → Embed (bge-small) → FAISS IndexFlatIP
  → Candidate pairs (top-K=20 hybrid, entity fill, ordered a<b, intra-doc skip)
  → Judge (LLM, prompt judge_v1) → Verify (span_match + entailment) → Flags → Report/Console
                              ↘ llm_calls (tracing)        ↘ eval_labels → harness → eval.json
```

All stages are pure functions over SQLite tables, materialized so `recall@K` and thresholds can be tuned without re-embedding.

---

## 3. Tech stack

| Layer | Choice | Notes |
|---|---|---|
| Python | 3.11+ | `pyproject.toml:6` |
| LLM | provider-independent `LLMClient` — `anthropic` / `openai` / `custom` (any OpenAI-compatible: Ollama, vLLM, LM Studio) | two tiers: `extractor` (cheap/fast, `claude-3-5-haiku-latest` / `gpt-4o-mini`) and `judge` (smart, `claude-3-5-sonnet-latest` / `gpt-4o`); `get_client(tier)` |
| Embeddings | `sentence-transformers` `BAAI/bge-small-en-v1.5` (configurable) | L2-normalized, cosine via FAISS |
| NER | `GLiNER` `urchade/gliner_medium-v2.1` | zero-shot, threshold 0.5 |
| Vector search | `faiss-cpu` `IndexFlatIP` (switch to `IVFFlat` >50k claims — documented, not yet auto) |  |
| Store | SQLite (`sqlite3` + versioned `src/adonis/migrations/*.sql`) | `001_init` + `002_ui_console` (`jobs`, `connections`, `staged_labels`) + `003_fix_uniqueness` (ordered pairs, flag uniqueness) |
| Report | `Jinja2` → `reports/index.html` |  |
| Web console | `FastAPI` + `Uvicorn` + `python-multipart` | Dashboard / Documents / Connections / Pipeline / Flags / Eval / Settings |
| Docs | `python-docx` | Drive `.docx` |
| Config | `pydantic-settings` (`ADONIS_*` in `.env`) | typed, validated at startup, `chmod 600` |
| Quality | `pytest` + `ruff` + `mypy --strict` (`python_version 3.11`) | 164 tests |

---

## 4. Repo layout

```
adonis/
├── README.md
├── pyproject.toml / Makefile / .env.example
├── data/
│   ├── corpus/                     # gitignored — put your exports here
│   ├── db/adonis.sqlite            # gitignored
│   └── eval/                       # labeling_notes.md
├── src/adonis/
│   ├── config.py                   # Settings (see §5)
│   ├── db.py                       # get_conn, apply_migrations, CRUD, batched deletes
│   ├── migrations/{001_init,002_ui_console,003_fix_uniqueness}.sql
│   ├── llm/{client,anthropic,openai}.py + prompts/{claims_v1,judge_v1,entail_v1}.txt
│   ├── ingest/{local_notes,notion,gdrive,drive_sync,pipeline,base}.py
│   ├── normalize/text.py
│   ├── extract/{chunk,claims,entities,canonicalize}.py
│   ├── pair/{embed,index,candidates}.py
│   ├── judge/{classify,demo}.py
│   ├── verify/{span_match,entailment,demo}.py
│   ├── score/confidence.py
│   ├── report/{render,template.html.j2}
│   ├── eval/{harness,metrics}.py
│   ├── cli/label_pairs.py
│   ├── pipeline/core.py            # claims→flags orchestration
│   └── web/{app,static/app.html,__main__.py}
├── scripts/
│   ├── ingest_corpus.py
│   ├── extract_claims.py           # M2 entry, demo vs --llm
│   ├── run_pipeline.py             # M3/M4 entry, demo vs --llm
│   ├── render_report.py
│   ├── eval_report.py              # → reports/eval.json
│   ├── measure_recall.py           # recall@K
│   └── eval_trick_set.py
├── tests/{unit,integration,fixtures,conftest,fakes}.py
└── reports/{index.html,eval.json} # gitignored except template
```

---

## 5. Prerequisites

* Python 3.11+
* `pip` / `venv`
* Optional for real extraction: spaCy model (installed via `make install`), GLiNER + embedding models auto-download on first use (~150 MB + ~130 MB).

---

## 6. Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"          # or: make install
# make install also runs: python -m spacy download en_core_web_sm
```

Verify:

```bash
make test        # 164 tests
make lint        # ruff
make typecheck   # mypy src/adonis (strict, 3.11)
```

---

## 7. Configuration

Preferred: web console (**Settings** tab). Manual: copy `.env.example` → `.env`.

```bash
cp .env.example .env
# edit .env, then:
python -m adonis.web   # http://127.0.0.1:8000/  — new shell at / , legacy at /settings.html
```

Key knobs (see `src/adonis/config.py` + `.env.example`):

| Var | Default | Purpose |
|---|---|---|
| `ADONIS_LLM_PROVIDER` | `anthropic` | `anthropic` \| `openai` \| `custom` |
| `ADONIS_LLM_API_KEY` | `` | required unless `custom` local |
| `ADONIS_LLM_BASE_URL` | `` | required for `custom` (e.g. `http://localhost:11434/v1`) |
| `ADONIS_EXTRACTOR_PROVIDER` / `ADONIS_JUDGE_PROVIDER` | `` | per-tier override (falls back to global) |
| `ADONIS_EXTRACTOR_MODEL` | `claude-3-5-haiku-latest` |  |
| `ADONIS_JUDGE_MODEL` | `claude-3-5-sonnet-latest` |  |
| `ADONIS_EMBEDDING_MODEL` | `BAAI/bge-small-en-v1.5` |  |
| `ADONIS_TOP_K` | `20` | FAISS neighbors per claim |
| `ADONIS_JUDGE_PER_CLAIM` | `3` | pairs selected per claim for judging |
| `ADONIS_SIMILARITY_WEIGHT` | `0.7` | `combined = w·cosine + (1-w)·entity_overlap` (canonical; deprecated `CANDIDATE_ENTITY_WEIGHT` still reads) |
| `ADONIS_TRIVIALITY_CUTOFF` | `0.5` | drop claims whose `triviality_score ≥ cutoff` (also drops `<10 char` + markdown title noise) |
| `ADONIS_SPAN_FUZZY_THRESHOLD` | `90` | rapidfuzz ratio |
| `ADONIS_ENTAIL_MIN_CONFIDENCE` | `0.8` | entailment pass threshold |
| `ADONIS_CHUNK_MAX_CHARS` | `1200` |  |
| `ADONIS_GLINER_MODEL` / `THRESHOLD` | `urchade/gliner_medium-v2.1` / `0.5` |  |
| `ADONIS_DB_PATH` / `CORPUS_DIR` / `REPORTS_DIR` | `data/db/adonis.sqlite` etc. |  |
| `ADONIS_GOOGLE_CLIENT_ID/SECRET/REDIRECT_URI` | `` | Drive OAuth (see Settings → Connections) |
| `ADONIS_GOOGLE_REFRESH_TOKEN` | `` | persisted after OAuth; rotated tokens auto-saved |
| `ADONIS_MAX_UPLOAD_MB` | `100` | console upload limit (streamed, 1 MB chunks) |

The console **Test** button POSTs `/api/test` (no save) and **Save** merges into `.env` (`chmod 600`, rejects `\n`/`\r` injection, validates providers).

---

## 8. Quickstart

### A) Demo mode — no API key (exercises every stage offline)

```bash
# 1. Ingest (put .md/.txt, Notion .zip, or .docx under data/corpus/sample/)
python scripts/ingest_corpus.py --in data/corpus/sample/

# 2. Extract claims+entities (one claim per chunk, valid spans)
python scripts/extract_claims.py
# or: python -m adonis.web → Pipeline → Extract (demo)

# 3. Candidates → judge → verify → flags
python scripts/run_pipeline.py
# re-run with wipe: python scripts/run_pipeline.py --refresh

python scripts/render_report.py   # → reports/index.html
open reports/index.html

# Label a few pairs (interactive)
python -m adonis.cli.label_pairs --pool entity
python scripts/measure_recall.py              # recall@5/10/20/50
python scripts/eval_report.py                 # → reports/eval.json
```

Demo judge applies coarse decision branches and therefore may overshoot — run the trick set to see verification catch it: `python scripts/eval_trick_set.py`.

### B) Real LLM (needs `.env`)

```bash
# console is easiest: set provider+key+models, Test, Save
python -m adonis.web

# or CLI:
python scripts/extract_claims.py --llm
python scripts/run_pipeline.py --llm --refresh
python scripts/render_report.py
python scripts/eval_report.py
```

### C) Full console (recommended)

```bash
python -m adonis.web   # http://127.0.0.1:8000/
```

* **Dashboard** — counts (documents/claims/pairs/flags/eval) + pipeline entry.
* **Documents** — paginated search (`LIKE … ESCAPE \`), viewer, claims preview, **Upload** (auto-ingest, `rglob` of temp dir), **Delete** (DB row only, cascades claims/candidate_pairs/judge/verify/flags/labels with batched `IN (?,…)`).
* **Connections** — `local` / `notion` / `drive` (Drive OAuth: `/api/connections/drive/auth` → callback, random `state`, refresh token persisted).
* **Pipeline** — async jobs (`queued→running→done/error`, cap 3 concurrent) with polling, parameters `top_k` / `selected_per_claim` / `max_pairs`, plus Extract controls (`document_id`, `limit`, `use_llm`).
* **Flags** — filter by label/min_conf/q/verified, verification badges (`verbatim/fuzzy/entailment`), triage `PATCH /api/flags/{id}` (`confirmed|rejected|unsure`).
* **Eval** — `GET /api/eval` metrics + `reports/eval.json`, `GET /api/eval/status`, staged labels (`pending→approved/rejected`) feeding `eval_labels`.
* **Settings** — provider/base URL/model + Google OAuth, masked secrets.

---

## 9. CLI & Make reference

```bash
make ingest      # scripts/ingest_corpus.py --in data/corpus/sample/
make pipeline    # scripts/run_pipeline.py (demo)
make report      # scripts/render_report.py
make eval        # scripts/eval_report.py
make serve       # python -m adonis.web
make test        # pytest (164 tests, needs python-multipart)
make lint        # ruff check src tests
make typecheck   # mypy src/adonis
make clean       # rm db + reports/*.html
```

Key scripts:

```
scripts/ingest_corpus.py --in DIR [--db]
scripts/extract_claims.py [--document-id ID] [--limit N] [--llm]
scripts/run_pipeline.py [--llm] [--refresh] [--top-k N] [--selected-per-claim N] [--max-pairs N]
scripts/render_report.py
scripts/measure_recall.py [--limit N]
scripts/eval_report.py [--json PATH]
scripts/eval_trick_set.py [--llm]
python -m adonis.cli.label_pairs [--pool entity|near_dup|unrelated] [--limit 20] [--labeled-by cli]
```

---

## 10. Data model (SQLite — see `src/adonis/migrations/*.sql`)

`documents` (hash dedup, `raw_text` normalized) → `claims` (`citation_span_start/end` into `raw_text`, `entities_json`, `topics_json`, `temporal_json`, `scope_json`) → `entity_mentions` + `entities` → `candidate_pairs` (**ordered** `claim_a_id < claim_b_id`, `similarity/entity_overlap/combined`, `strategy=embedding|hybrid|entity`) → `judge_outputs` (`prompt_version` mandatory) → `verification_results` (`overall_pass`) → `flags` (unique per `candidate_pair_id`, calibrated `final_confidence`) → `eval_labels` / `staged_labels` + `llm_calls` + `jobs`/`connections`.

Migrations are idempotent (`schema_migrations`): `apply_migrations()` also backfills ordering/dedup and `uq_flags_candidate`.

---

## 11. Security & operational limits

* **Ingest sandbox** — `/api/ingest` `path` must resolve inside `corpus_dir` (403 otherwise).
* **Upload** — streamed 1 MB chunks, early 413 at `MAX_UPLOAD_MB`, max 50 files, temp dir `rglob` only.
* **Env** — `save_settings` rejects `\n`/`\r` and enforces `ADONIS_` prefix.
* **Search** — `LIKE` wildcards escaped (`\%`, `\_`, `\\`).
* **Deletes** — batched `IN (?,…)` (450/900 chunk) avoids `too many SQL variables`.
* **Notion zip** — rejects `/` or `..` entries (zip-slip), `BadZipFile` → `ignored` not crash.
* **OAuth** — random `state` per `auth_url`, server validates `state` on callback.
* **Jobs** — max 3 active; consider TTL prune for long-running instances.
* **Pair ordering** — `trg_candidate_order_*` aborts `a > b`; code always swaps before insert.

---

## 12. Evaluation

* Hand-label via `label_pairs` pools: `entity` (shared entity → contradictions), `near_dup` (`similarity ≥0.85` → TN near-dup), `unrelated` (no shared entity → TN unrelated). Interactive keys: `c/t/s/a/n/d/u/k/q`.
* Harness `eval/harness.py` joins every `eval_labels` row to its `candidate_pairs`/`judge_outputs`/`verification_results` and computes **per-category P/R/F1**, `micro/macro F1`, `detection recall/precision` (any flag vs contradiction), and `citation faithfulness` (`overall_pass` rate). `scripts/eval_report.py` prints a table and writes `reports/eval.json`.
* **Gate before M4** — `scripts/measure_recall.py` recall@20 ≥80% (top-K 5/10/20/50).

---

## 13. Troubleshooting

| Symptom | Fix |
|---|---|
| `FileNotFoundError: prompt not found: …/llm/prompts/claims_v1.txt` | run from repo root; ensure `src/adonis/llm/prompts/*.txt` present (new guard in `extract/claims.py`, `judge/classify.py`, `verify/entailment.py`) |
| `no claims in store` in pipeline | `python scripts/extract_claims.py` first (or via console) |
| Report shows “No flags yet” | pipeline demo flagged 0 or verification failed — check `pipeline` errors, try `--refresh`; real model may be more precise |
| `too many SQL variables` | fixed via batched deletes in `db.py` (update to latest) |
| `multipart_not_installed` in tests | `pip install python-multipart` (added to `pyproject.toml`) |
| `mypy` error about `numpy` stub on 3.11 | expected — numpy stub needs 3.12; warnings are non-blocking, code is `type: ignore` safe |

---

## 14. Out of scope

Multi-relation evidence graph, incremental file-watch, cross-corpus entity linking, active learning, multi-user/hosted, revision history, cross-language, fine-tuning.

---

## 15. Further reading

* `src/adonis/migrations/*.sql` — schema and comments (`apply.py`).
* `src/adonis/web/static/app.html` — console UI and API wiring.
* `LICENSE` — MIT.

License: MIT — see `LICENSE`.
