# Adonis — Contradiction Finder Across Your Docs

See `PLAN.md` for the full project plan. This README tracks implementation status.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -m spacy download en_core_web_sm
```

## Configure

Copy `.env.example` → `.env` and fill in, or run the web console and do it
from the settings page (recommended):

```bash
python -m adonis.web          # -> http://127.0.0.1:8000/
```

The settings page lets you:
- pick an LLM provider (`anthropic`, `openai`, or `custom` — any
  OpenAI-compatible endpoint such as Ollama / vLLM / LM Studio),
- enter the API key (stored in `.env`, permissions 600) and test the
  connection before saving,
- set per-tier providers/models (extractor vs judge) and a base URL
  override, then run the pipeline from the same page.

Manual `.env` knobs: `ADONIS_LLM_PROVIDER`, `ADONIS_LLM_API_KEY`,
`ADONIS_EXTRACTOR_PROVIDER` / `ADONIS_JUDGE_PROVIDER` (optional tier
overrides), `ADONIS_EXTRACTOR_MODEL` / `ADONIS_JUDGE_MODEL`, and
`ADONIS_LLM_BASE_URL` (required for `custom`; e.g. `http://localhost:11434/v1`).

## Run

```bash
# M1 — ingest
python scripts/ingest_corpus.py --in data/corpus/sample/

# M2 — extract claims + entities (demo mode, no API key needed)
python scripts/extract_claims.py
# Real extraction (requires .env):
python scripts/extract_claims.py --llm

# M2 — hand-label claim pairs (interactive)
python -m adonis.cli.label_pairs

# M3 — candidates + judge (demo mode, no API key needed)
python scripts/run_pipeline.py
# Real judging (requires .env):
python scripts/run_pipeline.py --llm
python scripts/measure_recall.py       # recall@K vs the eval labels
python scripts/eval_trick_set.py       # judge vs the curated trick set
python scripts/eval_trick_set.py --llm # same, with the real judge

# M4 — verification + flags + HTML report
python scripts/run_pipeline.py --refresh   # re-judge everything, verify, fill flags
python scripts/render_report.py            # -> reports/index.html (open in browser)

# M5 — eval harness over the labeled set
python -m adonis.cli.label_pairs --pool entity      # flag-relevant pairs (contradictions)
python -m adonis.cli.label_pairs --pool near_dup    # high-similarity pairs (near_dup TNs)
python -m adonis.cli.label_pairs --pool unrelated   # no-shared-entity pairs (unrelated TNs)
python scripts/eval_report.py                       # per-category P/R/F1 + detection rates -> reports/eval.json
```

The extraction script downloads the GLiNER model (`urchade/gliner_medium-v2.1`,
~150MB) on first use; the pipeline downloads the embedder
(`BAAI/bge-small-en-v1.5`) on first use. `--llm` uses the configured tier
model; without it, deterministic demo clients exercise the pipeline offline.
Demo judge flags can overshoot: it applies only the coarse §1.4 branches
and calls any differently-worded pair a contradiction.

## Milestone status

- [x] M1 — Ingestion + normalized store (parsers: local .md/.txt, Notion export zip, Drive .docx; .gdoc skipped w/ note; dedupe by content hash; parse-failure tracking; LLM adapters: anthropic + openai)
- [x] M2 — Claim + entity extraction (sentence chunking, claims_v1 prompt w/ citation spans + temporal/scope + triviality filter, GLiNER NER over claim spans, entity canonicalization, llm_calls tracing, label CLI)
- [x] M3 — Candidate pair generation + first end-to-end smoke test (embed + FAISS top-K + entity-overlap candidates, hybrid scoring, judge_v1 prompt with span-level citations, superseded_by_time rule, measure_recall, trick set; 93 tests)
- [x] M4 — Full pipeline + citation verification + confidence + report (lexical span match verbatim+fuzzy, LLM entailment check, verification_results for every judge output, flags only on verified pairs, numpy logistic-regression confidence calibration, Jinja2 HTML report with file:// source links)
- [x] M5 — Eval harness (per-category P/R/F1 with flag-avoidance semantics for true negatives, micro/macro, detection recall/precision, citation faithfulness; label pools entity/near_dup/unrelated; reports/eval.json artifact; 155 tests)

Beyond the PLAN: web console (`python -m adonis.web`) — provider setup + API key
entry + custom OpenAI-compatible inference provider (Ollama/vLLM/LM Studio),
per-tier provider/model overrides, connection test, pipeline run, and report
access from the browser (FastAPI, no React).

See `PLAN.md` for what's demoable at each milestone.
