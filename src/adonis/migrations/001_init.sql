-- 001_init.sql — initial schema per PLAN.md section 4.
-- Foreign keys are enforced by the connection (PRAGMA foreign_keys = ON).

CREATE TABLE IF NOT EXISTS documents (
  id                  TEXT PRIMARY KEY,
  source              TEXT NOT NULL,
  source_id           TEXT,
  title               TEXT,
  path                TEXT,
  format              TEXT,
  raw_text            TEXT NOT NULL,
  metadata_json       TEXT,
  content_hash        TEXT NOT NULL UNIQUE,
  ingested_at         TEXT NOT NULL,
  parse_warnings_json TEXT
);

CREATE TABLE IF NOT EXISTS entities (
  id              TEXT PRIMARY KEY,
  canonical_name  TEXT NOT NULL,
  aliases_json    TEXT NOT NULL,
  mention_count   INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS claims (
  id                  TEXT PRIMARY KEY,
  document_id         TEXT NOT NULL REFERENCES documents(id),
  claim_text          TEXT NOT NULL,
  citation_span_start INTEGER NOT NULL,
  citation_span_end   INTEGER NOT NULL,
  entities_json       TEXT NOT NULL,
  topics_json         TEXT NOT NULL,
  temporal_json       TEXT,
  scope_json          TEXT,
  triviality_score    REAL,                    -- 0..1, higher = more trivial/noise
  extraction_model    TEXT NOT NULL,
  extraction_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_claims_doc ON claims(document_id);

CREATE TABLE IF NOT EXISTS entity_mentions (
  id           TEXT PRIMARY KEY,
  claim_id     TEXT NOT NULL REFERENCES claims(id),
  entity_id    TEXT NOT NULL REFERENCES entities(id),
  mention_text TEXT NOT NULL,
  span_start   INTEGER NOT NULL,
  span_end     INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_em_claim  ON entity_mentions(claim_id);
CREATE INDEX IF NOT EXISTS idx_em_entity ON entity_mentions(entity_id);

CREATE TABLE IF NOT EXISTS candidate_pairs (
  id                  TEXT PRIMARY KEY,
  claim_a_id          TEXT NOT NULL REFERENCES claims(id),
  claim_b_id          TEXT NOT NULL REFERENCES claims(id),
  similarity_score    REAL NOT NULL,
  entity_overlap      REAL NOT NULL,
  combined_score      REAL NOT NULL,
  strategy            TEXT NOT NULL,
  selected_for_judge  INTEGER NOT NULL DEFAULT 0,
  created_at          TEXT NOT NULL,
  UNIQUE(claim_a_id, claim_b_id)
);
CREATE INDEX IF NOT EXISTS idx_cp_a        ON candidate_pairs(claim_a_id);
CREATE INDEX IF NOT EXISTS idx_cp_b        ON candidate_pairs(claim_b_id);
CREATE INDEX IF NOT EXISTS idx_cp_selected ON candidate_pairs(selected_for_judge);

CREATE TABLE IF NOT EXISTS judge_outputs (
  id                  TEXT PRIMARY KEY,
  candidate_pair_id   TEXT NOT NULL REFERENCES candidate_pairs(id),
  label               TEXT NOT NULL,
  judge_confidence    REAL NOT NULL,
  reasoning_text      TEXT NOT NULL,
  cited_span_a_start  INTEGER,
  cited_span_a_end    INTEGER,
  cited_span_b_start  INTEGER,
  cited_span_b_end    INTEGER,
  judge_model         TEXT NOT NULL,
  prompt_version      TEXT NOT NULL,
  judged_at           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS verification_results (
  id                    TEXT PRIMARY KEY,
  judge_output_id       TEXT NOT NULL REFERENCES judge_outputs(id),
  span_a_verbatim       INTEGER NOT NULL,
  span_a_fuzzy          REAL NOT NULL,
  span_a_entailment     REAL NOT NULL,
  span_a_pass           INTEGER NOT NULL,
  span_b_verbatim       INTEGER NOT NULL,
  span_b_fuzzy          REAL NOT NULL,
  span_b_entailment     REAL NOT NULL,
  span_b_pass           INTEGER NOT NULL,
  overall_pass          INTEGER NOT NULL,
  verified_at           TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS flags (
  id                  TEXT PRIMARY KEY,
  candidate_pair_id   TEXT NOT NULL REFERENCES candidate_pairs(id),
  final_label         TEXT NOT NULL,
  final_confidence    REAL NOT NULL,
  user_decision       TEXT,
  user_decision_at    TEXT,
  notes               TEXT
);

CREATE TABLE IF NOT EXISTS eval_labels (
  id            TEXT PRIMARY KEY,
  claim_a_id    TEXT REFERENCES claims(id),
  claim_b_id    TEXT REFERENCES claims(id),
  doc_a_id      TEXT REFERENCES documents(id),
  doc_b_id      TEXT REFERENCES documents(id),
  span_a_start  INTEGER,
  span_a_end    INTEGER,
  span_b_start  INTEGER,
  span_b_end    INTEGER,
  label         TEXT NOT NULL,
  notes         TEXT,
  labeled_by    TEXT NOT NULL,
  labeled_at    TEXT NOT NULL,
  used_in_eval  INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS llm_calls (
  id                TEXT PRIMARY KEY,
  stage             TEXT NOT NULL,
  model             TEXT NOT NULL,
  prompt_version    TEXT NOT NULL,
  prompt_tokens     INTEGER,
  completion_tokens INTEGER,
  latency_ms        INTEGER,
  success           INTEGER NOT NULL,
  error             TEXT,
  called_at         TEXT NOT NULL
);
