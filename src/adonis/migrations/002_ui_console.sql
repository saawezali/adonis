-- 002_ui_console.sql — console jobs + connections + staged labels

CREATE TABLE IF NOT EXISTS jobs (
  id          TEXT PRIMARY KEY,
  kind        TEXT NOT NULL,              -- 'pipeline' | 'extract' | 'ingest'
  status      TEXT NOT NULL,              -- 'queued' | 'running' | 'done' | 'error'
  params_json TEXT,
  result_json TEXT,
  error       TEXT,
  created_at  TEXT NOT NULL,
  started_at  TEXT,
  finished_at TEXT
);

CREATE TABLE IF NOT EXISTS connections (
  id                    TEXT PRIMARY KEY,
  kind                  TEXT NOT NULL,    -- 'local' | 'notion' | 'drive'
  name                  TEXT NOT NULL,
  config_json           TEXT NOT NULL,    -- {path, folder_id, ...}
  status                TEXT NOT NULL,    -- 'connected' | 'disconnected' | 'error' | 'syncing'
  last_sync_at          TEXT,
  last_sync_stats_json  TEXT,
  error                 TEXT,
  created_at            TEXT NOT NULL
);

-- Staged eval labels: human labels that need review before they count
-- towards metrics. Approved rows are copied to eval_labels (used_in_eval=1).
CREATE TABLE IF NOT EXISTS staged_labels (
  id            TEXT PRIMARY KEY,
  claim_a_id    TEXT NOT NULL REFERENCES claims(id),
  claim_b_id    TEXT NOT NULL REFERENCES claims(id),
  doc_a_id      TEXT NOT NULL REFERENCES documents(id),
  doc_b_id      TEXT NOT NULL REFERENCES documents(id),
  span_a_start  INTEGER NOT NULL,
  span_a_end    INTEGER NOT NULL,
  span_b_start  INTEGER NOT NULL,
  span_b_end    INTEGER NOT NULL,
  label         TEXT NOT NULL,
  notes         TEXT,
  labeled_by    TEXT NOT NULL,
  status        TEXT NOT NULL DEFAULT 'pending', -- pending | approved | rejected
  reviewed_by   TEXT,
  reviewed_at   TEXT,
  created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_staged_status ON staged_labels(status);
CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_conn_kind ON connections(kind);
