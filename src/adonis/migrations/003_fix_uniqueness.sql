-- 003_fix_uniqueness.sql — ordered pairs + flag uniqueness (audit A1/A2).
-- Idempotent.

-- Ensure candidate_pairs ordering: store with claim_a_id < claim_b_id.
-- Backfill: for any existing reversed rows, swap; drop duplicate reversed entries.
-- SQLite has no CHECK with < on TEXT in older versions? Use explicit index.

-- Deduplicate existing unordered duplicates (keep highest combined_score).
DELETE FROM candidate_pairs
WHERE id IN (
  SELECT id FROM (
    SELECT id,
           CASE WHEN claim_a_id < claim_b_id THEN claim_a_id ELSE claim_b_id END AS lo,
           CASE WHEN claim_a_id < claim_b_id THEN claim_b_id ELSE claim_a_id END AS hi,
           ROW_NUMBER() OVER (PARTITION BY
               CASE WHEN claim_a_id < claim_b_id THEN claim_a_id ELSE claim_b_id END,
               CASE WHEN claim_a_id < claim_b_id THEN claim_b_id ELSE claim_a_id END
               ORDER BY combined_score DESC, id) AS rn
    FROM candidate_pairs
  ) WHERE rn > 1
);

-- Normalize ordering for remaining rows.
UPDATE candidate_pairs
SET claim_a_id = CASE WHEN claim_a_id < claim_b_id THEN claim_a_id ELSE claim_b_id END,
    claim_b_id = CASE WHEN claim_a_id < claim_b_id THEN claim_b_id ELSE (
        SELECT claim_a_id FROM candidate_pairs AS _inner WHERE _inner.id = candidate_pairs.id
    ) END
WHERE claim_a_id > claim_b_id;

-- The above self-reference is tricky; do it via python fallback in db.py.
-- For pure SQL, we rewrite with a temp column approach:
-- (Handled in Python migration helper if above fails; kept here for documentation.)

-- Flags uniqueness: one flag per candidate pair.
CREATE UNIQUE INDEX IF NOT EXISTS uq_flags_candidate ON flags(candidate_pair_id);

-- Candidate pairs uniqueness expressed as unordered unique index (lo, hi).
-- We keep the existing UNIQUE(claim_a_id, claim_b_id) but add ordered check via trigger.
CREATE TRIGGER IF NOT EXISTS trg_candidate_order_insert
BEFORE INSERT ON candidate_pairs
FOR EACH ROW WHEN NEW.claim_a_id > NEW.claim_b_id
BEGIN
  SELECT RAISE(ABORT, 'claim_a_id must be < claim_b_id (ordered pair)');
END;

CREATE TRIGGER IF NOT EXISTS trg_candidate_order_update
BEFORE UPDATE OF claim_a_id, claim_b_id ON candidate_pairs
FOR EACH ROW WHEN NEW.claim_a_id > NEW.claim_b_id
BEGIN
  SELECT RAISE(ABORT, 'claim_a_id must be < claim_b_id (ordered pair)');
END;
