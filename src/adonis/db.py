"""SQLite connection helpers + schema creation.

Per PLAN.md section 4. The schema is applied by migrations/apply.py (idempotent),
which sources migrations/*.sql in order. Clients use get_conn() and never touch
the sqlite3 module directly.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from adonis.config import get_settings
from adonis.ingest.base import DocumentRecord
from adonis.normalize.text import content_hash

_MIGRATIONS_DIR = Path(__file__).parent / "migrations"


def get_conn() -> sqlite3.Connection:
    """Return a sqlite3 connection with foreign keys enabled and row factory set."""
    settings = get_settings()
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(settings.db_path))
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.row_factory = sqlite3.Row
    return conn


def apply_migrations() -> None:
    """Apply all SQL files in the migrations dir in filename order. Idempotent."""
    migrations: Iterable[Path] = sorted(_MIGRATIONS_DIR.glob("*.sql"))
    conn = get_conn()
    try:
        conn.executescript(
            "CREATE TABLE IF NOT EXISTS schema_migrations ("
            "  name TEXT PRIMARY KEY,"
            "  applied_at TEXT NOT NULL DEFAULT (datetime('now'))"
            ");"
        )
        already = {row["name"] for row in conn.execute("SELECT name FROM schema_migrations")}
        for path in migrations:
            if path.name in already:
                continue
            conn.executescript(path.read_text(encoding="utf-8"))
            conn.execute("INSERT INTO schema_migrations (name) VALUES (?)", (path.name,))
            conn.commit()
    finally:
        conn.close()


def insert_document(conn: sqlite3.Connection, record: DocumentRecord) -> bool:
    """Insert a normalized document.

    Returns True if inserted; False if a document with the same content_hash
    already exists (duplicate). Raises ValueError if raw_text is empty so a
    parse producing nothing fails loudly instead of storing garbage.
    """
    raw = record.raw_text.strip()
    if not raw:
        raise ValueError(f"empty raw_text for {record.path!r}")
    digest = content_hash(raw)
    existing = conn.execute(
        "SELECT id FROM documents WHERE content_hash = ?", (digest,)
    ).fetchone()
    if existing is not None:
        return False
    now = datetime.now(UTC).isoformat()
    conn.execute(
        "INSERT INTO documents (id, source, source_id, title, path, format,"
        " raw_text, metadata_json, content_hash, ingested_at, parse_warnings_json)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            uuid.uuid4().hex,
            record.source,
            record.source_id,
            record.title,
            record.path,
            record.format,
            raw,
            json.dumps(record.metadata),
            digest,
            now,
            json.dumps(record.warnings),
        ),
    )
    conn.commit()
    return True


def iter_documents(
    conn: sqlite3.Connection,
    *,
    document_id: str | None = None,
    limit: int | None = None,
) -> list[sqlite3.Row]:
    """Documents in ingestion order, optionally filtered by id and/or limited."""
    sql = "SELECT * FROM documents"
    params: tuple[object, ...] = ()
    if document_id is not None:
        sql += " WHERE id = ?"
        params = (document_id,)
    sql += " ORDER BY ingested_at"
    if limit is not None:
        sql += " LIMIT ?"
        params = (*params, limit)
    return list(conn.execute(sql, params))


def insert_claim(
    conn: sqlite3.Connection,
    *,
    document_id: str,
    claim_text: str,
    span_start: int,
    span_end: int,
    topics: list[str],
    temporal: dict[str, str] | None,
    scope: dict[str, str] | None,
    triviality_score: float,
    extraction_model: str,
    extraction_at: str,
) -> str:
    """Insert a claim and return its id."""
    claim_id = uuid.uuid4().hex
    conn.execute(
        "INSERT INTO claims (id, document_id, claim_text, citation_span_start,"
        " citation_span_end, entities_json, topics_json, temporal_json, scope_json,"
        " triviality_score, extraction_model, extraction_at)"
        " VALUES (?, ?, ?, ?, ?, '[]', ?, ?, ?, ?, ?, ?)",
        (
            claim_id,
            document_id,
            claim_text,
            span_start,
            span_end,
            json.dumps(topics),
            json.dumps(temporal) if temporal is not None else None,
            json.dumps(scope) if scope is not None else None,
            triviality_score,
            extraction_model,
            extraction_at,
        ),
    )
    return claim_id


def update_claim_entities(
    conn: sqlite3.Connection, claim_id: str, entity_ids: list[str]
) -> None:
    conn.execute(
        "UPDATE claims SET entities_json = ? WHERE id = ?",
        (json.dumps(entity_ids), claim_id),
    )


def upsert_entity(
    conn: sqlite3.Connection,
    *,
    canonical_name: str,
    aliases: list[str],
    mention_count: int,
) -> str:
    """Return the id of the entity with this canonical name, creating or
    extending it: aliases are unioned, mention_count is incremented."""
    row = conn.execute(
        "SELECT id, aliases_json, mention_count FROM entities WHERE canonical_name = ?",
        (canonical_name,),
    ).fetchone()
    if row is not None:
        entity_id: str = row["id"]
        merged = sorted(set(json.loads(row["aliases_json"])) | set(aliases))
        conn.execute(
            "UPDATE entities SET aliases_json = ?, mention_count = ? WHERE id = ?",
            (json.dumps(merged), row["mention_count"] + mention_count, entity_id),
        )
        return entity_id
    entity_id = uuid.uuid4().hex
    conn.execute(
        "INSERT INTO entities (id, canonical_name, aliases_json, mention_count)"
        " VALUES (?, ?, ?, ?)",
        (entity_id, canonical_name, json.dumps(sorted(set(aliases))), mention_count),
    )
    return entity_id


def insert_entity_mention(
    conn: sqlite3.Connection,
    *,
    claim_id: str,
    entity_id: str,
    mention_text: str,
    span_start: int,
    span_end: int,
) -> None:
    conn.execute(
        "INSERT INTO entity_mentions (id, claim_id, entity_id, mention_text,"
        " span_start, span_end) VALUES (?, ?, ?, ?, ?, ?)",
        (
            uuid.uuid4().hex,
            claim_id,
            entity_id,
            mention_text,
            span_start,
            span_end,
        ),
    )


def insert_llm_call(
    conn: sqlite3.Connection,
    *,
    stage: str,
    model: str,
    prompt_version: str,
    prompt_tokens: int | None = None,
    completion_tokens: int | None = None,
    latency_ms: int | None = None,
    success: bool = True,
    error: str | None = None,
) -> None:
    conn.execute(
        "INSERT INTO llm_calls (id, stage, model, prompt_version, prompt_tokens,"
        " completion_tokens, latency_ms, success, error, called_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            uuid.uuid4().hex,
            stage,
            model,
            prompt_version,
            prompt_tokens,
            completion_tokens,
            latency_ms,
            int(success),
            error,
            datetime.now(UTC).isoformat(),
        ),
    )


def insert_eval_label(
    conn: sqlite3.Connection,
    *,
    claim_a_id: str,
    claim_b_id: str,
    doc_a_id: str,
    doc_b_id: str,
    span_a_start: int,
    span_a_end: int,
    span_b_start: int,
    span_b_end: int,
    label: str,
    labeled_by: str,
    notes: str | None = None,
) -> None:
    conn.execute(
        "INSERT INTO eval_labels (id, claim_a_id, claim_b_id, doc_a_id, doc_b_id,"
        " span_a_start, span_a_end, span_b_start, span_b_end, label, notes,"
        " labeled_by, labeled_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            uuid.uuid4().hex,
            claim_a_id,
            claim_b_id,
            doc_a_id,
            doc_b_id,
            span_a_start,
            span_a_end,
            span_b_start,
            span_b_end,
            label,
            notes,
            labeled_by,
            datetime.now(UTC).isoformat(),
        ),
    )


def insert_candidate_pair(
    conn: sqlite3.Connection,
    *,
    claim_a_id: str,
    claim_b_id: str,
    similarity_score: float,
    entity_overlap: float,
    combined_score: float,
    strategy: str,
    selected_for_judge: bool = False,
) -> bool:
    """Materialize a candidate pair. Returns True if inserted, False if the
    (claim_a_id, claim_b_id) pair already exists (UNIQUE constraint)."""
    try:
        conn.execute(
            "INSERT INTO candidate_pairs (id, claim_a_id, claim_b_id,"
            " similarity_score, entity_overlap, combined_score, strategy,"
            " selected_for_judge, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                uuid.uuid4().hex,
                claim_a_id,
                claim_b_id,
                similarity_score,
                entity_overlap,
                combined_score,
                strategy,
                int(selected_for_judge),
                datetime.now(UTC).isoformat(),
            ),
        )
    except sqlite3.IntegrityError:
        return False
    return True


def insert_judge_output(
    conn: sqlite3.Connection,
    *,
    candidate_pair_id: str,
    label: str,
    judge_confidence: float,
    reasoning_text: str,
    cited_span_a_start: int,
    cited_span_a_end: int,
    cited_span_b_start: int,
    cited_span_b_end: int,
    judge_model: str,
    prompt_version: str,
) -> str:
    judge_output_id = uuid.uuid4().hex
    conn.execute(
        "INSERT INTO judge_outputs (id, candidate_pair_id, label, judge_confidence,"
        " reasoning_text, cited_span_a_start, cited_span_a_end, cited_span_b_start,"
        " cited_span_b_end, judge_model, prompt_version, judged_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            judge_output_id,
            candidate_pair_id,
            label,
            judge_confidence,
            reasoning_text,
            cited_span_a_start,
            cited_span_a_end,
            cited_span_b_start,
            cited_span_b_end,
            judge_model,
            prompt_version,
            datetime.now(UTC).isoformat(),
        ),
    )
    return judge_output_id


def load_claims(
    conn: sqlite3.Connection, *, document_id: str | None = None, limit: int | None = None
) -> list[sqlite3.Row]:
    """Claim rows for embedding/candidate generation."""
    sql = "SELECT id, document_id, claim_text, entities_json FROM claims"
    params: tuple[object, ...] = ()
    if document_id is not None:
        sql += " WHERE document_id = ?"
        params = (document_id,)
    sql += " ORDER BY extraction_at"
    if limit is not None:
        sql += " LIMIT ?"
        params = (*params, limit)
    return list(conn.execute(sql, params))


def load_unjudged_pairs(
    conn: sqlite3.Connection, *, limit: int | None = None
) -> list[sqlite3.Row]:
    """Selected candidate pairs without a judge output, with claim + doc context."""
    sql = """
        SELECT cp.id AS pair_id, cp.claim_a_id, cp.claim_b_id, cp.combined_score,
               c1.claim_text AS text_a, c2.claim_text AS text_b,
               c1.temporal_json AS temporal_a, c2.temporal_json AS temporal_b,
               c1.scope_json AS scope_a, c2.scope_json AS scope_b,
               d1.id AS doc_a_id, d2.id AS doc_b_id,
               d1.title AS doc_a_title, d2.title AS doc_b_title,
               d1.raw_text AS raw_a, d2.raw_text AS raw_b,
               c1.citation_span_start AS span_a_start, c1.citation_span_end AS span_a_end,
               c2.citation_span_start AS span_b_start, c2.citation_span_end AS span_b_end
        FROM candidate_pairs cp
        JOIN claims c1 ON c1.id = cp.claim_a_id
        JOIN claims c2 ON c2.id = cp.claim_b_id
        JOIN documents d1 ON d1.id = c1.document_id
        JOIN documents d2 ON d2.id = c2.document_id
        LEFT JOIN judge_outputs jo ON jo.candidate_pair_id = cp.id
        WHERE cp.selected_for_judge = 1 AND jo.id IS NULL
        ORDER BY cp.combined_score DESC
    """
    if limit is not None:
        sql += " LIMIT ?"
        return list(conn.execute(sql, (limit,)))
    return list(conn.execute(sql))


def insert_verification_result(
    conn: sqlite3.Connection,
    *,
    judge_output_id: str,
    span_a_verbatim: bool,
    span_a_fuzzy: float,
    span_a_entailment: float,
    span_a_pass: bool,
    span_b_verbatim: bool,
    span_b_fuzzy: float,
    span_b_entailment: float,
    span_b_pass: bool,
) -> str:
    """Persist one verification pass for a judge output. Returns row id."""
    verification_id = uuid.uuid4().hex
    conn.execute(
        "INSERT INTO verification_results (id, judge_output_id, span_a_verbatim,"
        " span_a_fuzzy, span_a_entailment, span_a_pass, span_b_verbatim,"
        " span_b_fuzzy, span_b_entailment, span_b_pass, overall_pass, verified_at)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            verification_id,
            judge_output_id,
            int(span_a_verbatim),
            span_a_fuzzy,
            span_a_entailment,
            int(span_a_pass),
            int(span_b_verbatim),
            span_b_fuzzy,
            span_b_entailment,
            int(span_b_pass),
            int(span_a_pass and span_b_pass),
            datetime.now(UTC).isoformat(),
        ),
    )
    return verification_id


def insert_flag(
    conn: sqlite3.Connection,
    *,
    candidate_pair_id: str,
    final_label: str,
    final_confidence: float,
    notes: str | None = None,
) -> str:
    """Surface a verified contradiction as a flag. Returns row id."""
    flag_id = uuid.uuid4().hex
    conn.execute(
        "INSERT INTO flags (id, candidate_pair_id, final_label, final_confidence,"
        " user_decision, user_decision_at, notes)"
        " VALUES (?, ?, ?, ?, NULL, NULL, ?)",
        (flag_id, candidate_pair_id, final_label, final_confidence, notes),
    )
    return flag_id


def load_flags_with_context(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Flags joined with pairs, claims, documents, and verification for reports."""
    return list(
        conn.execute(
            """
            SELECT f.id AS flag_id, f.final_label, f.final_confidence, f.user_decision,
                   f.notes,
                   c1.claim_text AS text_a, c2.claim_text AS text_b,
                   d1.title AS doc_a_title, d1.path AS doc_a_path,
                   d2.title AS doc_b_title, d2.path AS doc_b_path,
                   d1.raw_text AS raw_a, d2.raw_text AS raw_b,
                   jo.cited_span_a_start, jo.cited_span_a_end,
                   jo.cited_span_b_start, jo.cited_span_b_end,
                   c1.citation_span_start AS claim_a_start,
                   c2.citation_span_start AS claim_b_start,
                   vr.span_a_verbatim, vr.span_b_verbatim,
                   vr.span_a_fuzzy, vr.span_b_fuzzy,
                   vr.span_a_entailment, vr.span_b_entailment,
                   vr.overall_pass, jo.judge_model, jo.prompt_version,
                   jo.reasoning_text
            FROM flags f
            JOIN candidate_pairs cp ON cp.id = f.candidate_pair_id
            JOIN claims c1 ON c1.id = cp.claim_a_id
            JOIN claims c2 ON c2.id = cp.claim_b_id
            JOIN documents d1 ON d1.id = c1.document_id
            JOIN documents d2 ON d2.id = c2.document_id
            JOIN judge_outputs jo ON jo.candidate_pair_id = cp.id
            JOIN verification_results vr ON vr.judge_output_id = jo.id
            ORDER BY f.final_confidence DESC
            """
        )
    )


def load_judged_pairs_with_verification(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Every judge output with its verification, for faithfulness stats."""
    return list(
        conn.execute(
            """
            SELECT jo.id AS judge_output_id, jo.label, jo.judge_confidence,
                   jo.judge_model,
                   vr.overall_pass, vr.span_a_pass, vr.span_b_pass,
                   vr.span_a_verbatim, vr.span_b_verbatim,
                   vr.span_a_fuzzy, vr.span_b_fuzzy,
                   vr.span_a_entailment, vr.span_b_entailment
            FROM judge_outputs jo
            LEFT JOIN verification_results vr ON vr.judge_output_id = jo.id
            """
        )
    )


def load_labeled_pairs(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """eval_labels rows usable for recall@K (contradiction categories only)."""
    return list(
        conn.execute(
            "SELECT claim_a_id, claim_b_id FROM eval_labels"
            " WHERE label NOT IN ('true_negative_unrelated', 'true_negative_near_dup')"
        )
    )


# ---------------------------------------------------------------------------
# Console extensions (002) — jobs / connections / staged labels / documents
# ---------------------------------------------------------------------------

def delete_document(conn: sqlite3.Connection, document_id: str) -> dict[str, int]:
    """Hard-delete a document row only (per spec: DB row, not original file).

    Cascades through claims → entity_mentions / candidate_pairs → judge →
    verification → flags → staged/eval labels linked by doc. Uses FK-aware
    manual deletes (SQLite FKs not cascaded in schema).
    Returns counts per table deleted.
    """
    cur = conn.execute("SELECT id FROM documents WHERE id = ?", (document_id,))
    if cur.fetchone() is None:
        return {"documents": 0}
    # Collect claim ids for this doc.
    claim_ids = [r["id"] for r in conn.execute("SELECT id FROM claims WHERE document_id = ?", (document_id,))]
    counts: dict[str, int] = {}
    if claim_ids:
        qmarks = ",".join("?" for _ in claim_ids)
        # Candidate pairs involving those claims -> collect pair ids.
        pair_rows = conn.execute(
            f"SELECT id FROM candidate_pairs WHERE claim_a_id IN ({qmarks}) OR claim_b_id IN ({qmarks})",
            (*claim_ids, *claim_ids),
        ).fetchall()
        pair_ids = [r["id"] for r in pair_rows]
        if pair_ids:
            pq = ",".join("?" for _ in pair_ids)
            # Judge outputs for those pairs.
            jo_rows = conn.execute(f"SELECT id FROM judge_outputs WHERE candidate_pair_id IN ({pq})", tuple(pair_ids)).fetchall()
            jo_ids = [r["id"] for r in jo_rows]
            if jo_ids:
                jq = ",".join("?" for _ in jo_ids)
                vr = conn.execute(f"DELETE FROM verification_results WHERE judge_output_id IN ({jq})", tuple(jo_ids))
                counts["verification_results"] = vr.rowcount
                # flags linked to pairs (not jo) — delete via pair ids
                fg = conn.execute(f"DELETE FROM flags WHERE candidate_pair_id IN ({pq})", tuple(pair_ids))
                counts["flags"] = fg.rowcount
                jo_del = conn.execute(f"DELETE FROM judge_outputs WHERE id IN ({jq})", tuple(jo_ids))
                counts["judge_outputs"] = jo_del.rowcount
            else:
                # still clean flags even without jo
                fg = conn.execute(f"DELETE FROM flags WHERE candidate_pair_id IN ({pq})", tuple(pair_ids))
                counts["flags"] = fg.rowcount
            # staged labels referencing these claims
            sl = conn.execute(
                f"DELETE FROM staged_labels WHERE claim_a_id IN ({qmarks}) OR claim_b_id IN ({qmarks})",
                (*claim_ids, *claim_ids),
            )
            counts["staged_labels"] = sl.rowcount
            # eval labels
            el = conn.execute(
                f"DELETE FROM eval_labels WHERE claim_a_id IN ({qmarks}) OR claim_b_id IN ({qmarks}) OR doc_a_id = ? OR doc_b_id = ?",
                (*claim_ids, *claim_ids, document_id, document_id),
            )
            counts["eval_labels"] = el.rowcount
            cp = conn.execute(f"DELETE FROM candidate_pairs WHERE id IN ({pq})", tuple(pair_ids))
            counts["candidate_pairs"] = cp.rowcount
        else:
            # no pairs, but still staged/eval
            sl = conn.execute(
                f"DELETE FROM staged_labels WHERE claim_a_id IN ({qmarks}) OR claim_b_id IN ({qmarks})",
                (*claim_ids, *claim_ids),
            )
            counts["staged_labels"] = sl.rowcount
            el = conn.execute(
                f"DELETE FROM eval_labels WHERE claim_a_id IN ({qmarks}) OR claim_b_id IN ({qmarks}) OR doc_a_id = ? OR doc_b_id = ?",
                (*claim_ids, *claim_ids, document_id, document_id),
            )
            counts["eval_labels"] = el.rowcount
        # entity_mentions
        em = conn.execute(f"DELETE FROM entity_mentions WHERE claim_id IN ({qmarks})", tuple(claim_ids))
        counts["entity_mentions"] = em.rowcount
        cl = conn.execute(f"DELETE FROM claims WHERE document_id = ?", (document_id,))
        counts["claims"] = cl.rowcount
    # staged/eval that reference doc directly even without claims (defensive)
    if "staged_labels" not in counts:
        sl = conn.execute("DELETE FROM staged_labels WHERE doc_a_id = ? OR doc_b_id = ?", (document_id, document_id))
        counts["staged_labels"] = sl.rowcount
        el = conn.execute("DELETE FROM eval_labels WHERE doc_a_id = ? OR doc_b_id = ?", (document_id, document_id))
        counts["eval_labels"] = el.rowcount
    doc = conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
    counts["documents"] = doc.rowcount
    conn.commit()
    return counts


def list_documents(
    conn: sqlite3.Connection,
    *,
    q: str | None = None,
    source: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[sqlite3.Row], int]:
    """Paginated document listing with optional search/source filter. Returns (rows, total)."""
    where: list[str] = []
    params: list[object] = []
    if q:
        where.append("(title LIKE ? OR raw_text LIKE ? OR path LIKE ?)")
        like = f"%{q}%"
        params.extend([like, like, like])
    if source:
        where.append("source = ?")
        params.append(source)
    where_sql = f" WHERE {' AND '.join(where)}" if where else ""
    total = conn.execute(f"SELECT COUNT(*) FROM documents{where_sql}", tuple(params)).fetchone()[0]
    rows = conn.execute(
        f"SELECT id, source, source_id, title, path, format, length(raw_text) AS text_len, content_hash, ingested_at, parse_warnings_json, metadata_json FROM documents{where_sql} ORDER BY ingested_at DESC LIMIT ? OFFSET ?",
        (*params, limit, offset),
    ).fetchall()
    return list(rows), int(total)


def list_claims_for_document(
    conn: sqlite3.Connection, document_id: str, *, limit: int = 100, offset: int = 0
) -> tuple[list[sqlite3.Row], int]:
    total = conn.execute("SELECT COUNT(*) FROM claims WHERE document_id = ?", (document_id,)).fetchone()[0]
    rows = conn.execute(
        "SELECT id, claim_text, citation_span_start, citation_span_end, entities_json, topics_json, temporal_json, scope_json, triviality_score, extraction_model FROM claims WHERE document_id = ? ORDER BY citation_span_start LIMIT ? OFFSET ?",
        (document_id, limit, offset),
    ).fetchall()
    return list(rows), int(total)


# Jobs helpers
def insert_job(conn: sqlite3.Connection, kind: str, params: dict[str, object] | None = None) -> str:
    jid = uuid.uuid4().hex
    conn.execute(
        "INSERT INTO jobs (id, kind, status, params_json, created_at) VALUES (?, ?, 'queued', ?, ?)",
        (jid, kind, json.dumps(params) if params else None, datetime.now(UTC).isoformat()),
    )
    conn.commit()
    return jid


def update_job(conn: sqlite3.Connection, job_id: str, **fields: object) -> None:
    sets: list[str] = []
    params: list[object] = []
    for k, v in fields.items():
        sets.append(f"{k} = ?")
        if k in ("params_json", "result_json") and isinstance(v, (dict, list)):
            params.append(json.dumps(v))
        else:
            params.append(v)
    if not sets:
        return
    params.append(job_id)
    conn.execute(f"UPDATE jobs SET {', '.join(sets)} WHERE id = ?", tuple(params))
    conn.commit()


# Staged labels helpers
def insert_staged_label(
    conn: sqlite3.Connection,
    *,
    claim_a_id: str,
    claim_b_id: str,
    doc_a_id: str,
    doc_b_id: str,
    span_a_start: int,
    span_a_end: int,
    span_b_start: int,
    span_b_end: int,
    label: str,
    labeled_by: str,
    notes: str | None = None,
) -> str:
    sid = uuid.uuid4().hex
    conn.execute(
        "INSERT INTO staged_labels (id, claim_a_id, claim_b_id, doc_a_id, doc_b_id, span_a_start, span_a_end, span_b_start, span_b_end, label, notes, labeled_by, status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)",
        (sid, claim_a_id, claim_b_id, doc_a_id, doc_b_id, span_a_start, span_a_end, span_b_start, span_b_end, label, notes, labeled_by, datetime.now(UTC).isoformat()),
    )
    conn.commit()
    return sid


# Connections helpers
def list_connections(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(conn.execute("SELECT * FROM connections ORDER BY created_at DESC"))
