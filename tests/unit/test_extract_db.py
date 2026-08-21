"""DB helpers for M2: claims, entities, mentions, llm_calls, eval labels."""

from __future__ import annotations

import json

from adonis.db import (
    apply_migrations,
    get_conn,
    insert_claim,
    insert_document,
    insert_entity_mention,
    insert_eval_label,
    insert_llm_call,
    iter_documents,
    update_claim_entities,
    upsert_entity,
)
from adonis.ingest.base import DocumentRecord


def test_iter_documents_and_claim_roundtrip(tmp_env):
    apply_migrations()
    conn = get_conn()
    try:
        rec = DocumentRecord(
            source="local",
            source_id=None,
            title="t",
            path="/x.md",
            format="md",
            raw_text="Atlas ships in March.",
            metadata={},
            warnings=[],
        )
        assert insert_document(conn, rec)
        docs = iter_documents(conn)
        assert len(docs) == 1
        doc = docs[0]

        claim_id = insert_claim(
            conn,
            document_id=doc["id"],
            claim_text="Atlas ships in March.",
            span_start=0,
            span_end=20,
            topics=["atlas", "release"],
            temporal={"as_of": "2026-03-01"},
            scope=None,
            triviality_score=0.1,
            extraction_model="fake",
            extraction_at="2026-01-01T00:00:00+00:00",
        )
        row = conn.execute("SELECT * FROM claims WHERE id = ?", (claim_id,)).fetchone()
        assert row["claim_text"] == "Atlas ships in March."
        assert row["entities_json"] == "[]"
        assert json.loads(row["topics_json"]) == ["atlas", "release"]
        assert json.loads(row["temporal_json"]) == {"as_of": "2026-03-01"}
        assert row["scope_json"] is None
        assert row["triviality_score"] == 0.1
    finally:
        conn.close()


def test_iter_documents_limit_and_filter(tmp_env):
    apply_migrations()
    conn = get_conn()
    try:
        for title, text in [("a", "doc a here."), ("b", "doc b here."), ("c", "doc c here.")]:
            insert_document(
                conn,
                DocumentRecord(
                    source="local", source_id=None, title=title, path=f"/{title}.md",
                    format="md", raw_text=text, metadata={}, warnings=[],
                ),
            )
        assert len(iter_documents(conn, limit=2)) == 2
        only = iter_documents(conn, limit=1)
        assert len(only) == 1
        assert len(iter_documents(conn, document_id=only[0]["id"])) == 1
        assert len(iter_documents(conn, document_id="nope")) == 0
    finally:
        conn.close()


def test_upsert_entity_unions_aliases_and_counts(tmp_env):
    apply_migrations()
    conn = get_conn()
    try:
        eid1 = upsert_entity(conn, canonical_name="atlas", aliases=["Atlas"], mention_count=1)
        eid2 = upsert_entity(conn, canonical_name="atlas", aliases=["atlas"], mention_count=2)
        assert eid1 == eid2
        row = conn.execute("SELECT * FROM entities WHERE id = ?", (eid1,)).fetchone()
        assert json.loads(row["aliases_json"]) == ["Atlas", "atlas"]
        assert row["mention_count"] == 3
    finally:
        conn.close()


def test_mentions_and_claim_entities_link(tmp_env):
    apply_migrations()
    conn = get_conn()
    try:
        insert_document(
            conn,
            DocumentRecord(
                source="local", source_id=None, title="t", path="/t.md",
                format="md", raw_text="Atlas ships in March.", metadata={}, warnings=[],
            ),
        )
        doc = iter_documents(conn)[0]
        claim_id = insert_claim(
            conn, document_id=doc["id"], claim_text="Atlas ships in March.",
            span_start=0, span_end=20, topics=[], temporal=None, scope=None,
            triviality_score=0.1, extraction_model="fake", extraction_at="now",
        )
        entity_id = upsert_entity(conn, canonical_name="atlas", aliases=["Atlas"], mention_count=1)
        insert_entity_mention(
            conn, claim_id=claim_id, entity_id=entity_id,
            mention_text="Atlas", span_start=0, span_end=5,
        )
        update_claim_entities(conn, claim_id, [entity_id])

        claim = conn.execute("SELECT * FROM claims WHERE id = ?", (claim_id,)).fetchone()
        assert json.loads(claim["entities_json"]) == [entity_id]
        em = conn.execute("SELECT * FROM entity_mentions").fetchall()
        assert len(em) == 1
        assert em[0]["claim_id"] == claim_id
        assert em[0]["entity_id"] == entity_id
        assert em[0]["span_start"] == 0 and em[0]["span_end"] == 5
    finally:
        conn.close()


def test_llm_call_and_eval_label_inserts(tmp_env):
    apply_migrations()
    conn = get_conn()
    try:
        insert_llm_call(
            conn, stage="extract", model="fake", prompt_version="claims_v1",
            latency_ms=42, success=True,
        )
        insert_llm_call(
            conn, stage="extract", model="fake", prompt_version="claims_v1",
            success=False, error="boom",
        )
        calls = conn.execute("SELECT * FROM llm_calls").fetchall()
        assert len(calls) == 2
        assert calls[0]["success"] == 1 and calls[0]["latency_ms"] == 42
        assert calls[1]["success"] == 0 and calls[1]["error"] == "boom"

        doc_a_id = insert_document(
            conn,
            DocumentRecord(
                source="local", source_id=None, title="a", path="/a.md",
                format="md", raw_text="a text here", metadata={}, warnings=[],
            ),
        )
        doc_b_id = insert_document(
            conn,
            DocumentRecord(
                source="local", source_id=None, title="b", path="/b.md",
                format="md", raw_text="b text here", metadata={}, warnings=[],
            ),
        )
        from adonis.db import iter_documents

        ids = {r["title"]: r["id"] for r in iter_documents(conn)}
        assert set(ids) == {"a", "b"}
        claim_a_id = insert_claim(
            conn, document_id=ids["a"], claim_text="a text here",
            span_start=0, span_end=11, topics=[], temporal=None, scope=None,
            triviality_score=0.1, extraction_model="fake", extraction_at="now",
        )
        claim_b_id = insert_claim(
            conn, document_id=ids["b"], claim_text="b text here",
            span_start=0, span_end=11, topics=[], temporal=None, scope=None,
            triviality_score=0.1, extraction_model="fake", extraction_at="now",
        )
        assert doc_a_id is True and doc_b_id is True  # insert_document returns bool
        insert_eval_label(
            conn, claim_a_id=claim_a_id, claim_b_id=claim_b_id,
            doc_a_id=ids["a"], doc_b_id=ids["b"],
            span_a_start=0, span_a_end=5, span_b_start=1, span_b_end=6,
            label="genuine_contradiction", labeled_by="test",
        )
        row = conn.execute("SELECT * FROM eval_labels").fetchone()
        assert row["label"] == "genuine_contradiction"
        assert row["used_in_eval"] == 0
    finally:
        conn.close()