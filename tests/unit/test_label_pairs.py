"""Label-pair candidate queries + persistence (interactive loop not exercised)."""

from __future__ import annotations

import pytest

from adonis.db import (
    apply_migrations,
    get_conn,
    insert_candidate_pair,
    insert_claim,
    insert_document,
    insert_entity_mention,
    upsert_entity,
)
from adonis.ingest.base import DocumentRecord


def _make_claim(conn, doc_id: str, text: str, start: int, end: int, entity_id: str) -> str:
    claim_id = insert_claim(
        conn, document_id=doc_id, claim_text=text, span_start=start, span_end=end,
        topics=[], temporal=None, scope=None, triviality_score=0.1,
        extraction_model="fake", extraction_at="now",
    )
    insert_entity_mention(
        conn, claim_id=claim_id, entity_id=entity_id,
        mention_text=text.split()[0], span_start=start, span_end=start + len(text.split()[0]),
    )
    return claim_id


def _seed(conn) -> dict[str, str]:
    docs = {}
    for title, text in [("a", "a text"), ("b", "b text"), ("c", "c text")]:
        insert_document(
            conn,
            DocumentRecord(
                source="local", source_id=None, title=title, path=f"/{title}.md",
                format="md", raw_text=text, metadata={}, warnings=[],
            ),
        )
        docs[title] = conn.execute("SELECT id FROM documents WHERE title = ?", (title,)).fetchone()["id"]
    atlas = upsert_entity(conn, canonical_name="atlas", aliases=["Atlas"], mention_count=3)
    postgres = upsert_entity(conn, canonical_name="postgres", aliases=["Postgres"], mention_count=1)
    # a1/a2 share atlas within doc a; b1 shares atlas with a; c1 shares postgres only.
    a1 = _make_claim(conn, docs["a"], "Atlas ships in March.", 0, 20, atlas)
    _make_claim(conn, docs["a"], "Atlas is great.", 0, 15, atlas)
    b1 = _make_claim(conn, docs["b"], "Atlas ships in July.", 0, 20, atlas)
    c1 = _make_claim(conn, docs["c"], "Postgres for storage.", 0, 21, postgres)
    conn.commit()
    return {"a1": a1, "b1": b1, "c1": c1}


def test_fetch_pending_pairs_only_cross_document(tmp_env):
    apply_migrations()
    conn = get_conn()
    try:
        ids = _seed(conn)
        from adonis.cli.label_pairs import fetch_pending_pairs

        pairs = fetch_pending_pairs(conn, limit=50)
        # same-document pairs (both a-claims share atlas) must never appear
        assert all(p.a_doc != p.b_doc for p in pairs)
        # cross-document pairs sharing an entity do appear; c1 (postgres
        # only) shares nothing with the atlas claims, so it is excluded.
        assert any({ids["a1"], ids["b1"]} == {p.a_id, p.b_id} for p in pairs)
        assert not any(ids["c1"] in {p.a_id, p.b_id} for p in pairs)
    finally:
        conn.close()


def test_pool_near_dup_requires_candidate_pairs(tmp_env):
    apply_migrations()
    conn = get_conn()
    try:
        _seed(conn)
        from adonis.cli.label_pairs import fetch_pending_pairs

        # No candidate_pairs rows yet -> nothing to label from this pool.
        assert fetch_pending_pairs(conn, limit=50, pool="near_dup") == []
    finally:
        conn.close()


def test_pool_unrelated_excludes_shared_entity(tmp_env):
    apply_migrations()
    conn = get_conn()
    try:
        ids = _seed(conn)
        from adonis.cli.label_pairs import fetch_pending_pairs

        pairs = fetch_pending_pairs(conn, limit=50, pool="unrelated")
        pair_keys = {(p.a_id, p.b_id) for p in pairs}
        # a1-b1 share atlas, so excluded; a1-c1 and b1-c1 share nothing
        assert (ids["a1"], ids["b1"]) not in pair_keys
        assert (ids["b1"], ids["a1"]) not in pair_keys
        assert any({ids["a1"], ids["c1"]} == {p.a_id, p.b_id} for p in pairs)
        assert any({ids["b1"], ids["c1"]} == {p.a_id, p.b_id} for p in pairs)
    finally:
        conn.close()


def test_pool_near_dup_ranks_high_similarity(tmp_env):
    apply_migrations()
    conn = get_conn()
    try:
        ids = _seed(conn)
        insert_candidate_pair(
            conn, claim_a_id=ids["a1"], claim_b_id=ids["b1"],
            similarity_score=0.9, entity_overlap=1.0, combined_score=0.93,
            strategy="hybrid", selected_for_judge=True,
        )
        insert_candidate_pair(
            conn, claim_a_id=ids["a1"], claim_b_id=ids["c1"],
            similarity_score=0.5, entity_overlap=0.0, combined_score=0.35,
            strategy="embedding", selected_for_judge=False,
        )
        from adonis.cli.label_pairs import fetch_pending_pairs

        pairs = fetch_pending_pairs(conn, limit=50, pool="near_dup")
        assert len(pairs) == 1  # only similarity >= 0.85
        assert {pairs[0].a_id, pairs[0].b_id} == {ids["a1"], ids["b1"]}
    finally:
        conn.close()


def test_pool_near_dup_respects_existing_labels(tmp_env):
    apply_migrations()
    conn = get_conn()
    try:
        ids = _seed(conn)
        insert_candidate_pair(
            conn, claim_a_id=ids["a1"], claim_b_id=ids["b1"],
            similarity_score=0.9, entity_overlap=1.0, combined_score=0.93,
            strategy="hybrid", selected_for_judge=True,
        )
        from adonis.cli.label_pairs import fetch_pending_pairs, label_pair

        pair = fetch_pending_pairs(conn, limit=10, pool="near_dup")[0]
        label_pair(conn, pair, "true_negative_near_dup", labeled_by="test")
        assert fetch_pending_pairs(conn, limit=10, pool="near_dup") == []
    finally:
        conn.close()


def test_unknown_pool_raises(tmp_env):
    apply_migrations()
    conn = get_conn()
    try:
        from adonis.cli.label_pairs import fetch_pending_pairs

        with pytest.raises(ValueError, match="unknown pool"):
            fetch_pending_pairs(conn, pool="bogus")
    finally:
        conn.close()


def test_label_pair_persists(tmp_env):
    apply_migrations()
    conn = get_conn()
    try:
        insert_document(
            conn,
            DocumentRecord(
                source="local", source_id=None, title="a", path="/a.md",
                format="md", raw_text="a text", metadata={}, warnings=[],
            ),
        )
        insert_document(
            conn,
            DocumentRecord(
                source="local", source_id=None, title="b", path="/b.md",
                format="md", raw_text="b text", metadata={}, warnings=[],
            ),
        )
        docs = {r["title"]: r["id"] for r in conn.execute("SELECT title, id FROM documents")}
        atlas = upsert_entity(conn, canonical_name="atlas", aliases=["Atlas"], mention_count=2)
        a1 = _make_claim(conn, docs["a"], "Atlas ships in March.", 0, 20, atlas)
        b1 = _make_claim(conn, docs["b"], "Atlas ships in July.", 0, 20, atlas)

        from adonis.cli.label_pairs import fetch_pending_pairs, label_pair

        pair = fetch_pending_pairs(conn, limit=10)[0]
        assert {pair.a_id, pair.b_id} == {a1, b1}
        label_pair(conn, pair, "genuine_contradiction", labeled_by="test")
        assert fetch_pending_pairs(conn, limit=10) == []  # already labeled
        row = conn.execute("SELECT * FROM eval_labels").fetchone()
        assert row["label"] == "genuine_contradiction"
        assert row["labeled_by"] == "test"
    finally:
        conn.close()