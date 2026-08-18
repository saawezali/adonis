"""End-to-end: scripts/extract_claims.py over a seeded store with fakes."""

from __future__ import annotations

from adonis.db import apply_migrations, get_conn, insert_document
from adonis.ingest.base import DocumentRecord
from tests.fakes import FakeLLMClient, FakeNER

DOC_A = "Atlas ships in March. Atlas costs 500 euros a month.\n"
DOC_B = "Atlas ships in July. The budget for Atlas is 800 euros.\n"


def _seed(tmp_env) -> None:
    apply_migrations()
    conn = get_conn()
    try:
        for title, text in [("roadmap", DOC_A), ("finance", DOC_B)]:
            insert_document(
                conn,
                DocumentRecord(
                    source="local", source_id=None, title=title, path=f"/{title}.md",
                    format="md", raw_text=text, metadata={}, warnings=[],
                ),
            )
    finally:
        conn.close()


CLAIMS_A = {"claims": [
    {"claim_text": "Atlas ships in March.", "span_start": 0, "span_end": 21,
     "triviality_score": 0.1, "topics": ["atlas", "release"],
     "temporal": None, "scope": None},
    {"claim_text": "Atlas costs 500 euros a month.", "span_start": 22, "span_end": 52,
     "triviality_score": 0.15, "topics": ["atlas", "cost"],
     "temporal": None, "scope": None},
]}
CLAIMS_B = {"claims": [
    {"claim_text": "Atlas ships in July.", "span_start": 0, "span_end": 20,
     "triviality_score": 0.1, "topics": ["atlas", "release"],
     "temporal": {"as_of": "2026-07-01"}, "scope": None},
]}

NER = FakeNER(
    by_text={
        "March": [
            {"text": "Atlas", "label": "PROJECT", "start": 0, "end": 5, "score": 0.9},
            {"text": "March", "label": "DATE", "start": 12, "end": 17, "score": 0.9},
        ],
        "500": [
            {"text": "Postgres", "label": "TECHNOLOGY", "start": 0, "end": 8, "score": 0.9},
        ],
        "July": [
            {"text": "Atlas", "label": "PROJECT", "start": 0, "end": 5, "score": 0.9},
            {"text": "July", "label": "DATE", "start": 12, "end": 16, "score": 0.9},
        ],
    }
)


def test_extract_script_end_to_end(tmp_env, monkeypatch):
    _seed(tmp_env)
    client = FakeLLMClient(by_text={DOC_A: CLAIMS_A, DOC_B: CLAIMS_B})
    monkeypatch.setattr(
        "adonis.extract.entities.load_model", lambda force=False: NER
    )

    from scripts.extract_claims import run

    conn = get_conn()
    try:
        stats = run(conn, client=client)

        assert stats.docs_seen == 2
        assert stats.docs_failed == 0
        assert stats.claims_inserted == 3
        assert stats.llm_calls == 2  # one call per doc (each is one chunk)
        assert stats.entities == 5  # clusters processed: 2 + 1 + 2 per claim
        assert stats.mentions == 5

        docs = {r["id"]: r for r in conn.execute("SELECT * FROM documents")}
        assert len(docs) == 2
        claims = conn.execute("SELECT * FROM claims ORDER BY document_id").fetchall()
        assert len(claims) == 3
        for claim in claims:
            raw = docs[claim["document_id"]]["raw_text"]
            assert raw[claim["citation_span_start"] : claim["citation_span_end"]].strip()

        entities = conn.execute("SELECT * FROM entities").fetchall()
        assert len(entities) == 4  # Atlas, March, Postgres, July (Atlas deduped)
        assert any(e["canonical_name"] == "Atlas" for e in entities)
        mentions = conn.execute("SELECT * FROM entity_mentions").fetchall()
        atlas_claims = {m["claim_id"] for m in mentions if m["mention_text"] == "Atlas"}
        assert len(atlas_claims) == 2  # one in each document
        calls = conn.execute("SELECT * FROM llm_calls").fetchall()
        assert len(calls) == 2
        assert all(c["stage"] == "extract" for c in calls)
        assert all(c["prompt_version"] == "claims_v1" for c in calls)
    finally:
        conn.close()


def test_extract_script_survives_one_bad_document(tmp_env, monkeypatch):
    _seed(tmp_env)
    monkeypatch.setattr(
        "adonis.extract.entities.load_model", lambda force=False: NER
    )

    class BoomClient(FakeLLMClient):
        def complete_json(self, system: str, user: str, **kwargs: object) -> dict[str, object]:
            if "July" in user:
                raise RuntimeError("api down")
            return {
                "claims": [
                    {
                        "claim_text": "Atlas ships in March.",
                        "span_start": 0,
                        "span_end": 21,
                        "triviality_score": 0.1,
                        "topics": ["atlas"],
                        "temporal": None,
                        "scope": None,
                    }
                ]
            }

    from scripts.extract_claims import run

    conn = get_conn()
    try:
        stats = run(conn, client=BoomClient())
        assert stats.docs_seen == 2
        assert stats.docs_failed == 0  # LLM failures are chunk-level, not doc-level
        assert stats.claims_inserted == 1
        assert len(stats.errors) == 1
        assert conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0] == 1
        calls = conn.execute(
            "SELECT success, error FROM llm_calls"
        ).fetchall()
        assert len(calls) == 3  # 2 docs + 1 failed chunk
        assert sum(c["success"] == 0 for c in calls) == 1
    finally:
        conn.close()


def test_extract_script_limit(tmp_env, monkeypatch):
    _seed(tmp_env)
    monkeypatch.setattr(
        "adonis.extract.entities.load_model", lambda force=False: NER
    )

    from scripts.extract_claims import run

    conn = get_conn()
    try:
        stats = run(conn, client=FakeLLMClient(), limit=1)
        assert stats.docs_seen == 1
        assert stats.claims_inserted == 0
    finally:
        conn.close()