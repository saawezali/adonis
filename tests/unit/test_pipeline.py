"""End-to-end M4: claims -> candidates -> judge -> verify -> flags with fakes."""

from __future__ import annotations

from adonis.db import (
    apply_migrations,
    get_conn,
    insert_claim,
    insert_document,
    insert_entity_mention,
    upsert_entity,
)
from adonis.ingest.base import DocumentRecord
from tests.fakes import FakeEmbedder


def _seed(tmp_env) -> None:
    apply_migrations()
    conn = get_conn()
    try:
        for title, text in [
            ("roadmap", "Atlas ships in March. The budget is 500 euros.\n"),
            ("finance", "Atlas ships in July. The budget is 800 euros.\n"),
            ("release", "Atlas ships in July.\n"),
        ]:
            insert_document(
                conn,
                DocumentRecord(
                    source="local", source_id=None, title=title, path=f"/{title}.md",
                    format="md", raw_text=text, metadata={}, warnings=[],
                ),
            )
        docs = {r["title"]: r["id"] for r in conn.execute("SELECT title, id FROM documents")}

        def claim(doc_id: str, text: str, start: int, end: int, entity: str) -> str:
            claim_id = insert_claim(
                conn, document_id=doc_id, claim_text=text, span_start=start, span_end=end,
                topics=[], temporal=None, scope=None, triviality_score=0.1,
                extraction_model="fake", extraction_at="now",
            )
            entity_id = upsert_entity(conn, canonical_name=entity, aliases=[entity], mention_count=1)
            insert_entity_mention(
                conn, claim_id=claim_id, entity_id=entity_id,
                mention_text=entity, span_start=start, span_end=start + len(entity),
            )
            return claim_id

        claim(docs["roadmap"], "Atlas ships in March.", 0, 21, "Atlas")
        claim(docs["roadmap"], "The budget is 500 euros.", 22, 45, "budget")
        claim(docs["finance"], "Atlas ships in July.", 0, 20, "Atlas")
        claim(docs["finance"], "The budget is 800 euros.", 21, 44, "budget")
        claim(docs["release"], "Atlas ships in July.", 0, 20, "Atlas")
        conn.commit()
    finally:
        conn.close()


def test_pipeline_end_to_end(tmp_env):
    _seed(tmp_env)
    conn = get_conn()
    try:
        from scripts.run_pipeline import DemoJudge, DemoVerifier, run

        embedder = FakeEmbedder(
            dim=4,
            overrides={
                "Atlas ships in March.": [1, 0, 0, 0],
                "Atlas ships in July.": [0.9, 0.1, 0, 0],
                "The budget is 500 euros.": [0, 1, 0, 0],
                "The budget is 800 euros.": [0, 0.95, 0.05, 0],
            },
        )
        stats = run(  # type: ignore[arg-type]
            conn,
            client=DemoJudge(),  # type: ignore[arg-type]
            embedder=embedder,
            verifier=DemoVerifier(),  # type: ignore[arg-type]
        )

        assert stats.claims == 5
        assert stats.candidates >= 2
        assert stats.new_candidates == stats.candidates
        assert stats.pairs_judged >= 2
        assert stats.judge_failures == 0
        assert stats.flags >= 2  # March vs July is a genuine_contradiction
        assert stats.verified >= 2
        assert stats.verification_failures == 0
        assert stats.citation_faithfulness == 1.0

        candidates = conn.execute("SELECT COUNT(*) FROM candidate_pairs").fetchone()[0]
        assert candidates == stats.candidates
        judged = conn.execute("SELECT COUNT(*) FROM judge_outputs").fetchone()[0]
        assert judged == stats.pairs_judged
        calls = conn.execute("SELECT stage, success FROM llm_calls").fetchall()
        assert all(c["success"] == 1 for c in calls)
        assert sum(1 for c in calls if c["stage"] == "judge") == judged
        assert sum(1 for c in calls if c["stage"] == "verify") == stats.verified * 2

        # Every flag-label judge output has a verification row.
        verified_rows = conn.execute(
            "SELECT vr.overall_pass, jo.label FROM verification_results vr"
            " JOIN judge_outputs jo ON jo.id = vr.judge_output_id"
        ).fetchall()
        assert len(verified_rows) == stats.verified
        assert all(r["overall_pass"] == 1 for r in verified_rows)

        # Every flag has overall_pass = 1 (acceptance criterion).
        flags = conn.execute(
            "SELECT f.candidate_pair_id, vr.overall_pass FROM flags f"
            " JOIN judge_outputs jo ON jo.candidate_pair_id = f.candidate_pair_id"
            " JOIN verification_results vr ON vr.judge_output_id = jo.id"
        ).fetchall()
        assert len(flags) == stats.flags
        assert all(f["overall_pass"] == 1 for f in flags)

        # Every stored cited span resolves to non-blank text in its document.
        docs = {r["id"]: r for r in conn.execute("SELECT * FROM documents")}
        for jo in conn.execute("SELECT * FROM judge_outputs"):
            pair = conn.execute(
                "SELECT * FROM candidate_pairs WHERE id = ?", (jo["candidate_pair_id"],)
            ).fetchone()
            c1 = conn.execute("SELECT * FROM claims WHERE id = ?", (pair["claim_a_id"],)).fetchone()
            c2 = conn.execute("SELECT * FROM claims WHERE id = ?", (pair["claim_b_id"],)).fetchone()
            assert docs[c1["document_id"]]["raw_text"][jo["cited_span_a_start"] : jo["cited_span_a_end"]].strip()
            assert docs[c2["document_id"]]["raw_text"][jo["cited_span_b_start"] : jo["cited_span_b_end"]].strip()
    finally:
        conn.close()


def test_pipeline_no_claims_is_a_noop(tmp_env):
    apply_migrations()
    conn = get_conn()
    try:
        from scripts.run_pipeline import DemoJudge, DemoVerifier, run

        stats = run(  # type: ignore[arg-type]
            conn,
            client=DemoJudge(),  # type: ignore[arg-type]
            embedder=FakeEmbedder(),
            verifier=DemoVerifier(),  # type: ignore[arg-type]
        )
        assert stats.claims == 0
        assert stats.candidates == 0
        assert stats.pairs_judged == 0
        assert stats.citation_faithfulness is None
        assert any("no claims" in e for e in stats.errors)
    finally:
        conn.close()


def test_pipeline_rerun_does_not_duplicate_candidates(tmp_env):
    _seed(tmp_env)
    conn = get_conn()
    try:
        from scripts.run_pipeline import DemoJudge, DemoVerifier, run

        kwargs = {  # type: ignore[arg-type]
            "client": DemoJudge(),
            "embedder": FakeEmbedder(dim=4),
            "verifier": DemoVerifier(),
        }
        first = run(conn, **kwargs)
        second = run(conn, **kwargs)
        assert first.new_candidates == first.candidates
        assert second.new_candidates == 0  # all pairs already materialized
        assert second.pairs_judged == 0  # all selected pairs already judged
    finally:
        conn.close()