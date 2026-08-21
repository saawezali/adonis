"""HTML report rendering: flags in, file out."""

from __future__ import annotations

from adonis.db import (
    apply_migrations,
    get_conn,
    insert_candidate_pair,
    insert_document,
    insert_flag,
    insert_judge_output,
    insert_verification_result,
    load_claims,
)
from adonis.ingest.base import DocumentRecord
from adonis.report.render import _snippet, render_report


def _seed(tmp_env) -> None:
    apply_migrations()
    conn = get_conn()
    try:
        insert_document(
            conn,
            DocumentRecord(
                source="local", source_id=None, title="roadmap", path="/docs/roadmap.md",
                format="md", raw_text="Atlas ships in March.\n", metadata={}, warnings=[],
            ),
        )
        insert_document(
            conn,
            DocumentRecord(
                source="local", source_id=None, title="finance", path="/docs/finance.md",
                format="md", raw_text="Atlas ships in July.\n", metadata={}, warnings=[],
            ),
        )
        conn.commit()
        docs = list(conn.execute("SELECT id, title FROM documents"))
        a = docs[0]["id"]
        b = docs[1]["id"]

        def claim(document_id: str, text: str, start: int, end: int) -> str:
            from adonis.db import insert_claim

            return insert_claim(
                conn, document_id=document_id, claim_text=text, span_start=start, span_end=end,
                topics=[], temporal=None, scope=None, triviality_score=0.1,
                extraction_model="fake", extraction_at="now",
            )

        c_a = claim(a, "Atlas ships in March.", 0, 21)
        c_b = claim(b, "Atlas ships in July.", 0, 20)
        conn.commit()

        insert_candidate_pair(
            conn, claim_a_id=c_a, claim_b_id=c_b,
            similarity_score=0.9, entity_overlap=1.0, combined_score=0.93,
            strategy="hybrid", selected_for_judge=True,
        )
        # Ordered storage: handle either order
        lo, hi = (c_a, c_b) if c_a < c_b else (c_b, c_a)
        pair_id = conn.execute(
            "SELECT id FROM candidate_pairs WHERE claim_a_id = ? AND claim_b_id = ?",
            (lo, hi),
        ).fetchone()["id"]
        judge_id = insert_judge_output(
            conn, candidate_pair_id=pair_id, label="genuine_contradiction",
            judge_confidence=0.9, reasoning_text="dates conflict",
            cited_span_a_start=0, cited_span_a_end=21,
            cited_span_b_start=0, cited_span_b_end=20,
            judge_model="demo-judge", prompt_version="judge_v1",
        )
        insert_verification_result(
            conn, judge_output_id=judge_id,
            span_a_verbatim=True, span_a_fuzzy=1.0, span_a_entailment=0.9, span_a_pass=True,
            span_b_verbatim=True, span_b_fuzzy=1.0, span_b_entailment=0.9, span_b_pass=True,
        )
        insert_flag(
            conn, candidate_pair_id=pair_id,
            final_label="genuine_contradiction", final_confidence=0.85,
        )
        conn.commit()
    finally:
        conn.close()


def test_snippet_adds_ellipses():
    text = "x" * 400
    snip = _snippet(text, 200, 205)
    assert snip.startswith("\u2026") and snip.endswith("\u2026")
    assert len(snip) < 400


def test_snippet_no_ellipses_for_short_doc():
    text = "Atlas ships in March."
    snip = _snippet(text, 0, 21)
    assert snip == text


def test_render_report_writes_html(tmp_env, tmp_path):
    _seed(tmp_env)
    conn = get_conn()
    try:
        out = render_report(conn, tmp_path / "report.html")
    finally:
        conn.close()
    html = out.read_text(encoding="utf-8")
    assert "genuine_contradiction" in html
    assert "Atlas ships in March." in html
    assert "Atlas ships in July." in html
    assert "file://" in html  # source links
    assert "citation faithfulness" in html
    assert "confidence 0.85" in html


def test_render_report_empty_store(tmp_env, tmp_path):
    apply_migrations()
    conn = get_conn()
    try:
        out = render_report(conn, tmp_path / "report.html")
    finally:
        conn.close()
    html = out.read_text(encoding="utf-8")
    assert "No flags yet" in html


def test_load_claims_does_not_break_report(tmp_env):
    apply_migrations()
    conn = get_conn()
    try:
        assert load_claims(conn) == []
    finally:
        conn.close()