"""Harness over a seeded store: labels + judge outputs -> eval report."""

from __future__ import annotations

from adonis.db import (
    apply_migrations,
    get_conn,
    insert_candidate_pair,
    insert_claim,
    insert_document,
    insert_eval_label,
    insert_judge_output,
    insert_verification_result,
)
from adonis.eval.harness import eval_json, labeled_pair_count, run_eval
from adonis.ingest.base import DocumentRecord


def _seed(tmp_env) -> None:
    apply_migrations()
    conn = get_conn()
    try:
        for title, text in [
            ("roadmap", "Atlas ships in March.\n"),
            ("finance", "Atlas ships in July.\n"),
            ("storage", "We run Postgres in production.\n"),
        ]:
            insert_document(
                conn,
                DocumentRecord(
                    source="local", source_id=None, title=title, path=f"/{title}.md",
                    format="md", raw_text=text, metadata={}, warnings=[],
                ),
            )
        docs = {r["title"]: r["id"] for r in conn.execute("SELECT title, id FROM documents")}

        def claim(doc: str, text: str) -> str:
            return insert_claim(
                conn, document_id=docs[doc], claim_text=text, span_start=0, span_end=len(text),
                topics=[], temporal=None, scope=None, triviality_score=0.1,
                extraction_model="fake", extraction_at="now",
            )

        c_march = claim("roadmap", "Atlas ships in March.")
        c_july = claim("finance", "Atlas ships in July.")
        c_postgres = claim("storage", "We run Postgres in production.")
        conn.commit()

        # One judged contradiction (verified), one labeled-but-unjudged pair
        # (exercises the missed-contradiction path), one unjudged TN pair.
        insert_candidate_pair(
            conn, claim_a_id=c_march, claim_b_id=c_july,
            similarity_score=0.9, entity_overlap=1.0, combined_score=0.93,
            strategy="hybrid", selected_for_judge=True,
        )
        pair_id = conn.execute("SELECT id FROM candidate_pairs").fetchone()["id"]

        insert_eval_label(
            conn, claim_a_id=c_march, claim_b_id=c_july,
            doc_a_id=docs["roadmap"], doc_b_id=docs["finance"],
            span_a_start=0, span_a_end=21, span_b_start=0, span_b_end=20,
            label="genuine_contradiction", labeled_by="test",
        )
        judge_id = insert_judge_output(
            conn, candidate_pair_id=pair_id, label="genuine_contradiction",
            judge_confidence=0.9, reasoning_text="dates conflict",
            cited_span_a_start=0, cited_span_a_end=21,
            cited_span_b_start=0, cited_span_b_end=20,
            judge_model="demo-judge", prompt_version="judge_v1",
        )
        insert_verification_result(
            conn, judge_output_id=judge_id,
            span_a_verbatim=True, span_a_fuzzy=1.0, span_a_entailment=0.95, span_a_pass=True,
            span_b_verbatim=True, span_b_fuzzy=1.0, span_b_entailment=0.95, span_b_pass=True,
        )

        insert_eval_label(
            conn, claim_a_id=c_march, claim_b_id=c_postgres,
            doc_a_id=docs["roadmap"], doc_b_id=docs["storage"],
            span_a_start=0, span_a_end=21, span_b_start=0, span_b_end=35,
            label="true_negative_unrelated", labeled_by="test",
        )
        # Contradiction pair that was never judged -> missed.
        insert_candidate_pair(
            conn, claim_a_id=c_july, claim_b_id=c_postgres,
            similarity_score=0.1, entity_overlap=0.0, combined_score=0.07,
            strategy="embedding", selected_for_judge=False,
        )
        insert_eval_label(
            conn, claim_a_id=c_july, claim_b_id=c_postgres,
            doc_a_id=docs["finance"], doc_b_id=docs["storage"],
            span_a_start=0, span_a_end=20, span_b_start=0, span_b_end=35,
            label="genuine_contradiction", labeled_by="test",
        )
        conn.commit()
    finally:
        conn.close()


def test_run_eval_counts(tmp_env):
    _seed(tmp_env)
    conn = get_conn()
    try:
        report = run_eval(conn)
    finally:
        conn.close()

    assert report.n_labeled == 3
    assert report.n_judged == 1
    assert report.n_verified == 1
    assert report.metrics.citation_faithfulness == 1.0
    assert report.metrics.detection_recall == 0.5  # 1 of 2 contradictions judged
    assert report.metrics.detection_precision == 1.0

    by = {c.category: c for c in report.metrics.categories}
    gc = by["genuine_contradiction"]
    assert (gc.tp, gc.fp, gc.fn) == (1, 0, 1)  # missed pair counts as FN
    assert gc.recall == 0.5
    assert gc.precision == 1.0
    tn = by["true_negative_unrelated"]
    assert (tn.tp, tn.fp, tn.fn) == (1, 0, 0)  # unjudged + unlabeled TN is a tp

    # Metrics categories cover the full taxonomy incl. true negatives.
    assert set(by) == {
        "genuine_contradiction", "superseded_by_time", "different_scope", "ambiguous",
        "true_negative_near_dup", "true_negative_unrelated",
    }


def test_eval_json_serializable(tmp_env):
    _seed(tmp_env)
    conn = get_conn()
    try:
        report = run_eval(conn)
        data = report.as_dict()
    finally:
        conn.close()
    assert data["n_labeled"] == 3
    assert data["citation_faithfulness"] == 1.0
    assert "genuine_contradiction" in data["categories"]
    assert data["categories"]["genuine_contradiction"]["recall"] == 0.5
    assert len(eval_json(report)) > 0


def test_labeled_pair_count(tmp_env):
    _seed(tmp_env)
    conn = get_conn()
    try:
        assert labeled_pair_count(conn) == 3
    finally:
        conn.close()