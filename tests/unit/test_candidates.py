"""Candidate generation: top-K, hybrid scores, dedup rules, selection."""

from __future__ import annotations

import pytest

from adonis.pair.candidates import ClaimRow, build_candidate_rows, claim_rows_from_db
from tests.fakes import FakeEmbedder


def _claim(id_: str, doc: str, text: str, entities: list[str]) -> ClaimRow:
    return ClaimRow(id=id_, document_id=doc, claim_text=text, entities=entities)


def test_empty_claims_yield_no_candidates():
    rows, stats = build_candidate_rows([], FakeEmbedder())
    assert rows == [] and stats.candidates == 0


def test_intra_document_pairs_skipped():
    claims = [
        _claim("a1", "doc1", "Atlas ships in March.", ["Atlas"]),
        _claim("a2", "doc1", "The budget is 500.", ["budget"]),
        _claim("b1", "doc2", "Atlas ships in July.", ["Atlas"]),
    ]
    embedder = FakeEmbedder(
        dim=4,
        overrides={
            "Atlas ships in March.": [1, 0, 0, 0],
            "Atlas ships in July.": [0.9, 0.05, 0, 0],
            "The budget is 500.": [0, 1, 0, 0],
        },
    )
    rows, stats = build_candidate_rows(
        claims, embedder, top_k=5, entity_weight=0.3, selected_per_claim=3
    )
    pair_ids = {(r.claim_a_id, r.claim_b_id) for r in rows}
    assert ("a1", "a2") not in pair_ids  # same document
    assert stats.intra_doc_skipped > 0
    assert ("a1", "b1") in pair_ids  # cross-document


def test_hybrid_strategy_and_scores():
    claims = [
        _claim("a1", "doc1", "Atlas ships in March.", ["Atlas", "release"]),
        _claim("b1", "doc2", "Atlas ships in July.", ["Atlas", "release"]),
    ]
    embedder = FakeEmbedder(
        dim=3,
        overrides={
            "Atlas ships in March.": [1, 0, 0],
            "Atlas ships in July.": [1, 0, 0],
        },
    )
    rows, _ = build_candidate_rows(claims, embedder, top_k=5, entity_weight=0.3)
    assert len(rows) == 1
    pair = rows[0]
    assert pair.strategy == "hybrid"
    assert pair.entity_overlap == 1.0
    assert pair.similarity_score == pytest.approx(1.0)
    assert pair.combined_score == pytest.approx(1.0)
    assert pair.selected_for_judge


def test_entity_strategy_catches_pairs_missed_by_embeddings():
    # b1 shares an entity with a1 but its embedding is far away (cosine -0.1),
    # while six fillers crowd a1's top-K; the pair must still be produced.
    claims = [
        _claim("a1", "doc1", "Atlas ships in March.", ["Atlas"]),
        _claim("b1", "doc2", "Support team is great.", ["Atlas"]),
    ]
    claims += [
        _claim(f"f{i}", f"doc{i}", f"Filler claim number {i} here.", [])
        for i in range(6)
    ]
    overrides = {
        "Atlas ships in March.": [1.0, 0.0, 0.0],
        "Support team is great.": [-0.5, 0.8660254, 0.0],
    }
    for i in range(6):
        overrides[f"Filler claim number {i} here."] = [0.98 - 0.01 * i, 0.0, 0.1]
    embedder = FakeEmbedder(dim=3, overrides=overrides)
    rows, stats = build_candidate_rows(claims, embedder, top_k=3, entity_weight=0.3)
    pair = next(
        (r for r in rows if {r.claim_a_id, r.claim_b_id} == {"a1", "b1"}), None
    )
    assert pair is not None
    assert pair.strategy == "entity"
    assert pair.similarity_score == pytest.approx(-0.5)
    assert pair.entity_overlap == 1.0
    assert pair.combined_score == pytest.approx(0.7 * -0.5 + 0.3 * 1.0)
    assert stats.entity_pairs >= 1


def test_selection_caps_pairs_per_claim():
    claims = [_claim("a1", "doc1", f"claim {i}", []) for i in range(2)] + [
        _claim("b1", "doc2", f"other {i}", []) for i in range(2)
    ]
    embedder = FakeEmbedder(dim=4)
    rows, _ = build_candidate_rows(claims, embedder, top_k=20, selected_per_claim=1)
    selected = [r for r in rows if r.selected_for_judge]
    assert len(selected) <= 4
    seen_claims = set()
    for r in selected:
        seen_claims.update((r.claim_a_id, r.claim_b_id))
    assert len(seen_claims) <= 4


def test_claim_rows_from_db_parses_entities():
    rows = [
        {
            "id": "c1",
            "document_id": "d1",
            "claim_text": "Atlas ships.",
            "entities_json": '["e1", "e2"]',
        },
        {
            "id": "c2",
            "document_id": "d2",
            "claim_text": "Budget is 500.",
            "entities_json": None,
        },
    ]
    claims = claim_rows_from_db(rows)  # type: ignore[arg-type]
    assert [c.entities for c in claims] == [["e1", "e2"], []]
    assert claims[0].claim_text == "Atlas ships."