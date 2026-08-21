"""Candidate pair generation (PLAN.md M3).

Strategy per claim: top-K embedding neighbors (hybrid when the pair also
shares an entity, else 'embedding'), plus every cross-document pair sharing
a canonical entity not already covered by the embedding top-K ('entity'
strategy). Intra-document pairs are never candidates, and symmetric pairs
are collapsed keeping the higher combined score.

combined = (1 - w) * similarity + w * entity_overlap, w = candidate_entity_weight.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from typing import TypeAlias

import numpy as np

from adonis.config import get_settings
from adonis.pair.embed import Embedder
from adonis.pair.index import build_index, search


@dataclass(frozen=True)
class ClaimRow:
    """The slice of a claim row the candidate stage needs."""

    id: str
    document_id: str
    claim_text: str
    entities: list[str]


@dataclass(frozen=True)
class CandidatePairRow:
    claim_a_id: str
    claim_b_id: str
    similarity_score: float
    entity_overlap: float
    combined_score: float
    strategy: str
    selected_for_judge: bool = False


@dataclass
class CandidateStats:
    candidates: int = 0
    embedding_pairs: int = 0
    entity_pairs: int = 0
    intra_doc_skipped: int = 0
    duplicate_skipped: int = 0
    errors: list[str] = field(default_factory=list)


DBRow: TypeAlias = sqlite3.Row


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def _entity_overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _claims_shared_entities(claims: list[ClaimRow]) -> dict[str, list[int]]:
    """entity -> list of claim index positions mentioning it."""
    shared: dict[str, list[int]] = {}
    for i, claim in enumerate(claims):
        for entity in claim.entities:
            shared.setdefault(entity, []).append(i)
    return shared


def build_candidate_rows(
    claims: list[ClaimRow],
    embedder: Embedder,
    *,
    top_k: int | None = None,
    entity_weight: float | None = None,
    selected_per_claim: int | None = None,
) -> tuple[list[CandidatePairRow], CandidateStats]:
    """Generate cross-document candidate pairs with hybrid scores.

    Returns (rows sorted by combined score desc, stats).
    """
    top_k = top_k if top_k is not None else get_settings().top_k
    entity_weight = (
        entity_weight
        if entity_weight is not None
        else get_settings().candidate_entity_weight
    )
    selected_per_claim = (
        selected_per_claim
        if selected_per_claim is not None
        else get_settings().judge_per_claim
    )
    stats = CandidateStats()

    if not claims:
        return [], stats

    texts = [c.claim_text for c in claims]
    vectors = np.asarray(embedder.encode(texts, normalize_embeddings=True), dtype=np.float32)
    if vectors.ndim != 2 or len(vectors) != len(claims):
        raise ValueError("embedder output shape mismatch")
    index = build_index(vectors)

    entity_sets = [set(c.entities) for c in claims]
    doc_of = {i: c.document_id for i, c in enumerate(claims)}
    best: dict[tuple[str, str], CandidatePairRow] = {}

    def keep(pair: CandidatePairRow) -> None:
        key = _pair_key(pair.claim_a_id, pair.claim_b_id)
        existing = best.get(key)
        if existing is None or pair.combined_score > existing.combined_score:
            best[key] = pair
        else:
            stats.duplicate_skipped += 1

    def make_pair(i: int, j: int, similarity: float, strategy: str) -> None:
        if doc_of[i] == doc_of[j]:
            stats.intra_doc_skipped += 1
            return
        overlap = _entity_overlap(entity_sets[i], entity_sets[j])
        combined = (1 - entity_weight) * similarity + entity_weight * overlap
        keep(
            CandidatePairRow(
                claim_a_id=claims[i].id,
                claim_b_id=claims[j].id,
                similarity_score=float(similarity),
                entity_overlap=float(overlap),
                combined_score=float(combined),
                strategy=strategy,
            )
        )

    # 1) Top-K embedding neighbors per claim.
    distances, indices = search(index, vectors, top_k + 1)
    for i in range(len(claims)):
        for j, dist in zip(indices[i].tolist(), distances[i].tolist()):
            if j == -1 or j == i:
                continue
            overlap = _entity_overlap(entity_sets[i], entity_sets[j])
            strategy = "hybrid" if overlap > 0 else "embedding"
            make_pair(i, int(j), float(dist), strategy)

    # 2) Entity-strategy pairs beyond the embedding top-K. Unlike the
    # top-K pass (which has FAISS distances handy), compute cosine directly
    # from the L2-normalized vectors.
    covered: set[tuple[int, int]] = set()
    for i in range(len(claims)):
        row = indices[i].tolist()
        covered.update((i, j) if i < j else (j, i) for j in row if j not in (-1, i))
    for positions in _claims_shared_entities(claims).values():
        for idx, i in enumerate(positions):
            for j in positions[idx + 1 :]:
                if (i, j) not in covered and (j, i) not in covered:
                    similarity = float(vectors[i] @ vectors[j])
                    make_pair(i, j, similarity, "entity")

    rows = sorted(best.values(), key=lambda p: -p.combined_score)
    stats.candidates = len(rows)
    stats.embedding_pairs = sum(1 for p in rows if p.strategy == "embedding")
    stats.entity_pairs = sum(1 for p in rows if p.strategy == "entity")

    # 3) Selection for the judge: top N per claim by combined score.
    per_claim: dict[str, list[CandidatePairRow]] = {}
    for pair in rows:
        per_claim.setdefault(pair.claim_a_id, []).append(pair)
        per_claim.setdefault(pair.claim_b_id, []).append(pair)
    selected: set[tuple[str, str]] = set()
    for pairs_for_claim in per_claim.values():
        ordered = sorted(pairs_for_claim, key=lambda p: -p.combined_score)
        for pair in ordered[:selected_per_claim]:
            selected.add(_pair_key(pair.claim_a_id, pair.claim_b_id))
    final: list[CandidatePairRow] = []
    for pair in rows:
        chosen = _pair_key(pair.claim_a_id, pair.claim_b_id) in selected
        final.append(
            CandidatePairRow(
                claim_a_id=pair.claim_a_id,
                claim_b_id=pair.claim_b_id,
                similarity_score=pair.similarity_score,
                entity_overlap=pair.entity_overlap,
                combined_score=pair.combined_score,
                strategy=pair.strategy,
                selected_for_judge=chosen,
            )
        )
    return final, stats


def claim_rows_from_db(rows: list[DBRow]) -> list[ClaimRow]:
    """Convert sqlite rows (id, document_id, claim_text, entities_json) to ClaimRow."""
    claims: list[ClaimRow] = []
    for row in rows:
        raw = row["entities_json"]
        entities = json.loads(raw) if raw else []
        claims.append(
            ClaimRow(
                id=row["id"],
                document_id=row["document_id"],
                claim_text=row["claim_text"],
                entities=[e for e in entities if isinstance(e, str)],
            )
        )
    return claims
