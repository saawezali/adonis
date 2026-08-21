"""Interactive claim-pair labeling CLI (PLAN.md M2..M5).

Three label pools, chosen with --pool:
  entity    cross-document pairs sharing a canonical entity (contradiction
            candidates),
  near_dup  candidate_pairs with high embedding similarity (potential
            true_negative_near_dup),
  unrelated cross-document pairs with no shared entity (potential
            true_negative_unrelated).

Labels go into eval_labels, matching the taxonomy in PLAN.md section 4.
Run: python -m adonis.cli.label_pairs [--pool near_dup|unrelated|entity]
"""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import dataclass

from adonis.db import apply_migrations, get_conn, insert_eval_label

LABEL_KEYS: dict[str, str] = {
    "c": "genuine_contradiction",
    "t": "superseded_by_time",
    "s": "different_scope",
    "a": "ambiguous",
    "n": "not_conflicting",
    "d": "true_negative_near_dup",
    "u": "true_negative_unrelated",
}

_LABEL_HINT = (
    "c=contradiction  t=superseded_by_time  s=different_scope  a=ambiguous\n"
    "n=not_conflicting  d=near_dup  u=unrelated  k=skip  q=quit\n"
)

POOLS = ("entity", "near_dup", "unrelated")


@dataclass(frozen=True)
class PendingPair:
    """A candidate pair, ready to label."""

    a_id: str
    b_id: str
    a_doc: str
    b_doc: str
    a_text: str
    b_text: str
    a_doc_id: str
    b_doc_id: str
    a_start: int
    a_end: int
    b_start: int
    b_end: int


def fetch_pending_pairs(
    conn: sqlite3.Connection, limit: int = 20, pool: str = "entity"
) -> list[PendingPair]:
    """Unlabeled claim pairs from the given pool (see module docstring)."""
    if pool == "entity":
        sql = """
            SELECT c1.id AS a_id, c2.id AS b_id,
                   d1.title AS a_doc, d2.title AS b_doc,
                   c1.claim_text AS a_text, c2.claim_text AS b_text,
                   d1.id AS a_doc_id, d2.id AS b_doc_id,
                   c1.citation_span_start AS a_start, c1.citation_span_end AS a_end,
                   c2.citation_span_start AS b_start, c2.citation_span_end AS b_end
            FROM claims c1
            JOIN claims c2
              ON c2.id > c1.id AND c2.document_id != c1.document_id
            JOIN entity_mentions m1 ON m1.claim_id = c1.id
            JOIN entity_mentions m2 ON m2.claim_id = c2.id AND m2.entity_id = m1.entity_id
            JOIN documents d1 ON d1.id = c1.document_id
            JOIN documents d2 ON d2.id = c2.document_id
            LEFT JOIN eval_labels el
              ON (el.claim_a_id = c1.id AND el.claim_b_id = c2.id)
              OR (el.claim_a_id = c2.id AND el.claim_b_id = c1.id)
            WHERE el.id IS NULL
            GROUP BY c1.id, c2.id
            ORDER BY RANDOM()
            LIMIT ?
            """
        params: tuple[object, ...] = (limit,)
    elif pool == "near_dup":
        # Pairs the embedding stage ranked as very similar (same topic,
        # likely wording-level drift) but not yet judged/labeled.
        sql = """
            SELECT c1.id AS a_id, c2.id AS b_id,
                   d1.title AS a_doc, d2.title AS b_doc,
                   c1.claim_text AS a_text, c2.claim_text AS b_text,
                   d1.id AS a_doc_id, d2.id AS b_doc_id,
                   c1.citation_span_start AS a_start, c1.citation_span_end AS a_end,
                   c2.citation_span_start AS b_start, c2.citation_span_end AS b_end
            FROM candidate_pairs cp
            JOIN claims c1 ON c1.id = cp.claim_a_id
            JOIN claims c2 ON c2.id = cp.claim_b_id
            JOIN documents d1 ON d1.id = c1.document_id
            JOIN documents d2 ON d2.id = c2.document_id
            LEFT JOIN eval_labels el
              ON (el.claim_a_id = c1.id AND el.claim_b_id = c2.id)
              OR (el.claim_a_id = c2.id AND el.claim_b_id = c1.id)
            WHERE el.id IS NULL AND cp.similarity_score >= 0.85
            ORDER BY cp.similarity_score DESC
            LIMIT ?
            """
        params = (limit,)
    elif pool == "unrelated":
        sql = """
            SELECT c1.id AS a_id, c2.id AS b_id,
                   d1.title AS a_doc, d2.title AS b_doc,
                   c1.claim_text AS a_text, c2.claim_text AS b_text,
                   d1.id AS a_doc_id, d2.id AS b_doc_id,
                   c1.citation_span_start AS a_start, c1.citation_span_end AS a_end,
                   c2.citation_span_start AS b_start, c2.citation_span_end AS b_end
            FROM claims c1
            JOIN claims c2
              ON c2.id > c1.id AND c2.document_id != c1.document_id
            JOIN documents d1 ON d1.id = c1.document_id
            JOIN documents d2 ON d2.id = c2.document_id
            LEFT JOIN eval_labels el
              ON (el.claim_a_id = c1.id AND el.claim_b_id = c2.id)
              OR (el.claim_a_id = c2.id AND el.claim_b_id = c1.id)
            WHERE el.id IS NULL
              AND NOT EXISTS (
                SELECT 1
                FROM entity_mentions m1
                JOIN entity_mentions m2 ON m2.entity_id = m1.entity_id
                WHERE m1.claim_id = c1.id AND m2.claim_id = c2.id
              )
            ORDER BY RANDOM()
            LIMIT ?
            """
        params = (limit,)
    else:
        raise ValueError(f"unknown pool {pool!r}; expected one of {POOLS}")
    rows = conn.execute(sql, params).fetchall()
    return [
        PendingPair(
            a_id=r["a_id"],
            b_id=r["b_id"],
            a_doc=r["a_doc"],
            b_doc=r["b_doc"],
            a_text=r["a_text"],
            b_text=r["b_text"],
            a_doc_id=r["a_doc_id"],
            b_doc_id=r["b_doc_id"],
            a_start=r["a_start"],
            a_end=r["a_end"],
            b_start=r["b_start"],
            b_end=r["b_end"],
        )
        for r in rows
    ]


def label_pair(conn: sqlite3.Connection, pair: PendingPair, label: str, labeled_by: str) -> None:
    """Persist one label for a pair (insert_eval_label semantics)."""
    insert_eval_label(
        conn,
        claim_a_id=pair.a_id,
        claim_b_id=pair.b_id,
        doc_a_id=pair.a_doc_id,
        doc_b_id=pair.b_doc_id,
        span_a_start=pair.a_start,
        span_a_end=pair.a_end,
        span_b_start=pair.b_start,
        span_b_end=pair.b_end,
        label=label,
        labeled_by=labeled_by,
    )
    conn.commit()


def _print_pair(pair: PendingPair) -> None:
    print("=" * 72)
    print(f"A) [{pair.a_doc}]  {pair.a_text}")
    print(f"   span {pair.a_start}:{pair.a_end}")
    print(f"B) [{pair.b_doc}]  {pair.b_text}")
    print(f"   span {pair.b_start}:{pair.b_end}")
    print("-" * 72)


def run_cli(
    conn: sqlite3.Connection, limit: int = 20, labeled_by: str = "cli", pool: str = "entity"
) -> int:
    """Interactive loop. Returns number of labels applied."""
    done = 0
    while True:
        pairs = fetch_pending_pairs(conn, limit=limit, pool=pool)
        if not pairs:
            if pool == "near_dup":
                print(
                    "No high-similarity unlabeled pairs; run scripts/run_pipeline.py first."
                )
            else:
                print("No unlabeled pairs found. Run scripts/extract_claims.py first.")
            break
        print(f"\n{len(pairs)} pending pair(s) [pool: {pool}].")
        print(_LABEL_HINT)
        for pair in pairs:
            _print_pair(pair)
            while True:
                key = input("label> ").strip().lower()
                if key == "q":
                    return done
                if key == "k":
                    break
                if key in LABEL_KEYS:
                    label_pair(conn, pair, LABEL_KEYS[key], labeled_by)
                    done += 1
                    break
                print(f"unknown key {key!r}. {_LABEL_HINT}")
        again = input("show more pairs? [y/n] ").strip().lower()
        if again != "y":
            break
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description="Hand-label claim pairs.")
    ap.add_argument("--limit", type=int, default=20, help="pairs per screen")
    ap.add_argument("--labeled-by", default="cli", help="labeler identifier")
    ap.add_argument(
        "--pool", choices=POOLS, default="entity",
        help="which pair pool to label from (default entity)",
    )
    args = ap.parse_args()
    apply_migrations()
    conn = get_conn()
    try:
        done = run_cli(conn, limit=args.limit, labeled_by=args.labeled_by, pool=args.pool)
    finally:
        conn.close()
    print(f"labels applied: {done}")


if __name__ == "__main__":
    main()
