"""Measure candidate recall against human eval labels.

For every labeled contradiction-ish pair (eval_labels), checks whether the
pair was materialized as a candidate and where it ranks by combined_score
among all candidates. Prints recall@K plus the judge's flag coverage.

Run: python scripts/measure_recall.py [--limit N]
"""

from __future__ import annotations

import argparse

from adonis.db import get_conn, load_labeled_pairs

_FLAG_LABELS = {"genuine_contradiction", "superseded_by_time", "different_scope", "ambiguous"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Recall@K over eval_labels.")
    ap.add_argument("--limit", type=int, default=None, help="cap #candidates considered")
    args = ap.parse_args()

    conn = get_conn()
    try:
        labeled = load_labeled_pairs(conn)
        need: set[frozenset[str]] = {frozenset((r["claim_a_id"], r["claim_b_id"])) for r in labeled}
        n = len(need)
        print(f"labeled pairs: {n}")
        if n == 0:
            print("no eval labels; run `python -m adonis.cli.label_pairs` first")
            return

        rows = conn.execute(
            "SELECT claim_a_id, claim_b_id, combined_score FROM candidate_pairs"
            " ORDER BY combined_score DESC" + (" LIMIT ?" if args.limit else ""),
            () if args.limit is None else (args.limit,),
        ).fetchall()
        total = conn.execute("SELECT COUNT(*) FROM candidate_pairs").fetchone()[0]

        found: dict[frozenset[str], int] = {}
        for rank, row in enumerate(rows, 1):
            key = frozenset((row["claim_a_id"], row["claim_b_id"]))
            if key in need and key not in found:
                found[key] = rank

        print(f"candidates: {total}")
        print(f"covered:    {len(found)}/{n} ({len(found) / n:.0%})")
        for k in (5, 10, 20, 50):
            hits = sum(1 for rank in found.values() if rank <= k)
            print(f"recall@{k:<3} {hits}/{n} ({hits / n:.0%})")

        missed = need - found.keys()
        if missed:
            print("\nmissed pairs:")
            for key in sorted(missed):
                a, b = tuple(key)
                row = conn.execute(
                    "SELECT c1.claim_text AS ta, d1.title AS tit_a,"
                    " c2.claim_text AS tb, d2.title AS tit_b"
                    " FROM claims c1"
                    " JOIN documents d1 ON d1.id = c1.document_id"
                    " JOIN claims c2 ON c2.id = ?"
                    " JOIN documents d2 ON d2.id = c2.document_id"
                    " WHERE c1.id = ?",
                    (b, a),
                ).fetchone()
                if row is not None:
                    print(f"  [{row['tit_a']}] {row['ta'].strip()[:60]!r}")
                    print(f"  [{row['tit_b']}] {row['tb'].strip()[:60]!r}")

        flagged = 0
        for key in need:
            a, b = tuple(key)
            hit = conn.execute(
                "SELECT jo.label FROM judge_outputs jo"
                " JOIN candidate_pairs cp ON cp.id = jo.candidate_pair_id"
                " WHERE (cp.claim_a_id = ? AND cp.claim_b_id = ?)"
                "    OR (cp.claim_a_id = ? AND cp.claim_b_id = ?)",
                (a, b, b, a),
            ).fetchone()
            if hit is not None and hit["label"] in _FLAG_LABELS:
                flagged += 1
        print(f"judge-flagged: {flagged}/{n} ({flagged / n:.0%})")
    finally:
        conn.close()


if __name__ == "__main__":
    main()