"""Ingest a corpus directory into the document store.

M1 entry point. Walks --in, dispatches each file to the appropriate adapter
(markdown/txt notes, Notion export zips, .docx/.gdoc Drive exports),
normalizes text, and inserts rows. Prints summary stats including the
parse-failure rate.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from adonis.db import apply_migrations, get_conn
from adonis.ingest.pipeline import IngestStats, ingest_corpus


def run(input_dir: Path) -> IngestStats:
    """Apply migrations, ingest, and return stats (with a docs count)."""
    apply_migrations()
    conn = get_conn()
    try:
        stats = ingest_corpus(input_dir, conn)
        stats.doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
    finally:
        conn.close()
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest a corpus directory.")
    ap.add_argument("--in", dest="input_dir", required=True, help="corpus directory")
    ap.add_argument(
        "--db",
        action="store_true",
        help="also print documents currently in the store",
    )
    args = ap.parse_args()

    corpus = Path(args.input_dir)
    if not corpus.is_dir():
        raise SystemExit(f"not a directory: {corpus}")

    stats = run(corpus)

    print(f"files seen:     {stats.files_seen}")
    print(f"inserted:       {stats.inserted}")
    print(f"duplicates:     {stats.duplicates}")
    print(f"skipped:        {stats.skipped}")
    print(f"ignored:        {stats.ignored}")
    print(f"failed:         {stats.failed}")
    if stats.files_seen:
        pct = 100.0 * stats.failed / stats.files_seen
        print(f"parse failure:  {pct:.1f}% ({stats.failed}/{stats.files_seen})")
    if args.db:
        print(f"documents in store: {stats.doc_count}")
    for note in stats.notes:
        print(f"  note: {note}")
    for err in stats.errors:
        print(f"  error: {err}")


if __name__ == "__main__":
    main()