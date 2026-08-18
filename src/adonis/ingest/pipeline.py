"""Corpus ingestion orchestrator.

Walks a corpus directory, dispatches each file to the right adapter by
extension, and writes DocumentRecords into the store. Tracks per-file outcome
so parse-failure rate is visible (per PLAN.md acceptance for M1).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from pathlib import Path

from adonis.db import insert_document
from adonis.ingest.base import DocumentRecord
from adonis.ingest.gdrive import parse_docx, parse_gdoc
from adonis.ingest.local_notes import parse_local_note
from adonis.ingest.notion import parse_notion_export

_LOCAL_SUFFIXES = {".md", ".markdown", ".txt"}
_NOTION_SUFFIXES = {".zip"}


@dataclass
class IngestStats:
    """Outcome counts for one ingestion run."""

    files_seen: int = 0
    inserted: int = 0
    duplicates: int = 0
    failed: int = 0
    skipped: int = 0
    ignored: int = 0
    doc_count: int = 0
    errors: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def ingest_corpus(corpus_dir: Path, conn: sqlite3.Connection) -> IngestStats:
    """Ingest every supported file under corpus_dir into the store."""
    stats = IngestStats()
    files = sorted(p for p in corpus_dir.rglob("*") if p.is_file())
    for path in files:
        stats.files_seen += 1
        _ingest_file(path, conn, stats)
    return stats


def _ingest_file(path: Path, conn: sqlite3.Connection, stats: IngestStats) -> None:
    suffix = path.suffix.lower()
    try:
        if suffix in _LOCAL_SUFFIXES:
            _store(conn, parse_local_note(path), stats)
        elif suffix in _NOTION_SUFFIXES:
            if not _looks_like_notion(path):
                stats.ignored += 1
                stats.notes.append(f"{path}: zip does not look like a Notion export")
                return
            for record in parse_notion_export(path):
                _store(conn, record, stats)
        elif suffix == ".docx":
            _store(conn, parse_docx(path), stats)
        elif suffix == ".gdoc":
            if parse_gdoc(path) is None:
                stats.skipped += 1
                stats.notes.append(f"{path}: .gdoc shortcut has no offline content")
        else:
            stats.ignored += 1
    # One bad file must not kill the run; it is counted and reported instead.
    except Exception as exc:  # noqa: BLE001
        stats.failed += 1
        stats.errors.append(f"{path}: {exc!r}")


def _store(conn: sqlite3.Connection, record: DocumentRecord, stats: IngestStats) -> None:
    if insert_document(conn, record):
        stats.inserted += 1
    else:
        stats.duplicates += 1


def _looks_like_notion(path: Path) -> bool:
    """A Notion export zip contains markdown pages and/or CSV databases."""
    import zipfile

    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
    return any(n.endswith((".md", ".markdown", ".csv")) for n in names)