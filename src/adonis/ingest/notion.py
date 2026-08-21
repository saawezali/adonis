"""Notion export (zip) parser.

A Notion export zip contains nested markdown files (one per page) plus CSV
database exports. Each .md and .csv entry becomes one document; attachments
(images, binaries) are ignored.
"""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

from adonis.ingest.base import DocumentRecord
from adonis.normalize.text import content_hash, normalize_text, strip_markdown

_MD_SUFFIXES = {".md", ".markdown"}
_CSV_SUFFIX = ".csv"


def parse_notion_export(zip_path: Path) -> list[DocumentRecord]:
    """Parse every markdown page and CSV database in a Notion export zip."""
    records: list[DocumentRecord] = []
    with zipfile.ZipFile(zip_path) as zf:
        for entry in sorted(zf.namelist()):
            if entry.endswith("/"):
                continue
            suffix = Path(entry).suffix.lower()
            if suffix in _MD_SUFFIXES:
                raw = zf.read(entry).decode("utf-8", errors="replace")
                records.append(_md_record(raw, entry))
            elif suffix == _CSV_SUFFIX:
                raw = zf.read(entry).decode("utf-8", errors="replace")
                records.append(_csv_record(raw, entry))
    return records


def _md_record(raw: str, entry: str) -> DocumentRecord:
    path = Path(entry)
    text = strip_markdown(raw)
    return DocumentRecord(
        source="notion",
        source_id=content_hash(text),
        title=path.stem,
        path=entry,
        format="md",
        raw_text=text,
        metadata={"export_entry": entry},
    )


def _csv_record(raw: str, entry: str) -> DocumentRecord:
    columns: list[str] = []
    row_count = 0
    try:
        rows = list(csv.reader(io.StringIO(raw)))
        if rows:
            columns = rows[0]
            row_count = len(rows) - 1
    except csv.Error:
        pass
    text = normalize_text(raw)
    return DocumentRecord(
        source="notion",
        source_id=content_hash(text),
        title=Path(entry).stem,
        path=entry,
        format="csv",
        raw_text=text,
        metadata={"export_entry": entry, "columns": columns, "row_count": row_count},
    )