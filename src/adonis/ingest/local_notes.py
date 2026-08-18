"""Local notes parser (.md / .txt)."""

from __future__ import annotations

from pathlib import Path

from adonis.ingest.base import DocumentRecord
from adonis.normalize.text import normalize_text, strip_markdown

_MD_SUFFIXES = {".md", ".markdown"}


def parse_local_note(path: Path) -> DocumentRecord:
    """Read a local markdown or plain-text note into a DocumentRecord.

    Markdown files are stripped of formatting; .txt files are normalized only.
    """
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() in _MD_SUFFIXES:
        text = strip_markdown(text)
    else:
        text = normalize_text(text)
    return DocumentRecord(
        source="local",
        source_id=str(path),
        title=path.stem,
        path=str(path),
        format="md" if path.suffix.lower() in _MD_SUFFIXES else "txt",
        raw_text=text,
        metadata={"filename": path.name},
    )