"""Shared types for ingest adapters."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SUPPORTED_SOURCES = ("notion", "gdrive", "local")


@dataclass
class DocumentRecord:
    """A normalized, parseable document ready for the documents table."""

    source: str
    title: str
    path: str
    format: str
    raw_text: str
    source_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
