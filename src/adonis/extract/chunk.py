"""Document chunking for claim extraction.

Per PLAN.md M2: chunk documents into sentence-aligned pieces small enough
for one extraction call, keeping absolute character offsets into the
normalized raw_text. Sentences come from a rule-based spaCy sentencizer
(no model download required); over-long sentences are split at word/space
boundaries near the cap so offsets always stay valid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import spacy

from adonis.config import get_settings


@dataclass(frozen=True)
class Chunk:
    """A slice of a document's normalized text (absolute char offsets)."""

    start: int
    end: int
    text: str


_nlp: Any = None  # spaCy ships no stubs; typed as Any.


def _get_nlp() -> Any:
    """Lazily build the spaCy blank-English sentencizer (no model download)."""
    global _nlp
    if _nlp is None:
        nlp = spacy.blank("en")
        nlp.add_pipe("sentencizer")
        _nlp = nlp
    return _nlp


def sentence_spans(text: str) -> list[tuple[int, int, str]]:
    """Return (start, end, text) for every non-blank sentence in `text`.

    The spaCy sentencizer drops trailing whitespace, so each sentence is
    extended to the start of the next one (or end of text) to keep the spans
    tiling the document without gaps.
    """
    doc = _get_nlp()(text)
    raw = [sent.start_char for sent in doc.sents] + [len(text)]
    spans: list[tuple[int, int, str]] = []
    for i in range(len(raw) - 1):
        start, end = raw[i], raw[i + 1]
        piece = text[start:end]
        if piece.strip():
            spans.append((start, end, piece))
    return spans


def _split_long_sentence(
    sentence: str, start: int, max_chars: int
) -> list[tuple[int, int, str]]:
    """Split a sentence that exceeds max_chars at word boundaries.

    Returns (start, end, text) pieces; offsets are relative to the document.
    """
    if len(sentence) <= max_chars:
        return [(start, start + len(sentence), sentence)]
    pieces: list[tuple[int, int, str]] = []
    offset = 0
    while offset < len(sentence):
        end = min(offset + max_chars, len(sentence))
        if end < len(sentence):
            last_space = sentence.rfind(" ", offset, end)
            if last_space > offset:
                end = last_space
        pieces.append(
            (start + offset, start + end, sentence[offset:end])
        )
        offset = end
    return pieces


def chunk_document(text: str, max_chars: int | None = None) -> list[Chunk]:
    """Split `text` into sentence-aligned chunks not exceeding max_chars.

    A single over-long sentence is split at word boundaries; a single
    over-long word forms its own chunk. Chunks never contain blank text.
    """
    max_chars = max_chars or get_settings().chunk_max_chars
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1")
    if not text.strip():
        return []
    chunks: list[Chunk] = []
    current: list[tuple[int, int]] = []
    current_len = 0
    for start, end, sentence in sentence_spans(text):
        for s, e, piece in _split_long_sentence(sentence, start, max_chars):
            if current and current_len + len(piece) > max_chars:
                chunks.append(
                    Chunk(
                        current[0][0],
                        current[-1][1],
                        text[current[0][0] : current[-1][1]],
                    )
                )
                current = []
                current_len = 0
            current.append((s, e))
            current_len += len(piece)
    if current:
        chunks.append(
            Chunk(current[0][0], current[-1][1], text[current[0][0] : current[-1][1]])
        )
    return chunks
