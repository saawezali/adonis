"""Text normalization utilities.

Standardizes raw extracted text before storage so that:
  - character offsets (citation spans) are computed against a stable form
  - content_hash is comparable across re-ingests
  - markdown is reduced to readable plain text for embedding and span matching

All functions are deterministic and dependency-free.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

_FENCED_CODE = re.compile(r"```[a-zA-Z0-9_-]*\n(.*?)```", re.DOTALL)
_IMAGE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_REF_LINK = re.compile(r"\[([^\]]+)\]\[[^\]]*\]")
_INLINE_CODE = re.compile(r"`([^`]+)`")
_FOOTNOTE = re.compile(r"\[\^\d+\]")
_STRIKE = re.compile(r"~~([^~]+)~~")
_BOLD = re.compile(r"\*\*([^*]+)\*\*")
_UNDERLINE = re.compile(r"__([^_]+)__")
_ITALIC = re.compile(r"(?<!\*)\*([^*\n]+)\*(?!\*)")
_HEADER = re.compile(r"^#{1,6}\s*", re.MULTILINE)
_QUOTE = re.compile(r"^\s{0,3}>\s?", re.MULTILINE)
_BULLET = re.compile(r"^\s*[-+]\s+", re.MULTILINE)
_ASTERISK_BULLET = re.compile(r"^\s*\*\s+", re.MULTILINE)
_NUMBERED = re.compile(r"^\s*(\d+)\.\s+", re.MULTILINE)
_CHECKBOX = re.compile(r"^\s*(?:[-+*]\s+)?\[[ xX]\]\s*", re.MULTILINE)
_SEPARATOR = re.compile(r"^[ \t]*[\|:\-][\|:\- ]*$", re.MULTILINE)
_PIPE_LEAD = re.compile(r"^\s*\|\s*", re.MULTILINE)
_PIPE_TAIL = re.compile(r"\|\s*$", re.MULTILINE)
_TRAILING_SPACE = re.compile(r" +$", re.MULTILINE)
_MULTI_BLANK = re.compile(r"\n{3,}")


def normalize_text(text: str) -> str:
    """Canonical form: NFC unicode, LF line endings, no zero-width chars,
    no trailing whitespace, no runs of 3+ blank lines, stripped ends."""
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
    text = _TRAILING_SPACE.sub("", text)
    text = _MULTI_BLANK.sub("\n\n", text)
    return text.strip()


def strip_markdown(text: str) -> str:
    """Reduce markdown to readable plain text, deterministically.

    Handles: code fences (keep inner text), images (drop), links (keep label),
    inline code / emphasis / strikethrough (keep content), headers, quotes,
    lists, checkboxes, and markdown table structure.
    """
    text = normalize_text(text)
    text = _FENCED_CODE.sub(lambda m: m.group(1), text)
    text = _IMAGE.sub("", text)
    text = _LINK.sub(r"\1", text)
    text = _REF_LINK.sub(r"\1", text)
    text = _INLINE_CODE.sub(r"\1", text)
    text = _FOOTNOTE.sub("", text)
    text = _STRIKE.sub(r"\1", text)
    text = _BOLD.sub(r"\1", text)
    text = _UNDERLINE.sub(r"\1", text)
    text = _ITALIC.sub(r"\1", text)
    text = _HEADER.sub("", text)
    text = _QUOTE.sub("", text)
    text = _CHECKBOX.sub("- ", text)
    text = _BULLET.sub("- ", text)
    text = _ASTERISK_BULLET.sub("- ", text)
    text = _NUMBERED.sub(r"\1. ", text)
    text = _SEPARATOR.sub("", text)
    text = _PIPE_LEAD.sub("", text)
    text = _PIPE_TAIL.sub("", text)
    return normalize_text(text)


def content_hash(text: str) -> str:
    """Stable fingerprint of normalized text, used for dedup across re-ingests."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
