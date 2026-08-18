"""Normalization tests: unicode, line endings, markdown stripping, hashing."""

from __future__ import annotations

from adonis.normalize.text import content_hash, normalize_text, strip_markdown


def test_normalize_nfc_and_line_endings():
    text = "caf\xe9\r\nR\u0301e\u0301sume\u0301\r\nline2"
    out = normalize_text(text)
    assert out == "caf\u00e9\n\u0154\u00e9sum\u00e9\nline2"


def test_normalize_strips_zero_width_and_trailing_ws():
    out = normalize_text("a \ufeffb  \nc  \n\n\n\nd")
    assert out == "a b\nc\n\nd"


def test_normalize_is_deterministic():
    a = normalize_text(" Hello \r\nworld \r\n\r\n\r\n tail  \ufeff")
    b = normalize_text("Hello\nworld\n\n tail")
    assert a == b


def test_strip_markdown_common_constructs():
    md = (
        "# Title\n"
        "\n"
        "Welcome **home**! Read [the doc](https://x.dev) and `code`.\n"
        "\n"
        "![image](img.png)\n"
        "\n"
        "- first item\n"
        "- [x] done task\n"
        "- [ ] open task\n"
        "\n"
        "1. one\n"
        "2. two\n"
        "\n"
        "> quoted thought\n"
        "\n"
        "| col a | col b |\n"
        "| ----- | ----- |\n"
        "| 1     | 2     |\n"
        "\n"
        "```python\n"
        "print('keep me')\n"
        "```\n"
    )
    out = strip_markdown(md)
    assert "Title" in out
    assert "**home**" not in out and "Welcome home!" in out
    assert "the doc" in out
    assert "code" in out
    assert "img.png" not in out
    assert "- first item" in out
    assert "- done task" in out
    assert "- open task" in out
    assert "1. one" in out and "2. two" in out
    assert "> quoted" not in out and "quoted thought" in out
    assert "col a" in out and "1     |     2" not in out
    assert "'keep me'" in out
    assert "| ----- |" not in out


def test_content_hash_stable_and_distinct():
    a = content_hash("same text")
    b = content_hash("same text")
    c = content_hash("different")
    assert a == b
    assert a != c