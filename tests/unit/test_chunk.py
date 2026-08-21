"""chunk_document: sentence alignment, offsets, long-sentence splitting."""

from __future__ import annotations

from adonis.extract.chunk import chunk_document, sentence_spans


def test_sentence_spans_offsets():
    text = "First sentence. Second one!\n\nThird?"
    spans = sentence_spans(text)
    assert len(spans) == 3
    for start, end, piece in spans:
        assert text[start:end] == piece
    # The sentencizer drops trailing whitespace, so the extension tiling makes
    # sentence 1 include the space before sentence 2.
    assert spans[0][2] == "First sentence. "
    assert spans[0][0] == 0 and spans[0][1] == 16


def test_chunk_single_small_document():
    text = "One sentence here. Another here."
    chunks = chunk_document(text, max_chars=1000)
    assert len(chunks) == 1
    assert chunks[0].start == 0
    assert chunks[0].end == len(text)
    assert chunks[0].text == text


def test_chunk_respects_max_chars_and_offsets():
    text = "Alpha sentence. " * 60  # ~900 chars
    chunks = chunk_document(text, max_chars=200)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= 200
        assert chunk.text == text[chunk.start : chunk.end]
    joined = "".join(c.text for c in chunks)
    assert joined == text


def test_chunk_splits_long_single_sentence():
    text = "Word " * 300  # one 1500-char sentence
    chunks = chunk_document(text, max_chars=200)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk.text) <= 200
        assert chunk.text == text[chunk.start : chunk.end]


def test_chunk_empty_text():
    assert chunk_document("   \n\n  ") == []


def test_chunk_blank_only_sentences_skipped():
    text = "Real sentence.\n\n\n\n"
    chunks = chunk_document(text, max_chars=100)
    assert len(chunks) == 1
    assert chunks[0].text == "Real sentence."
