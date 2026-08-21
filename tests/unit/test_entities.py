"""Entity mention extraction: offsets, dedupe, injectable model."""

from __future__ import annotations

from adonis.extract.entities import Mention, extract_mentions
from tests.fakes import FakeNER


def test_extract_mentions_offsets_shifted():
    ner = FakeNER(
        by_text={
            "Atlas ships": [
                {"text": "Atlas", "label": "PROJECT", "start": 0, "end": 5, "score": 0.9}
            ]
        }
    )
    mentions = extract_mentions("Atlas ships in March.", offset=120, model=ner)
    assert mentions == [
        Mention("Atlas", "PROJECT", 120, 125, 0.9)
    ]


def test_extract_mentions_no_hits():
    assert extract_mentions("nothing here.", model=FakeNER()) == []


def test_dedupe_overlapping_mentions_keeps_higher_score():
    ner = FakeNER(
        by_text={
            "Atlas Database": [
                {"text": "Atlas Database", "label": "PROJECT", "start": 0, "end": 14, "score": 0.8},
                {"text": "Atlas", "label": "PROJECT", "start": 0, "end": 5, "score": 0.95},
            ]
        }
    )
    mentions = extract_mentions("Atlas Database rules.", offset=0, model=ner)
    assert len(mentions) == 1
    assert mentions[0].text == "Atlas"


def test_dedupe_non_overlapping_keeps_both():
    ner = FakeNER(
        by_text={
            "Atlas is": [
                {"text": "Atlas", "label": "PROJECT", "start": 0, "end": 5, "score": 0.9},
                {"text": "Postgres", "label": "TECHNOLOGY", "start": 11, "end": 20, "score": 0.9},
            ]
        }
    )
    mentions = extract_mentions("Atlas is Postgres", offset=0, model=ner)
    assert len(mentions) == 2
