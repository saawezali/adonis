"""Entity canonicalization: exact grouping, fuzzy merge, representative pick."""

from __future__ import annotations

from adonis.extract.canonicalize import ClaimMention, cluster_mentions, normalize_name
from adonis.extract.entities import Mention


def _cm(claim_id: str, text: str, label: str) -> ClaimMention:
    return ClaimMention(claim_id=claim_id, mention=Mention(text, label, 0, len(text)))


def test_normalize_name():
    assert normalize_name("  The Atlas  DB. ") == "atlas db"
    assert normalize_name("Atlas DB") == "atlas db"


def test_exact_groups_merge_into_single_cluster():
    mentions = [
        _cm("c1", "Atlas", "PROJECT"),
        _cm("c2", "atlas", "PROJECT"),
    ]
    clusters = cluster_mentions(mentions)
    assert len(clusters) == 1
    cluster = clusters[0]
    assert cluster.canonical_name == "Atlas"
    assert cluster.aliases == ["Atlas", "atlas"]
    assert cluster.label == "PROJECT"
    assert len(cluster.mentions) == 2


def test_different_labels_stay_separate_even_if_similar_name():
    mentions = [
        _cm("c1", "Atlas", "PROJECT"),
        _cm("c2", "Atlas", "PERSON"),
    ]
    clusters = cluster_mentions(mentions)
    assert len(clusters) == 2


def test_fuzzy_merge_close_names_same_label():
    mentions = [
        _cm("c1", "PostgreSQL", "TECHNOLOGY"),
        _cm("c2", "PostgresSQL", "TECHNOLOGY"),
    ]
    clusters = cluster_mentions(mentions, fuzzy_threshold=0.8)
    assert len(clusters) == 1
    assert clusters[0].canonical_name == "PostgreSQL"
    assert set(clusters[0].aliases) == {"PostgreSQL", "PostgresSQL"}


def test_distant_names_do_not_merge():
    mentions = [
        _cm("c1", "Postgres", "TECHNOLOGY"),
        _cm("c2", "MySQL", "TECHNOLOGY"),
    ]
    clusters = cluster_mentions(mentions, fuzzy_threshold=0.8)
    assert len(clusters) == 2


def test_representative_is_most_frequent():
    mentions = [
        _cm("c1", "Atlas", "PROJECT"),
        _cm("c2", "Atlas", "PROJECT"),
        _cm("c3", "atlas", "PROJECT"),
    ]
    clusters = cluster_mentions(mentions)
    assert clusters[0].canonical_name == "Atlas"


def test_blank_mentions_skipped():
    assert cluster_mentions([_cm("c1", "  ", "OTHER")]) == []