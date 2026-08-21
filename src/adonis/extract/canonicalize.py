"""Entity canonicalization: mention strings -> entity clusters.

Clustering is deterministic and local: exact normalized-string groups first,
then greedy merge of clusters whose canonical names are fuzzy-close (rapidfuzz)
and share the NER label. LLM-assisted resolution of ambiguous clusters is
gated behind Settings.canonicalize_llm_merge (off in M2).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from rapidfuzz import fuzz

from adonis.config import get_settings
from adonis.extract.entities import Mention


@dataclass(frozen=True)
class ClaimMention:
    """A mention tied to the claim it was found in."""

    claim_id: str
    mention: Mention


@dataclass
class EntityCluster:
    """One canonical entity: a representative name, aliases, and its mentions."""

    canonical_name: str
    label: str
    aliases: list[str] = field(default_factory=list)
    mentions: list[ClaimMention] = field(default_factory=list)


def normalize_name(text: str) -> str:
    """Canonical key for clustering: lowercase, article-stripped, collapsed."""
    name = re.sub(r"\s+", " ", text).strip().lower()
    name = re.sub(r"^the\s+", "", name)
    return name.rstrip(".")


def cluster_mentions(
    mentions: list[ClaimMention], fuzzy_threshold: float | None = None
) -> list[EntityCluster]:
    """Cluster mentions into entities.

    Exact normalized-name groups form base clusters; clusters are then merged
    greedily when their canonical names are fuzzy-close (ratio >= threshold)
    and share the dominant label. The representative name is the most frequent
    raw mention string (original casing); aliases are unique raw strings.
    """
    if fuzzy_threshold is None:
        fuzzy_threshold = get_settings().canonicalize_fuzzy_threshold

    # Partition by (label, normalized name): mentions with different NER labels
    # never share an entity, no matter how similar the strings look.
    groups: dict[tuple[str, str], list[ClaimMention]] = {}
    for cm in mentions:
        key = normalize_name(cm.mention.text)
        if not key:
            continue
        groups.setdefault((cm.mention.label, key), []).append(cm)

    clusters = [
        EntityCluster(
            canonical_name=_representative(records),
            label=label,
            aliases=sorted({cm.mention.text for cm in records}),
            mentions=records,
        )
        for (label, _key), records in groups.items()
    ]

    # Greedy fuzzy merge (same label only).
    merged = True
    while merged:
        merged = False
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                a, b = clusters[i], clusters[j]
                if a.label != b.label:
                    continue
                ratio = fuzz.ratio(a.canonical_name, b.canonical_name) / 100.0
                if ratio >= fuzzy_threshold:
                    combined = a.mentions + b.mentions
                    clusters[i] = EntityCluster(
                        canonical_name=_representative(combined),
                        label=a.label,
                        aliases=sorted({cm.mention.text for cm in combined}),
                        mentions=combined,
                    )
                    del clusters[j]
                    merged = True
                    break
            if merged:
                break

    clusters.sort(key=lambda c: (-len(c.mentions), c.canonical_name))
    return clusters


def _representative(records: list[ClaimMention]) -> str:
    return Counter(cm.mention.text for cm in records).most_common(1)[0][0]
